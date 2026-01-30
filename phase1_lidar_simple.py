import sys
import os
import json
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, 'mast3r')
sys.path.insert(0, 'mast3r/dust3r')
sys.path.insert(0, 'Fin3R')

from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from vggt.models.renorm_lora import get_renormalized_peft_model
from peft import LoraConfig

from calibration_results import scenename_to_calibration



def load_lidar_points(lidar_path: Path) -> np.ndarray:
    pts = []
    with open(lidar_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _as_mat3(flat9) -> np.ndarray:
    return np.array(flat9, dtype=np.float64).reshape(3, 3)


def _as_vec3(flat3) -> np.ndarray:
    return np.array(flat3, dtype=np.float64).reshape(3,)


def _infer_scene_name_from_folder(scene_folder_name: str) -> str:
    if "_" in scene_folder_name:
        return scene_folder_name.split("_", 1)[1]
    return scene_folder_name


def project_lidar_to_image_using_calib_auto(
    lidar_xyz: np.ndarray,          # (N,3) points in LiDAR frame
    calib: dict,                    # calib dict from calibration_results.py
    image_width: int,
    image_height: int,
    max_depth: float = 200.0,
):
    """
    Project LiDAR points into camera image using camera<->LiDAR calibration (R,t).
    Auto-chooses whether (R,t) is LiDAR->Cam or Cam->LiDAR by selecting the direction
    that produces more in-image valid projections.

    Returns:
      depth_map: (H,W) float32
      valid_mask: (H,W) bool
      info: dict with debug counts
    """
    if lidar_xyz.shape[0] == 0:
        return None, None, {"message": "Empty LiDAR point cloud."}

    # Intrinsics from calibration file (preferred) OR can use metadata P3x3 if you want.
    K = _as_mat3(calib["camera_intrinsic"])
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    R = _as_mat3(calib["camera_ext_R"])
    t = _as_vec3(calib["camera_ext_t"])

    # Candidate A: assume LiDAR->Cam
    # p_cam = R * p_lidar + t
    p_cam_A = (R @ lidar_xyz.T).T + t[None, :]

    # Candidate B: assume Cam->LiDAR, invert it
    # if p_lidar = R * p_cam + t  -> p_cam = R^T * (p_lidar - t)
    p_cam_B = (R.T @ (lidar_xyz - t[None, :]).T).T

    def score_and_extract(p_cam: np.ndarray):
        Z = p_cam[:, 2]
        valid_z = Z > 1e-6
        p = p_cam[valid_z]
        if p.shape[0] == 0:
            return 0, None, None, None

        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        in_img = (
            (u >= 0) & (u < image_width) &
            (v >= 0) & (v < image_height) &
            (Z > 0) & (Z < max_depth)
        )
        cnt = int(in_img.sum())
        if cnt == 0:
            return 0, None, None, None
        return cnt, u[in_img], v[in_img], Z[in_img]

    cntA, uA, vA, zA = score_and_extract(p_cam_A)
    cntB, uB, vB, zB = score_and_extract(p_cam_B)

    if cntA == 0 and cntB == 0:
        return None, None, {
            "chosen": None,
            "count_in_image_A": cntA,
            "count_in_image_B": cntB,
            "message": "No points projected into image using either direction. Likely axis convention mismatch or wrong calibration/image size."
        }

    if cntB > cntA:
        chosen = "B (R,t was Cam->LiDAR; inverted to LiDAR->Cam)"
        u, v, z = uB, vB, zB
        chosen_cnt = cntB
    else:
        chosen = "A (R,t was LiDAR->Cam directly)"
        u, v, z = uA, vA, zA
        chosen_cnt = cntA

    # Rasterize depth map (min depth per pixel)
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    valid_mask = np.zeros((image_height, image_width), dtype=bool)

    u_int = np.rint(u).astype(np.int32)
    v_int = np.rint(v).astype(np.int32)

    for i in range(len(z)):
        x = int(u_int[i])
        y = int(v_int[i])
        if 0 <= x < image_width and 0 <= y < image_height:
            d = float(z[i])
            if (not valid_mask[y, x]) or (d < depth_map[y, x]):
                depth_map[y, x] = d
                valid_mask[y, x] = True

    return depth_map, valid_mask, {
        "chosen": chosen,
        "count_in_image_A": cntA,
        "count_in_image_B": cntB,
        "valid_pixels": int(valid_mask.sum()),
        "chosen_count": int(chosen_cnt),
        "K_used": K.tolist(),
    }


def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray):
    """Compute standard depth metrics with median scaling."""
    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]

    if len(pred) < 100:
        return None

    scale = np.median(gt) / (np.median(pred) + 1e-12)
    pred_scaled = pred * scale

    pred_scaled = np.clip(pred_scaled, 1e-3, 100)
    gt = np.clip(gt, 1e-3, 100)

    thresh = np.maximum(gt / pred_scaled, pred_scaled / gt)
    delta1 = (thresh < 1.25).mean() * 100
    delta2 = (thresh < 1.25 ** 2).mean() * 100
    delta3 = (thresh < 1.25 ** 3).mean() * 100

    abs_rel = np.mean(np.abs(gt - pred_scaled) / gt)
    sq_rel = np.mean(((gt - pred_scaled) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred_scaled) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred_scaled)) ** 2))

    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'scale': float(scale),
        'num_valid': int(len(pred)),
    }


def evaluate_single_frame(img_path: Path, lidar_path: Path, metadata: dict, calib: dict, model, device='cuda'):
    """Evaluate a single frame using LiDAR->camera projection via calibration_results.py."""
    images = load_images([str(img_path)], size=512)
    actual_h, actual_w = images[0]['true_shape'][0]

    with torch.no_grad():
        img_pair = [images[0], images[0]]
        output = inference([tuple(img_pair)], model, device, batch_size=1, verbose=False)

    # Pred depth from MASt3R output
    pred_depth = output['pred1']['pts3d'][..., 2].squeeze().detach().cpu().numpy()

    lidar_points = load_lidar_points(lidar_path)

    # Use original image dimensions from metadata
    orig_w, orig_h = int(metadata['Width']), int(metadata['Height'])

    gt_depth_orig, valid_mask_orig, info = project_lidar_to_image_using_calib_auto(
        lidar_points, calib, orig_w, orig_h, max_depth=200.0
    )
    if gt_depth_orig is None:
        return None

    # Resize GT to match prediction size
    from scipy.ndimage import zoom
    scale_h = actual_h / orig_h
    scale_w = actual_w / orig_w

    gt_depth = zoom(gt_depth_orig, (scale_h, scale_w), order=1)
    valid_mask = zoom(valid_mask_orig.astype(np.float32), (scale_h, scale_w), order=0) > 0.5

    return compute_depth_metrics(pred_depth, gt_depth, valid_mask)


def main():
    print("Phase 1: LiDAR Ground Truth Evaluation (using camera<->LiDAR extrinsic)")

    DATASET_PATH = "/media/jitesh/Extreme SSD/fin3r-mast3r_analysis/datasets/UAVScenes/interval5_CAM_LIDAR"
    SCENE_FOLDER = "interval5_AMtown01"
    MAST3R_CKPT = "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    FIN3R_LORA = "Fin3R/checkpoints/mast3r_lora.pth"
    OUTPUT_DIR = "results/phase1_lidar_simple_calib"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    scene_path = Path(DATASET_PATH) / SCENE_FOLDER
    metadata_path = scene_path / "sampleinfos_interpolated.json"

    # infer scene name and fetch calibration
    scene_name = _infer_scene_name_from_folder(SCENE_FOLDER)
    if scene_name not in scenename_to_calibration:
        print(f" Could not find calibration for scene '{scene_name}' in scenename_to_calibration.")
        print("Available keys example:", list(scenename_to_calibration.keys())[:10])
        return
    calib = scenename_to_calibration[scene_name]
    print(f"Using calibration for scene: {scene_name}")

    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    # Find frames with LiDAR
    print("\nFinding frames with LiDAR")
    eval_frames = []
    for idx, frame in enumerate(tqdm(metadata_list[:300], desc="Scanning")):
        img_name = frame.get('OriginalImageName', None)
        if img_name is None:
            continue

        img_path = scene_path / "interval5_CAM" / img_name
        if not img_path.exists():
            continue

        img_timestamp = img_name.replace('.jpg', '')
        lidar_pattern = scene_path / "interval5_LIDAR" / f"image{img_timestamp}_lidar*.txt"
        matching = glob.glob(str(lidar_pattern))
        if matching:
            eval_frames.append({
                'idx': idx,
                'img_path': img_path,
                'lidar_path': Path(matching[0]),
                'metadata': frame
            })
            if len(eval_frames) >= 200:
                break

    print(f"✓ Found {len(eval_frames)} frames with LiDAR")
    if len(eval_frames) == 0:
        print("✗ No frames with LiDAR found!")
        return

    # Quick projection sanity check on first frame
    print("\n" + "=" * 80)
    print("Testing LiDAR->image projection on first frame...")
    print("=" * 80)

    test_frame = eval_frames[0]
    lidar_points = load_lidar_points(test_frame['lidar_path'])
    orig_w = int(test_frame['metadata']['Width'])
    orig_h = int(test_frame['metadata']['Height'])

    print(f"LiDAR points: {len(lidar_points)}")
    if len(lidar_points) > 0:
        print(f"LiDAR range: X[{lidar_points[:,0].min():.2f}, {lidar_points[:,0].max():.2f}], "
              f"Y[{lidar_points[:,1].min():.2f}, {lidar_points[:,1].max():.2f}], "
              f"Z[{lidar_points[:,2].min():.2f}, {lidar_points[:,2].max():.2f}]")

    gt_depth, valid_mask, info = project_lidar_to_image_using_calib_auto(
        lidar_points, calib, orig_w, orig_h, max_depth=200.0
    )
    print("Projection debug:", info)

    if gt_depth is None:
        print("✗ Projection failed using calibration_results.py.")
        print("Most likely remaining issue: axis convention mismatch (LiDAR frame axes differ).")
        print("Next fix would be to add a static axis-swap between LiDAR and camera frames.")
        return

    print("✓ Projection successful!")
    print(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100.0*valid_mask.sum()/valid_mask.size:.6f}%)")
    print(f"  Depth range: [{gt_depth[valid_mask].min():.2f}, {gt_depth[valid_mask].max():.2f}]")

    # Run full evaluation
    print("\n" + "=" * 80)
    print("Running full evaluation...")
    print("=" * 80)

    # Vanilla
    print("\nVanilla MASt3R...")
    vanilla_model = AsymmetricMASt3R.from_pretrained(MAST3R_CKPT).to(device)
    vanilla_model.eval()

    vanilla_results = []
    for frame in tqdm(eval_frames, desc="Vanilla"):
        m = evaluate_single_frame(
            frame['img_path'], frame['lidar_path'], frame['metadata'], calib,
            vanilla_model, device=device
        )
        if m is not None:
            vanilla_results.append(m)

    del vanilla_model
    torch.cuda.empty_cache()

    # Fin3R
    print("\nFin3R-MASt3R...")
    fin3r_model = AsymmetricMASt3R.from_pretrained(MAST3R_CKPT).to(device)
    fin3r_model.eval()

    lora_config = LoraConfig(r=8, lora_alpha=8, target_modules=["qkv"], lora_dropout=0.1)
    for block in fin3r_model.enc_blocks:
        block.attn = get_renormalized_peft_model(block.attn, lora_config)

    state_dict = torch.load(FIN3R_LORA, map_location=device, weights_only=False)
    fin3r_model.load_state_dict(state_dict, strict=False)

    fin3r_results = []
    for frame in tqdm(eval_frames, desc="Fin3R"):
        m = evaluate_single_frame(
            frame['img_path'], frame['lidar_path'], frame['metadata'], calib,
            fin3r_model, device=device
        )
        if m is not None:
            fin3r_results.append(m)

    del fin3r_model
    torch.cuda.empty_cache()

    # Print results
    if len(vanilla_results) == 0 or len(fin3r_results) == 0:
        print("\n✗ No valid results (too few valid pixels after projection).")
        print(f"Vanilla frames valid: {len(vanilla_results)} / {len(eval_frames)}")
        print(f"Fin3R frames valid:   {len(fin3r_results)} / {len(eval_frames)}")
        return

    print("\n" + "=" * 80)
    print("RESULTS (median-scaled)")
    print("=" * 80)

    metric_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3']
    v_means = {m: float(np.mean([r[m] for r in vanilla_results])) for m in metric_names}
    f_means = {m: float(np.mean([r[m] for r in fin3r_results])) for m in metric_names}

    print("\n| Method         | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |")
    print("|" + "-" * 90 + "|")
    print(f"| Vanilla MASt3R | {v_means['abs_rel']:.4f}   | {v_means['sq_rel']:.4f}  | {v_means['rmse']:.3f} | "
          f"{v_means['rmse_log']:.4f}    | {v_means['delta1']:.2f}%    | {v_means['delta2']:.2f}%     | {v_means['delta3']:.2f}%     |")
    print(f"| Fin3R-MASt3R   | {f_means['abs_rel']:.4f}   | {f_means['sq_rel']:.4f}  | {f_means['rmse']:.3f} | "
          f"{f_means['rmse_log']:.4f}    | {f_means['delta1']:.2f}%    | {f_means['delta2']:.2f}%     | {f_means['delta3']:.2f}%     |")

    print(f"\nEvaluated on {len(vanilla_results)} frames (out of {len(eval_frames)} with LiDAR)")

    out_path = Path(OUTPUT_DIR) / "phase1_depth_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"vanilla": vanilla_results, "fin3r": fin3r_results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
