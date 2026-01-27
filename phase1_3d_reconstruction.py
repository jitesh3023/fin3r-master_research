# Phase 1: 3D Reconstruction Comparison

import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, 'mast3r')
sys.path.insert(0, 'mast3r/dust3r')
sys.path.insert(0, 'Fin3R')

from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from vggt.models.renorm_lora import get_renormalized_peft_model
from peft import LoraConfig


def load_mast3r_model(checkpoint_path, device='cuda'):
    model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return model


def apply_fin3r_lora(model, lora_path, device='cuda'):
    lora_config = LoraConfig(r=8, lora_alpha=8, target_modules=["qkv"], lora_dropout=0.1)
    for block in model.enc_blocks:
        block.attn = get_renormalized_peft_model(block.attn, lora_config)
    state_dict = torch.load(lora_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model


def extract_point_cloud(output, images, conf_threshold=1.0):
    """
    Extract colored 3D point cloud from MASt3R output
    """
    # Get 3D points and confidence
    pts3d = output['pred1']['pts3d'].squeeze().cpu().numpy()  # H x W x 3
    conf = output['pred1']['conf'].squeeze().cpu().numpy()    # H x W
    
    # Get image colors
    img = images[0]['img'].squeeze().permute(1, 2, 0).cpu().numpy()  # H x W x 3
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    
    # Flatten
    H, W = pts3d.shape[:2]
    pts3d_flat = pts3d.reshape(-1, 3)
    colors_flat = img.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    
    # Filter by confidence
    valid = conf_flat >= conf_threshold
    pts3d_filtered = pts3d_flat[valid]
    colors_filtered = colors_flat[valid]
    
    print(f"  Points: {len(pts3d_filtered):,} / {len(pts3d_flat):,} ({100*len(pts3d_filtered)/len(pts3d_flat):.1f}%)")
    
    return pts3d_filtered, colors_filtered


def visualize_3d_comparison(vanilla_pts, vanilla_colors, fin3r_pts, fin3r_colors, 
                            vanilla_img, save_dir):
    """Create 3D point cloud comparison visualization"""
    
    print("\n=== Creating 3D Visualizations ===")
    
    # Figure 1: Side-by-side 3D views
    fig = plt.figure(figsize=(20, 8))
    
    # Vanilla point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Downsample for visualization (if too many points)
    if len(vanilla_pts) > 50000:
        indices = np.random.choice(len(vanilla_pts), 50000, replace=False)
        v_pts = vanilla_pts[indices]
        v_col = vanilla_colors[indices]
    else:
        v_pts = vanilla_pts
        v_col = vanilla_colors
    
    ax1.scatter(v_pts[:, 0], v_pts[:, 1], v_pts[:, 2], 
               c=v_col, s=0.5, alpha=0.6)
    ax1.set_title(f'Vanilla MASt3R\n{len(vanilla_pts):,} points', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)
    
    # Fin3R point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    
    if len(fin3r_pts) > 50000:
        indices = np.random.choice(len(fin3r_pts), 50000, replace=False)
        f_pts = fin3r_pts[indices]
        f_col = fin3r_colors[indices]
    else:
        f_pts = fin3r_pts
        f_col = fin3r_colors
    
    ax2.scatter(f_pts[:, 0], f_pts[:, 1], f_pts[:, 2], 
               c=f_col, s=0.5, alpha=0.6)
    ax2.set_title(f'Fin3R-MASt3R\n{len(fin3r_pts):,} points', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)
    
    # Input image
    ax3 = fig.add_subplot(133)
    ax3.imshow(vanilla_img)
    ax3.set_title('Input Image', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3d_reconstruction_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_dir}/3d_reconstruction_comparison.png")
    plt.close()
    
    # Figure 2: Top-down views (bird's eye)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Vanilla top-down
    axes[0].scatter(v_pts[:, 0], v_pts[:, 1], c=v_pts[:, 2], 
                   cmap='turbo', s=1, alpha=0.6)
    axes[0].set_title(f'Vanilla MASt3R (Top View)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    
    # Fin3R top-down
    axes[1].scatter(f_pts[:, 0], f_pts[:, 1], c=f_pts[:, 2], 
                   cmap='turbo', s=1, alpha=0.6)
    axes[1].set_title(f'Fin3R-MASt3R (Top View)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')
    
    # Input image
    axes[2].imshow(vanilla_img)
    axes[2].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3d_reconstruction_topview.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_dir}/3d_reconstruction_topview.png")
    plt.close()


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def save_ply(filename, points, colors):
    """Save point cloud as PLY file for viewing in external tools"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, col in zip(points, colors):
            r, g, b = (col * 255).astype(np.uint8)
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {r} {g} {b}\n")
    
    print(f"✓ Saved PLY to {filename}")


def compute_reconstruction_metrics(vanilla_pts, fin3r_pts):
    """Compute metrics comparing reconstructions"""
    print("\n=== 3D Reconstruction Metrics ===")
    
    # Point density
    v_density = len(vanilla_pts)
    f_density = len(fin3r_pts)
    print(f"Point count:")
    print(f"  Vanilla: {v_density:,}")
    print(f"  Fin3R:   {f_density:,} ({100*f_density/v_density:.1f}% of vanilla)")
    
    # Spatial extent
    v_extent = vanilla_pts.max(axis=0) - vanilla_pts.min(axis=0)
    f_extent = fin3r_pts.max(axis=0) - fin3r_pts.min(axis=0)
    print(f"\nSpatial extent (X, Y, Z):")
    print(f"  Vanilla: [{v_extent[0]:.2f}, {v_extent[1]:.2f}, {v_extent[2]:.2f}]")
    print(f"  Fin3R:   [{f_extent[0]:.2f}, {f_extent[1]:.2f}, {f_extent[2]:.2f}]")
    
    # Point distribution variance (measure of smoothness)
    v_var = np.var(vanilla_pts, axis=0)
    f_var = np.var(fin3r_pts, axis=0)
    print(f"\nSpatial variance (measure of spread):")
    print(f"  Vanilla: [{v_var[0]:.4f}, {v_var[1]:.4f}, {v_var[2]:.4f}]")
    print(f"  Fin3R:   [{f_var[0]:.4f}, {f_var[1]:.4f}, {f_var[2]:.4f}]")
    
    # Local smoothness (average distance to nearest neighbors)
    from scipy.spatial import cKDTree
    
    print("\nComputing local smoothness...")
    v_tree = cKDTree(vanilla_pts[::10])  # Downsample for speed
    f_tree = cKDTree(fin3r_pts[::10])
    
    v_dists, _ = v_tree.query(vanilla_pts[::10], k=6)  # k=6 (self + 5 neighbors)
    f_dists, _ = f_tree.query(fin3r_pts[::10], k=6)
    
    v_smoothness = np.mean(v_dists[:, 1:])  # Exclude self
    f_smoothness = np.mean(f_dists[:, 1:])
    
    print(f"Average nearest-neighbor distance:")
    print(f"  Vanilla: {v_smoothness:.4f}")
    print(f"  Fin3R:   {f_smoothness:.4f}")
    
    if f_smoothness < v_smoothness:
        print(f"  → Fin3R is {100*(v_smoothness-f_smoothness)/v_smoothness:.1f}% smoother")
    else:
        print(f"  → Vanilla is {100*(f_smoothness-v_smoothness)/v_smoothness:.1f}% smoother")


def main():
    print("=" * 70)
    print("Phase 1: 3D Reconstruction Comparison")
    print("=" * 70)
    
    DATASET_PATH = "/media/jitesh/Extreme SSD/fin3r-mast3r_analysis/datasets/UAVScenes/interval5_CAM_LIDAR"
    MAST3R_CKPT = "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    FIN3R_LORA = "Fin3R/checkpoints/mast3r_lora.pth"
    OUTPUT_DIR = "results/phase1_3d_reconstruction"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load image
    scene_path = Path(DATASET_PATH) / "interval5_AMtown01" / "interval5_CAM"
    img_path = list(scene_path.glob("*.jpg"))[1000]
    
    print(f"\nLoading image: {img_path.name}")
    images = load_images([str(img_path)], size=512)
    
    # Vanilla MASt3R
    print("\n" + "=" * 70)
    print("Running Vanilla MASt3R")
    print("=" * 70)
    vanilla_model = load_mast3r_model(MAST3R_CKPT, device)
    
    with torch.no_grad():
        img_pair = [images[0], images[0]]
        vanilla_out = inference([tuple(img_pair)], vanilla_model, device, batch_size=1, verbose=False)
    
    print("Extracting vanilla point cloud...")
    vanilla_pts, vanilla_colors = extract_point_cloud(vanilla_out, images, conf_threshold=1.0)
    
    del vanilla_model
    torch.cuda.empty_cache()
    
    # Fin3R-MASt3R
    print("\n" + "=" * 70)
    print("Running Fin3R-MASt3R")
    print("=" * 70)
    fin3r_model = load_mast3r_model(MAST3R_CKPT, device)
    fin3r_model = apply_fin3r_lora(fin3r_model, FIN3R_LORA, device)
    
    with torch.no_grad():
        fin3r_out = inference([tuple(img_pair)], fin3r_model, device, batch_size=1, verbose=False)
    
    print("Extracting Fin3R point cloud...")
    fin3r_pts, fin3r_colors = extract_point_cloud(fin3r_out, images, conf_threshold=1.0)
    
    del fin3r_model
    torch.cuda.empty_cache()
    
    # Get image for visualization
    vanilla_img = images[0]['img'].squeeze().permute(1, 2, 0).cpu().numpy()
    vanilla_img = (vanilla_img - vanilla_img.min()) / (vanilla_img.max() - vanilla_img.min())
    
    # Compute metrics
    compute_reconstruction_metrics(vanilla_pts, fin3r_pts)
    
    # Visualize
    visualize_3d_comparison(vanilla_pts, vanilla_colors, fin3r_pts, fin3r_colors,
                           vanilla_img, OUTPUT_DIR)
    
    # Save PLY files for external viewing
    print("\n" + "=" * 70)
    print("Saving PLY files (open with MeshLab/CloudCompare)")
    print("=" * 70)
    save_ply(f"{OUTPUT_DIR}/vanilla_pointcloud.ply", vanilla_pts, vanilla_colors)
    save_ply(f"{OUTPUT_DIR}/fin3r_pointcloud.ply", fin3r_pts, fin3r_colors)
    
    print("\n" + "=" * 70)
    print("3D Reconstruction Comparison Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - 3d_reconstruction_comparison.png (3D views)")
    print("  - 3d_reconstruction_topview.png (bird's eye view)")
    print("  - vanilla_pointcloud.ply (open in MeshLab)")
    print("  - fin3r_pointcloud.ply (open in MeshLab)")


if __name__ == "__main__":
    main()