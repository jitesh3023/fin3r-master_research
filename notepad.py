# import sys
# import os
# import json
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import matplotlib.pyplot as plt

# # Add paths
# sys.path.insert(0, 'mast3r')
# sys.path.insert(0, 'mast3r/dust3r')
# sys.path.insert(0, 'Fin3R')

# from mast3r.model import AsymmetricMASt3R
# from dust3r.inference import inference
# from dust3r.utils.image import load_images

# from vggt.models.renorm_lora import get_renormalized_peft_model
# from peft import LoraConfig


# def load_mast3r_model(checkpoint_path, device='cuda'):
#     print(f"Loading MASt3R from {checkpoint_path}...")
#     model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
#     model.eval()
#     return model


# def apply_fin3r_lora(model, lora_path, device='cuda'):
#     print("\nApplying Fin3R LoRA adapters")
    
#     # LoRA configuration
#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=8,
#         target_modules=["qkv"],
#         lora_dropout=0.1,
#     )
    
#     # Apply to encoder blocks
#     num_blocks = len(model.enc_blocks)
#     for i, block in enumerate(model.enc_blocks):
#         block.attn = get_renormalized_peft_model(block.attn, lora_config)
    
#     print(f"   Applied LoRA to {num_blocks} encoder blocks")
    
#     # Load LoRA weights
#     state_dict = torch.load(lora_path, map_location=device, weights_only=False)
#     model.load_state_dict(state_dict, strict=False)
#     print(f"   Loaded Fin3R weights")
    
#     return model


# def load_uav_image_pair(dataset_path, scene="interval5_AMtown01", idx1=0, idx2=10):

#     scene_path = Path(dataset_path) / scene
#     metadata_path = scene_path / "sampleinfos_interpolated.json"
    
#     with open(metadata_path, 'r') as f:
#         metadata = json.load(f)
    
#     # Get image paths
#     frame1 = metadata[idx1]
#     frame2 = metadata[idx2]
    
#     img_path1 = scene_path / "interval5_CAM" / frame1['OriginalImageName']
#     img_path2 = scene_path / "interval5_CAM" / frame2['OriginalImageName']
    
#     print(f"\nLoading image pair:")
#     print(f"  Image 1: {frame1['OriginalImageName']}")
#     print(f"  Image 2: {frame2['OriginalImageName']}")
    
#     # Load using dust3r's image loader
#     images = load_images([str(img_path1), str(img_path2)], size=512)
    
#     return images, frame1, frame2


# def run_inference(model, images, device='cuda'):
#     """Run MASt3R inference on image pair"""
#     with torch.no_grad():
#         output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
#     return output


# def visualize_results(vanilla_out, fin3r_out, images, save_path=None):
#     """Visualize depth maps comparison"""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     # Original images
#     img1 = images[0]['img'].squeeze().permute(1, 2, 0).cpu().numpy()
#     img2 = images[1]['img'].squeeze().permute(1, 2, 0).cpu().numpy()
#     axes[0, 0].imshow(img1)
#     axes[0, 0].set_title('Image 1', fontsize=14)
#     axes[0, 0].axis('off')
    
#     axes[1, 0].imshow(img2)
#     axes[1, 0].set_title('Image 2', fontsize=14)
#     axes[1, 0].axis('off')
    
#     # Vanilla MASt3R depths
#     v_depth1 = vanilla_out['pred1']['pts3d'][..., 2].squeeze().cpu().numpy()
#     v_depth2 = vanilla_out['pred2']['pts3d_in_other_view'][..., 2].squeeze().cpu().numpy()
    
#     im1 = axes[0, 1].imshow(v_depth1, cmap='turbo')
#     axes[0, 1].set_title('Vanilla MASt3R - Depth 1', fontsize=14)
#     axes[0, 1].axis('off')
#     plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
#     im2 = axes[1, 1].imshow(v_depth2, cmap='turbo')
#     axes[1, 1].set_title('Vanilla MASt3R - Depth 2', fontsize=14)
#     axes[1, 1].axis('off')
#     plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
#     # Fin3R-MASt3R depths
#     f_depth1 = fin3r_out['pred1']['pts3d'][..., 2].squeeze().cpu().numpy()
#     f_depth2 = fin3r_out['pred2']['pts3d_in_other_view'][..., 2].squeeze().cpu().numpy()
    
#     im3 = axes[0, 2].imshow(f_depth1, cmap='turbo')
#     axes[0, 2].set_title('Fin3R-MASt3R - Depth 1', fontsize=14)
#     axes[0, 2].axis('off')
#     plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
#     im4 = axes[1, 2].imshow(f_depth2, cmap='turbo')
#     axes[1, 2].set_title('Fin3R-MASt3R - Depth 2', fontsize=14)
#     axes[1, 2].axis('off')
#     plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"\n✓ Saved visualization to {save_path}")
    
#     plt.show()
#     return fig


# def print_depth_stats(output, name="Model"):
#     """Print depth statistics"""
#     depth1 = output['pred1']['pts3d'][..., 2].cpu().numpy()
#     # pred2 has pts3d_in_other_view instead of pts3d
#     depth2 = output['pred2']['pts3d_in_other_view'][..., 2].cpu().numpy()
    
#     print(f"\n{name} - Depth Statistics:")
#     print(f"  Image 1: mean={depth1.mean():.3f}, std={depth1.std():.3f}, "
#           f"range=[{depth1.min():.3f}, {depth1.max():.3f}]")
#     print(f"  Image 2: mean={depth2.mean():.3f}, std={depth2.std():.3f}, "
#           f"range=[{depth2.min():.3f}, {depth2.max():.3f}]")
    
#     # Confidence if available
#     if 'conf' in output['pred1']:
#         conf1 = output['pred1']['conf'].cpu().numpy()
#         print(f"  Confidence 1: mean={conf1.mean():.3f}, std={conf1.std():.3f}")


# def main():
#     print("="*80)
#     print("Phase 1: Fin3R-MASt3R Evaluation on UAVScenes")
#     print("="*80)
    
#     # Configuration
#     DATASET_PATH = "/media/jitesh/Extreme SSD/fin3r-mast3r_analysis/datasets/UAVScenes/interval5_CAM_LIDAR"
#     MAST3R_CKPT = "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
#     FIN3R_LORA = "Fin3R/checkpoints/mast3r_lora.pth"
#     OUTPUT_DIR = "results/phase1"
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"\nUsing device: {device}")
    
#     # Load UAV image pair
#     print("\n" + "="*80)
#     print("1. Loading UAVScenes Data")
#     print("="*80)
#     images, frame1, frame2 = load_uav_image_pair(DATASET_PATH, idx1=0, idx2=10)
#     print(f"  ✓ Loaded image pair (resolution: {images[0]['img'].size})")
    
#     # Load Vanilla MASt3R
#     print("\n" + "="*80)
#     print("2. Running Vanilla MASt3R")
#     print("="*80)
#     vanilla_model = load_mast3r_model(MAST3R_CKPT, device)
#     vanilla_output = run_inference(vanilla_model, images, device)
#     print("  ✓ Inference complete")
#     print_depth_stats(vanilla_output, "Vanilla MASt3R")
    
#     # Free GPU memory before loading second model
#     del vanilla_model
#     torch.cuda.empty_cache()
#     print("\n  ✓ Cleared vanilla model from GPU")
    
#     # Load model with Fin3R LoRA
#     print("\n" + "="*80)
#     print("3. Running Fin3R-MASt3R")
#     print("="*80)
#     fin3r_model = load_mast3r_model(MAST3R_CKPT, device)
#     fin3r_model = apply_fin3r_lora(fin3r_model, FIN3R_LORA, device)
#     fin3r_output = run_inference(fin3r_model, images, device)
#     print("  ✓ Inference complete")
#     print_depth_stats(fin3r_output, "Fin3R-MASt3R")
    
#     # Free GPU memory
#     del fin3r_model
#     torch.cuda.empty_cache()
    
#     # Visualize comparison
#     print("\n" + "="*80)
#     print("4. Visualizing Results")
#     print("="*80)
#     save_path = f"{OUTPUT_DIR}/comparison_pair_0.png"
#     visualize_results(vanilla_output, fin3r_output, images, save_path)
    
#     # Save raw outputs for further analysis
#     print("\nSaving raw outputs...")
#     torch.save({
#         'vanilla': vanilla_output,
#         'fin3r': fin3r_output,
#         'metadata': {
#             'frame1': frame1,
#             'frame2': frame2,
#         }
#     }, f"{OUTPUT_DIR}/outputs_pair_0.pth")
#     print(f"  ✓ Saved to {OUTPUT_DIR}/outputs_pair_0.pth")
    
#     print("\n" + "="*80)
#     print("✓ Phase 1 Complete!")
#     print("="*80)
#     print(f"\nResults saved to: {OUTPUT_DIR}/")
#     print("\nNext steps:")
#     print("1. Examine depth maps - does Fin3R show better detail?")
#     print("2. Try more image pairs with different baselines")
#     print("3. Proceed to Phase 2: SLAM integration")


# if __name__ == "__main__":
#     main()