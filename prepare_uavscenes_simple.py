#!/usr/bin/env python3
"""
Prepare UAVScenes for MASt3R-SLAM (Simple RGB folder format)

Usage:
    python prepare_uavscenes_simple.py \
        --uavscenes_dir datasets/UAVScenes/interval5_CAM_LIDAR \
        --scene interval5_AMtown01 \
        --output_dir MASt3R-SLAM/datasets/uavscenes \
        --max_frames 500 \
        --start_frame 1000
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uavscenes_dir', type=str, required=True)
    parser.add_argument('--scene', type=str, default='interval5_AMtown01')
    parser.add_argument('--output_dir', type=str, default='MASt3R-SLAM/datasets/uavscenes')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames (None = use all)')
    parser.add_argument('--start_frame', type=int, default=0)
    
    args = parser.parse_args()
    
    scene_path = Path(args.uavscenes_dir) / args.scene
    output_dir = Path(args.output_dir) / args.scene
    
    print("=" * 80)
    print("Preparing UAVScenes for MASt3R-SLAM (Simple RGB Format)")
    print("=" * 80)
    print(f"Scene: {args.scene}")
    print(f"Output: {output_dir}")
    print(f"Frames: {args.start_frame} to {end_frame-1 if args.max_frames else 'end'}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = scene_path / "sampleinfos_interpolated.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Select frames
    if args.max_frames is None:
        selected = metadata[args.start_frame:]
        end_frame = len(metadata)
    else:
        end_frame = min(args.start_frame + args.max_frames, len(metadata))
        selected = metadata[args.start_frame:end_frame]
    
    print(f"\n✓ Selected {len(selected)} frames (frame {args.start_frame} to {end_frame-1})")
    print("\nCopying images...")
    
    # Copy images with sequential naming
    copied = 0
    skipped = 0
    for idx, frame in enumerate(tqdm(selected)):
        img_name = frame['OriginalImageName']
        src = scene_path / "interval5_CAM" / img_name
        
        # Check if image exists
        if not src.exists():
            skipped += 1
            continue
        
        # Sequential naming: 000000.jpg, 000001.jpg, ...
        dst = output_dir / f"{copied:06d}.jpg"
        shutil.copy2(src, dst)
        copied += 1
    
    print(f"\n✓ Copied {copied} images to {output_dir}")
    if skipped > 0:
        print(f"  (Skipped {skipped} missing images)")
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print(f"\nTo run MASt3R-SLAM:")
    print(f"  cd MASt3R-SLAM")
    print(f"  python main.py --dataset {output_dir} --config config/base.yaml")
    print(f"\nOr with visualization disabled (faster):")
    print(f"  python main.py --dataset {output_dir} --config config/base.yaml --no-viz")


if __name__ == "__main__":
    main()