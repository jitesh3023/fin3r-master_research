#!/usr/bin/env python3
"""
Patch MASt3R-SLAM main.py to properly support single_thread mode

Usage:
    python patch_main_singlethread.py /path/to/MASt3R-SLAM/main.py
"""

import sys
import shutil
from pathlib import Path


def patch_main_py(main_py_path):
    """Patch main.py to check single_thread before creating mp.Manager()"""
    
    main_py_path = Path(main_py_path)
    
    if not main_py_path.exists():
        print(f"Error: {main_py_path} not found!")
        return False
    
    # Backup original
    backup_path = main_py_path.with_suffix('.py.backup')
    shutil.copy2(main_py_path, backup_path)
    print(f"✓ Backed up to: {backup_path}")
    
    # Read file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'if config["single_thread"]' in content and 'manager = mp.Manager()' in content:
        # Find the pattern to see if it's already wrapped
        if content.find('if config["single_thread"]') < content.find('manager = mp.Manager()'):
            print("✓ Already patched!")
            return True
    
    # Find and replace the manager creation section
    original = '''    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)'''
    
    patched = '''    # Check single_thread mode before creating multiprocessing manager
    if config["single_thread"]:
        manager = None
        main2viz = None
        viz2main = None
    else:
        manager = mp.Manager()
        main2viz = new_queue(manager, args.no_viz)
        viz2main = new_queue(manager, args.no_viz)'''
    
    if original in content:
        content = content.replace(original, patched)
        print("✓ Patched manager creation")
    else:
        print("⚠ Could not find exact manager creation pattern")
        print("  You may need to manually edit main.py")
        return False
    
    # Find and patch keyframes/states creation
    original2 = '''    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)'''
    
    patched2 = '''    if config["single_thread"]:
        # Use non-shared objects for single-threaded mode
        keyframes = SharedKeyframes(None, h, w)
        states = SharedStates(None, h, w)
    else:
        keyframes = SharedKeyframes(manager, h, w)
        states = SharedStates(manager, h, w)'''
    
    if original2 in content:
        content = content.replace(original2, patched2)
        print("✓ Patched keyframes/states creation")
    else:
        print("⚠ Could not find keyframes/states pattern")
    
    # Write patched content
    with open(main_py_path, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Successfully patched {main_py_path}")
    print(f"  Original backed up to: {backup_path}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_main_singlethread.py /path/to/main.py")
        print("\nExample:")
        print("  python patch_main_singlethread.py MASt3R-SLAM/main.py")
        sys.exit(1)
    
    main_py = sys.argv[1]
    success = patch_main_py(main_py)
    
    if success:
        print("\n" + "=" * 80)
        print("Patching complete!")
        print("=" * 80)
        print("\nNow you can run:")
        print("  python main.py --dataset <path> --config config/single_thread.yaml --no-viz")
    else:
        print("\n✗ Patching failed. Check error messages above.")
        sys.exit(1)