#!/usr/bin/env python3
"""
Asteroids AI Training Launcher

This script provides convenient ways to run the Asteroids AI training system
with different configurations for unattended learning.

Usage examples:
  # Basic automatic training (default speed, with graphics)
  python run_training.py
  
  # Fast training with graphics
  python run_training.py --speed 3.0
  
  # Maximum speed headless training (no graphics)
  python run_training.py --headless --speed 5.0
  
  # Ultra-fast training for overnight runs
  python run_training.py --headless --speed 10.0
"""

import subprocess
import sys
import argparse
import time
import os

def print_banner():
    """Print system information banner for training launcher."""
    print("=" * 60)
    print("ASTEROIDS AI TRAINING SYSTEM")
    print("=" * 60)
    print("This system will train neural AI enemies to hunt the player.")
    print("The AI learns through reinforcement learning across 3 phases:")
    print("  Phase 1: Learn to approach stationary target")
    print("  Phase 2: Learn to follow slow-moving target") 
    print("  Phase 3: Learn advanced hunting patterns")
    print()
    print("The game will automatically restart when the player dies,")
    print("continuously training the AI to become better at hunting.")
    print("=" * 60)

def print_training_info(args):
    """Print current training configuration and expected performance metrics."""
    print(f"Training Configuration:")
    print(f"   Auto-training: {'ENABLED' if args.auto_train else 'DISABLED'}")
    print(f"   Speed: {args.speed}x normal")
    print(f"   Headless: {'ENABLED (no graphics)' if args.headless else 'DISABLED (with graphics)'}")
    print(f"   Expected performance: ~{estimate_episodes_per_hour(args)} episodes/hour")
    print()
    
    if args.headless:
        print("HEADLESS MODE - Maximum training speed")
        print("   - No graphics rendering")
        print("   - Reduced CPU usage")
        print("   - Optimal for overnight training")
    else:
        print("GRAPHICS MODE - Visual training monitoring")
        print("   - Real-time AI visualization")
        print("   - Training metrics overlay")
        print("   - Watch the AI learn")
    
    print()
    print("Files that will be created/updated:")
    print("   - ai_enemy_brain.pth (main AI model)")
    print("   - ai_enemy_brain_backup.pth (backup)")
    print("   - training_log.json (detailed episode logs)")
    print("   - ai_brain_phase_X_milestone.pth (phase checkpoints)")
    print()

def estimate_episodes_per_hour(args):
    """Calculate estimated episodes per hour based on current configuration."""
    base_episodes_per_hour = 120  # baseline estimate for normal speed with graphics
    
    # apply speed multiplier
    speed_factor = args.speed
    
    # headless mode provides significant performance boost
    headless_factor = 2.0 if args.headless else 1.0
    
    # high speeds have diminishing returns due to game logic overhead
    if args.speed > 5.0:
        speed_factor = 5.0 + (args.speed - 5.0) * 0.5
    
    return int(base_episodes_per_hour * speed_factor * headless_factor)

def check_dependencies():
    """Verify that required Python packages are available."""
    try:
        import pygame
        import torch
        import numpy as np
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Launch Asteroids AI Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--auto-train', action='store_true', default=True,
                       help='Enable automatic training mode (default: enabled)')
    parser.add_argument('--manual', action='store_true',
                       help='Disable auto-training for manual play')
    parser.add_argument('--speed', type=float, default=2.0,
                       help='Training speed multiplier (default: 2.0, max recommended: 10.0)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without graphics for maximum speed')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test run (5 minutes max)')
    
    args = parser.parse_args()
    
    # disable auto-training when manual mode specified
    if args.manual:
        args.auto_train = False
    
    # enforce speed limits for stability
    if args.speed < 0.1:
        print("Warning: Minimum speed is 0.1x")
        args.speed = 0.1
    elif args.speed > 20.0:
        print("Warning: Maximum speed is 20.0x")
        args.speed = 20.0
    
    # headless mode requires auto-training
    if args.headless:
        args.auto_train = True
    
    print_banner()
    
    # verify dependencies before starting
    if not check_dependencies():
        return 1
    
    print_training_info(args)
    
    if not args.auto_train:
        print("MANUAL MODE - Game will not auto-restart")
        print("   Press Y after game over to restart manually")
        print("   Press T in-game to toggle training mode")
    else:
        estimated_time = "indefinite (until stopped)" if not args.quick_test else "5 minutes"
        print(f"AUTOMATIC TRAINING MODE - Running for {estimated_time}")
        print("   Game will auto-restart every time player dies")
        print("   AI model auto-saves every 10 successful episodes")
        print("   Press Ctrl+C to stop training safely")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # construct command line arguments for main.py
    cmd = [sys.executable, "main.py"]
    
    if args.auto_train:
        cmd.append("--auto-train")
    
    cmd.extend(["--speed", str(args.speed)])
    
    if args.headless:
        cmd.append("--headless")
    
    try:
        start_time = time.time()
        
        if args.quick_test:
            # set timeout for quick test runs
            import signal
            import threading
            
            def timeout_handler():
                time.sleep(300)  # 5 minute timeout
                print("\nQuick test complete (5 minutes)")
                os._exit(0)
            
            timer = threading.Thread(target=timeout_handler, daemon=True)
            timer.start()
        
        # execute training process
        result = subprocess.run(cmd, check=False)
        
        end_time = time.time()
        duration = end_time - start_time
        hours = duration / 3600
        
        print(f"\nTraining session ended")
        print(f"   Duration: {hours:.1f} hours")
        print(f"   AI model should be saved in ai_enemy_brain.pth")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        print("   AI model should be automatically saved")
        return 0
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 