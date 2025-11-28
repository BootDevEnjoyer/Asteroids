#!/usr/bin/env python3
"""
Developer tool for automated AI training.

For normal interactive use, run: python main.py

This script is intended for unattended training sessions, particularly
headless overnight runs. It provides SDL environment configuration for
headless mode and optional time-limited test runs.

Usage:
    # Default headless training at 5x speed
    python run_training.py

    # Fast headless training
    python run_training.py --speed 10.0

    # Training with graphics (for monitoring)
    python run_training.py --graphics

    # Quick 5-minute test run
    python run_training.py --quick-test
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def check_dependencies() -> bool:
    """Verify required packages are available."""
    required = ["pygame", "torch", "numpy"]
    missing = [pkg for pkg in required if not _can_import(pkg)]
    
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Install: pip install -r requirements.txt")
        return False
    return True


def _can_import(package: str) -> bool:
    """Check if a package can be imported."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def build_env(headless: bool) -> dict:
    """Configure environment for training, especially for headless mode."""
    env = os.environ.copy()
    if headless:
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env.setdefault("SDL_AUDIODRIVER", "dummy")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Asteroids AI Training (Developer Tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For interactive play, run: python main.py",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Training speed multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--graphics",
        action="store_true",
        help="Enable graphics (default: headless)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run for 5 minutes only (for testing)",
    )

    args = parser.parse_args()
    headless = not args.graphics

    # Clamp speed
    args.speed = max(0.1, min(20.0, args.speed))

    if not check_dependencies():
        return 1

    # Locate main.py
    main_path = Path(__file__).resolve().parent / "main.py"
    if not main_path.exists():
        print(f"Error: {main_path} not found")
        return 1

    # Build command
    cmd = [
        sys.executable,
        str(main_path),
        "--auto-train",
        "--speed", str(args.speed),
    ]
    if headless:
        cmd.append("--headless")

    # Print configuration
    mode = "headless" if headless else "graphics"
    duration = "5 min" if args.quick_test else "indefinite"
    print(f"AI Training: {mode} mode, {args.speed}x speed, {duration}")
    print("Press Ctrl+C to stop\n")

    env = build_env(headless)
    start_time = time.time()

    try:
        if args.quick_test:
            proc = subprocess.Popen(cmd, env=env)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("\nQuick test complete (5 min)")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            return proc.returncode or 0
        else:
            result = subprocess.run(cmd, env=env)
            return result.returncode

    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        return 0
    finally:
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        print(f"Duration: {hours:.1f}h")


if __name__ == "__main__":
    raise SystemExit(main())
