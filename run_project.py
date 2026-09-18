"""
run_project.py
==============
Root launcher script for Restaurant Workforce Analytics & Staff Scheduling Optimization.

Usage:
    python run_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[INFO] Execution interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] An unexpected error occurred: {exc}")
        raise exc
