import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_strong_baselines import main  # noqa: E402


if __name__ == "__main__":
    sys.argv[1:1] = ["--baseline", "aya-23"]
    main()
