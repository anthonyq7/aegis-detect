import shutil
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    return Path.home() / ".aegis" / "models"

def remove_cache() -> None:

    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"\u2717 Removed cache directory: {cache_dir}")
        sys.exit(0)
    else:
        print(f"Cache directory does not exist: {cache_dir}")
        sys.exit(0)

if __name__ == "__main__":
    remove_cache()

