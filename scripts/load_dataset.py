# scripts/load_dataset.py

import os
import sys
import zipfile
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "datasets").is_dir() or (cur / "configs").is_dir():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Could not locate project root containing 'datasets' or 'configs'")

def load_bird_dataset() -> tuple[Path, Path] | None:
    """
    Download + unzip the BirdSQL training set (Yuxuan13/bird_train)
    into datasets/bird_train/train/, yielding:
      - train_with_schema.json
      - train_databases/  (unzipped)
    Returns (json_path, db_root) on success, or None on failure.
    """

    hf_repo      = "Yuxuan13/bird_train"
    repo_type    = "dataset"
    # files live at repo root
    json_in_repo = "train_with_schema.json"
    zip_in_repo  = "train_databases.zip"

    # where we'll drop them
    repo_root  = _find_repo_root(Path(__file__).parent)
    # # ---------------------------------------------------
    # # debugging repo_root = home
    # repo_root = Path.home() # for testing purposes, set to home
    # # ---------------------------------------------------
    local_root = repo_root / "datasets" / "bird_train" / "train"
    json_path  = local_root / "train_with_schema.json"
    db_root    = local_root / "train_databases"
    local_root.mkdir(parents=True, exist_ok=True)

    print("=" * 40, file=sys.stderr)
    print(f"[local root] {local_root}", file=sys.stderr)
    print("=" * 40, file=sys.stderr)

    # 1) Download JSON if missing
    if not json_path.exists():
        print("[load_bird_dataset] Downloading JSON…", file=sys.stderr)
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename=json_in_repo,
                repo_type=repo_type,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"[load_bird_dataset] ERROR fetching JSON: {e}", file=sys.stderr)
            return None
    else:
        print("[load_bird_dataset] JSON already present; skipping.", file=sys.stderr)

    # 2) Download & extract DB ZIP if missing
    zip_path = local_root / "train_databases.zip"
    
    # Download zip if it doesn't exist
    if not zip_path.exists():
        print("[load_bird_dataset] Downloading DB zip…", file=sys.stderr)
        try:
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                filename=zip_in_repo,
                repo_type=repo_type,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
            zip_path = Path(zip_path) 
        except Exception as e:
            print(f"[load_bird_dataset] ERROR fetching DB zip: {e}", file=sys.stderr)
            return None
    else:
        print("[load_bird_dataset] DB zip already downloaded.", file=sys.stderr)
    
    # Extract zip if database directory doesn't exist
    if not db_root.is_dir():
        print("[load_bird_dataset] Extracting DB zip…", file=sys.stderr)
        try:
            tmp_dir = local_root / "tmp_unzip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            
            # The zip file contains a train_databases folder, so we need to move its contents
            extracted_db_dir = tmp_dir / "train_databases"
            if extracted_db_dir.exists():
                # Move the extracted train_databases folder to our target location
                extracted_db_dir.rename(db_root)
            else:
                # If the zip doesn't contain a train_databases folder, move the tmp_dir contents
                tmp_dir.rename(db_root)
            
            # Clean up any remaining tmp directory
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"[load_bird_dataset] ERROR extracting DB zip: {e}", file=sys.stderr)
            if 'tmp_dir' in locals() and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
    else:
        print("[load_bird_dataset] DB already extracted; skipping extraction.", file=sys.stderr)
    
    # Delete the zip file after successful extraction to save space
    if zip_path.exists() and db_root.is_dir():
        try:
            zip_path.unlink()
            print(f"[load_bird_dataset] Deleted zip file: {zip_path}", file=sys.stderr)
        except Exception as e:
            print(f"[load_bird_dataset] Warning: Could not delete zip file: {e}", file=sys.stderr)

    # final sanity check
    if not json_path.exists() or not db_root.is_dir():
        print("[load_bird_dataset] ERROR: files missing after load", file=sys.stderr)
        return None

    print(f"[load_bird_dataset] Done ✔ JSON → {json_path}, DB → {db_root}", file=sys.stderr)
    return json_path, db_root


def main() -> None:
    """
    Entry point: controlled by env-var LOAD_BIRD_DATASET.
    """
    if os.getenv("LOAD_BIRD_DATASET", "").lower() in {"1", "true", "yes"}:
        result = load_bird_dataset()
        if result:
            j, d = result
            print(f"[main] BirdSQL → JSON: {j}, DB: {d}")
    else:
        print("[main] Skipping BirdSQL.")

if __name__ == "__main__":
    main()
