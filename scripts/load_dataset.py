#!/usr/bin/env python3
"""
Bird dataset loading utilities for LMGameRL.
Separate from main package installation.
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

def _find_repo_root(start: Path) -> Path:
    """Find repository root by looking for characteristic directories."""
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "LMGameRL").is_dir() or (cur / "pyproject.toml").is_file():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Could not locate project root")

def load_bird_dataset() -> tuple[Path, Path] | None:
    """
    Download + unzip the BirdSQL training set (Yuxuan13/bird_train)
    into datasets/bird_train/train/, yielding:
      - train_with_schema.json  
      - train_databases/  (unzipped)
    Returns (json_path, db_root) on success, or None on failure.
    """
    hf_repo = "Yuxuan13/bird_train"
    repo_type = "dataset"
    json_in_repo = "train_with_schema.json"
    zip_in_repo = "train_databases.zip"

    # Find repo root and set up paths
    try:
        repo_root = _find_repo_root(Path(__file__).parent)
    except FileNotFoundError:
        print("❌ Could not find LMGameRL project root", file=sys.stderr)
        return None
    # # -----------------------------------------
    # # debug load_dataset
    # repo_root = Path.home()
    # # -----------------------------------------
    local_root = repo_root / "datasets" / "bird_train" / "train"
    json_path = local_root / "train_with_schema.json"
    db_root = local_root / "train_databases"
    local_root.mkdir(parents=True, exist_ok=True)

    print(f"📁 Dataset directory: {local_root}", file=sys.stderr)

    # 1) Download JSON if missing
    if not json_path.exists():
        print("📥 Downloading Bird dataset JSON...", file=sys.stderr)
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename=json_in_repo,
                repo_type=repo_type,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"❌ ERROR fetching JSON: {e}", file=sys.stderr)
            return None
    else:
        print("✅ JSON already present", file=sys.stderr)

    # 2) Download & extract DB ZIP if missing
    zip_path = local_root / "train_databases.zip"
    
    if not zip_path.exists():
        print("📥 Downloading Bird database zip...", file=sys.stderr)
        try:
            downloaded_zip = hf_hub_download(
                repo_id=hf_repo,
                filename=zip_in_repo,
                repo_type=repo_type,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
            zip_path = Path(downloaded_zip)
        except Exception as e:
            print(f"❌ ERROR fetching DB zip: {e}", file=sys.stderr)
            return None
    else:
        print("✅ DB zip already downloaded", file=sys.stderr)
    
    # Extract zip if database directory doesn't exist
    if not db_root.is_dir():
        print("📦 Extracting Bird database zip...", file=sys.stderr)
        try:
            tmp_dir = local_root / "tmp_unzip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            
            extracted_db_dir = tmp_dir / "train_databases"
            if extracted_db_dir.exists():
                extracted_db_dir.rename(db_root)
            else:
                tmp_dir.rename(db_root)
            
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            print(f"❌ ERROR extracting DB zip: {e}", file=sys.stderr)
            if 'tmp_dir' in locals() and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
    else:
        print("✅ DB already extracted", file=sys.stderr)
    
    # Clean up zip file after successful extraction
    if zip_path.exists() and db_root.is_dir():
        try:
            zip_path.unlink()
            print(f"🗑️  Deleted zip file: {zip_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Warning: Could not delete zip file: {e}", file=sys.stderr)

    # Final verification
    if not json_path.exists() or not db_root.is_dir():
        print("❌ ERROR: files missing after load", file=sys.stderr)
        return None

    print(f"✅ Bird dataset ready → JSON: {json_path}, DB: {db_root}", file=sys.stderr)
    return json_path, db_root

def main():
    """Main entry point with CLI options."""
    parser = argparse.ArgumentParser(description="Load Bird dataset for LMGameRL")
    parser.add_argument("--bird", action="store_true", help="Load Bird dataset")
    
    args = parser.parse_args()
    
    # Check environment variables for backward compatibility
    load_bird = args.bird or os.getenv("LOAD_BIRD_DATASET", "").lower() in {"1", "true", "yes"}
    
    if not load_bird:
        print("📋 Bird dataset loading not requested.")
        return
    
    print("🚀 LMGameRL Bird Dataset Loader")
    print("=" * 30)
    
    result = load_bird_dataset()
    if not result:
        print("❌ Bird dataset loading failed")
        sys.exit(1)
    
    json_path, db_root = result
    print(f"\n✅ Bird dataset loaded successfully!")
    print(f"   JSON: {json_path}")
    print(f"   DB:   {db_root}")

if __name__ == "__main__":
    main()