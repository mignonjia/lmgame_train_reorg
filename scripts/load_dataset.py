# scripts/load_dataset.py

import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "external" / "webshop-minimal").is_dir():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Could not locate project root containing 'external/webshop-minimal'")

def load_webshop_dataset() -> Path | None:
    """
    Download the WebShop JSON files directly from Hugging Face Dataset Hub
    (Yuxuan13/webshop_dataset) into:
      <repo>/external/webshop-minimal/webshop_minimal/data/full/

    Returns the path to items_shuffle.json on success, or None on failure.
    """
    repo_id   = "Yuxuan13/webshop_dataset"
    repo_type = "dataset"
    filenames = ["items_shuffle.json", "items_ins_v2.json"]

    # target directory in the local repo
    repo_root = _find_repo_root(Path(__file__).parent)
    data_dir  = repo_root / "external" / "webshop-minimal" \
                          / "webshop_minimal" / "data" / "full"
    data_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for fname in filenames:
        try:
            print(f"[load_webshop_dataset] Downloading {fname} from {repo_id}…", file=sys.stderr)
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type=repo_type,
                cache_dir=str(data_dir),      # you can omit to use default cache
                library_name="hf_hub_download"
            )
        except Exception as e:
            print(f"[load_webshop_dataset] ERROR downloading {fname}: {e}", file=sys.stderr)
            return None

        # hf_hub_download returns the cache path; now copy it into data_dir
        dest = data_dir / fname
        if not dest.exists():
            try:
                Path(path).replace(dest)
            except Exception:
                # fallback to a byte‐copy if replace fails
                dest.write_bytes(Path(path).read_bytes())
        local_paths.append(dest)

    print("[load_webshop_dataset] All files downloaded ✔", file=sys.stderr)
    return local_paths[0]  # items_shuffle.json
def load_bird_dataset() -> tuple[Path, Path] | None:
    """
    Download + unzip the BirdSQL training set (Yuxuan13/bird_train)
    into datasets/bird_train/train/, yielding:
      - train_with_schema.json
      - train_databases/  (unzipped)
    Returns (json_path, db_root) on success, or None on failure.
    """
    from huggingface_hub import hf_hub_download
    import zipfile, shutil, sys

    hf_repo      = "Yuxuan13/bird_train"
    # files live at repo root
    json_in_repo = "train_with_schema.json"
    zip_in_repo  = "train_databases.zip"

    # where we'll drop them
    repo_root  = _find_repo_root(Path(__file__).parent)
    local_root = repo_root / "datasets" / "bird_train" / "train"
    json_path  = local_root / "train_with_schema.json"
    db_root    = local_root / "train_databases"
    local_root.mkdir(parents=True, exist_ok=True)

    # 1) Download JSON if missing
    if not json_path.exists():
        print("[load_bird_dataset] Downloading JSON…", file=sys.stderr)
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename=json_in_repo,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"[load_bird_dataset] ERROR fetching JSON: {e}", file=sys.stderr)
            return None
    else:
        print("[load_bird_dataset] JSON already present; skipping.", file=sys.stderr)

    # 2) Download & extract DB ZIP if missing
    if not db_root.is_dir():
        print("[load_bird_dataset] Downloading + extracting DB zip…", file=sys.stderr)
        try:
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                filename=zip_in_repo,
                local_dir=str(local_root),
                local_dir_use_symlinks=False
            )
            tmp_dir = local_root / "tmp_unzip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            tmp_dir.rename(db_root)
        except Exception as e:
            print(f"[load_bird_dataset] ERROR fetching/extracting DB: {e}", file=sys.stderr)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
    else:
        print("[load_bird_dataset] DB already extracted; skipping.", file=sys.stderr)

    # final sanity check
    if not json_path.exists() or not db_root.is_dir():
        print("[load_bird_dataset] ERROR: files missing after load", file=sys.stderr)
        return None

    print(f"[load_bird_dataset] Done ✔ JSON → {json_path}, DB → {db_root}", file=sys.stderr)
    return json_path, db_root


def main() -> None:
    """
    Entry point: controlled by env-vars LOAD_WEBSHOP_DATASET and LOAD_BIRD_DATASET.
    """
    if os.getenv("LOAD_WEBSHOP_DATASET", "").lower() in {"1", "true", "yes"}:
        path = load_webshop_dataset()
        print(f"[main] WebShop → {path}")
    else:
        print("[main] Skipping WebShop.")

    if os.getenv("LOAD_BIRD_DATASET", "").lower() in {"1", "true", "yes"}:
        result = load_bird_dataset()
        if result:
            j, d = result
            print(f"[main] BirdSQL → JSON: {j}, DB: {d}")
    else:
        print("[main] Skipping BirdSQL.")

if __name__ == "__main__":
    main()
