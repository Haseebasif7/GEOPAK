"""
Download checkpoints from Modal volume after training.

Usage (local via Modal CLI):
    modal run model/province/download_checkpoints_from_modal_fixed.py
    modal run model/province/download_checkpoints_from_modal_fixed.py --output-dir ./checkpoints

This uses a Modal function (with the volume mounted at /data) to list and
copy checkpoint files from the volume back to your local machine.
"""

import sys
from pathlib import Path

import modal


# Modal app and volume
app = modal.App("geopak-download-checkpoints")
volume = modal.Volume.from_name("geopak-data", create_if_missing=False)


@app.function(volumes={"/data": volume})
def list_checkpoint_files() -> list[str]:
    """
    Runs on Modal.
    Returns a list of checkpoint filenames (not the file contents).
    """
    from pathlib import Path

    base_dir = Path("/data/checkpoints/phase1")
    if not base_dir.exists():
        return []

    return [f.name for f in base_dir.iterdir() if f.is_file()]


@app.function(volumes={"/data": volume})
def read_checkpoint_file(filename: str) -> bytes:
    """
    Runs on Modal.
    Reads a single checkpoint file and returns its bytes.
    """
    from pathlib import Path

    file_path = Path("/data/checkpoints/phase1") / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    with open(file_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(output_dir: str = "./checkpoints") -> None:
    print("=" * 70)
    print("DOWNLOADING CHECKPOINTS FROM MODAL VOLUME")
    print("=" * 70)

    # Ensure Modal is installed / configured
    try:
        import modal as _  # noqa: F401
    except ImportError:
        print("❌ Modal is not installed. Install it with: pip install modal")
        sys.exit(1)

    # First, list checkpoint files (without loading them)
    print("\n📦 Connecting to Modal volume and listing checkpoints...")
    checkpoint_files = list_checkpoint_files.remote()

    if not checkpoint_files:
        print("⚠️  No checkpoints found in volume at /data/checkpoints/phase1")
        print("   Make sure training has completed and checkpoints were saved.")
        return

    output_path = Path(output_dir)
    dest_dir = output_path / "phase1"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading {len(checkpoint_files)} checkpoint file(s) to: {dest_dir}")
    print("=" * 70)

    # Download files one at a time to avoid memory issues
    for filename in checkpoint_files:
        print(f"   Downloading {filename}...", end=" ", flush=True)
        try:
            content = read_checkpoint_file.remote(filename)
            dst_file = dest_dir / filename
            with open(dst_file, "wb") as f:
                f.write(content)
            size_mb = len(content) / (1024 * 1024)
            print(f"✅ ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ DOWNLOAD COMPLETE!")
    print("=" * 70)

