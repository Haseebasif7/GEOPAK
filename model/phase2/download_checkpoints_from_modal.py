"""Download Phase 2 checkpoints from Modal volume after training."""

import sys
from pathlib import Path

import modal


# Modal app and volume
app = modal.App("geopak-download-phase2-checkpoints")
volume = modal.Volume.from_name("geopak-data", create_if_missing=False)


@app.function(volumes={"/data": volume})
def list_checkpoint_files() -> list[str]:
    """
    Runs on Modal.
    Returns a list of checkpoint filenames (not the file contents).
    """
    from pathlib import Path

    base_dir = Path("/data/checkpoints/phase2")
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

    file_path = Path("/data/checkpoints/phase2") / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    with open(file_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(output_dir: str = "./checkpoints") -> None:
    print("DOWNLOADING PHASE 2 CHECKPOINTS FROM MODAL VOLUME")

    # Ensure Modal is installed / configured
    try:
        import modal as _  # noqa: F401
    except ImportError:
        print("❌ Modal is not installed. Install it with: pip install modal")
        sys.exit(1)

    # First, list checkpoint files (without loading them)
    print("\nConnecting to Modal volume and listing checkpoints...")
    checkpoint_files = list_checkpoint_files.remote()

    if not checkpoint_files:
        print("⚠️  No checkpoints found in volume at /data/checkpoints/phase2")
        print("   Make sure training has completed and checkpoints were saved.")
        return

    output_path = Path(output_dir)
    dest_dir = output_path / "phase2"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(checkpoint_files)} checkpoint file(s) to: {dest_dir}")

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
