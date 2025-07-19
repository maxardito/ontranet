import subprocess
from pathlib import Path

# Labels for the 5 audio streams
STEM_LABELS = ["mixture", "drums", "bass", "other", "vocal"]

def extract_stems_from_file(filepath: Path, index: int, output_dir: Path):
    for stream_index, label in enumerate(STEM_LABELS):
        # Skip mixture labels
        if(label == "mixture"):
            continue
        output_path = output_dir / f"{index}-{label}.mp3"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(filepath),
            "-map", f"0:{stream_index}",
            "-c:a", "libmp3lame",
            "-q:a", "2",  # high quality MP3
            str(output_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed on {filepath.name}, stream {stream_index} ({label}):\n{result.stderr.decode()}"
            )
        print(f"✓ Extracted {output_path.name}")

def batch_extract_stems(input_dir: Path, output_dir: Path):
    input_dir = input_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_files = sorted(input_dir.glob("*.stem.mp4"))
    for i, filepath in enumerate(stem_files, start=1):
        print(f"\nProcessing {filepath.name} ({i}/{len(stem_files)})")
        extract_stems_from_file(filepath, i, output_dir)
