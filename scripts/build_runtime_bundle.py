"""Package inference weights and val20 demos for sharing outside ordinary Git."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs/single-gpu-study"))
from project_paths import STATE_ROOT, portable_record


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local/share/runtime-assets.zip")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = ROOT / "local/share/runtime-staging"
    staging.mkdir(parents=True, exist_ok=False)
    checkpoint = STATE_ROOT / "evaluations/open_microwave-seed7-u20000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["task"] == "open_microwave" and payload["step"] == 20000 and not payload["pilot_only"]
    inference = {key: payload[key] for key in ("model", "config", "task", "step", "pilot_only")}
    destination = staging / "state/evaluations" / checkpoint.name
    destination.parent.mkdir(parents=True)
    torch.save(inference, destination)
    del payload, inference
    demos = STATE_ROOT / "data/open_microwave/val"
    manifest = json.loads((demos / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = [(destination, "state/evaluations/" + checkpoint.name)]
    files += [(path, "state/data/open_microwave/val/" + path.relative_to(demos).as_posix())
              for path in sorted(demos.rglob("*")) if path.is_file()]
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_STORED) as z:
        for path, relative in files:
            z.write(path, relative)
    digest = hashlib.sha256()
    with args.output.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    description = {"file": args.output.name, "bytes": args.output.stat().st_size,
                   "sha256": digest.hexdigest(), "files": len(files), "validation_episodes": 20,
                   "checkpoint_step": 20000, "inference_only": True,
                   "weights_unchanged_optimizer_omitted": True,
                   "unpack_into": "local/", "ordinary_git": False}
    args.output.with_suffix(".json").write_text(json.dumps(description, indent=2))
    print(json.dumps(description, indent=2))


if __name__ == "__main__":
    main()
