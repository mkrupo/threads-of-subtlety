"""Download and verify only the pinned CardiffNLP scorer files.

prepare-model-assets.py also stages and verifies every parser checkpoint.
Use this helper on a machine that only scores EDUs. It writes the same
``cardiff/`` layout: links to the pinned revision's quantized ONNX model,
tokenizer and config, each checked against the published SHA-256.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from huggingface_hub import snapshot_download

RESULTS = Path(__file__).resolve().parents[1] / "results"
FILES = {"config.json": "config.json", "tokenizer.json": "tokenizer.json", "model.onnx": "onnx/model_quantized.onnx"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", required=True, type=Path)
    args = cli.parse_args()
    scorer = json.loads((RESULTS / "sfu-edu-scores-v1.json").read_text())["model"]
    model = Path(snapshot_download(scorer["id"], revision=scorer["revision"], cache_dir=args.assets / "hf" / "hub",
                                   allow_patterns=list(FILES.values())))
    target = args.assets / "cardiff"
    target.mkdir(parents=True, exist_ok=True)
    for name, relative in FILES.items():
        source = model / relative
        if sha256(source.read_bytes()).hexdigest() != scorer["sha256"][name]:
            raise ValueError(f"Unexpected SHA-256 for {source}")
        link = target / name
        if link.is_symlink() or link.exists():
            if link.resolve(strict=True) != source.resolve(strict=True):
                raise FileExistsError(f"{link} points to a different file")
        else:
            link.symlink_to(source)
    print(f"Verified {scorer['id']}@{scorer['revision']} in {target}")


if __name__ == "__main__":
    main()
