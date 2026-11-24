"""Cache frozen mDeBERTa CLS vectors in the verified EDU order."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .parse_sfu import read_jsonl
from .parser_pilot import digest, write_json


MODEL = "microsoft/mdeberta-v3-base"
REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"


def encode(scenes_path, audit_path, output, device, batch_size):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    audit = json.loads(audit_path.read_text())
    scenes_hash = digest(scenes_path)
    if scenes_hash != audit["segmentation"]["scenes_sha256"]:
        raise ValueError("The scene file differs from the parser audit")
    scenes = read_jsonl(scenes_path)
    edus = [edu for scene in scenes for edu in scene["edus"]]
    if len(scenes) != audit["scenes"] or len(edus) != audit["edus"]:
        raise ValueError("The scene or EDU count differs from the parser audit")
    if output.exists():
        raise FileExistsError(output)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=False)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    print(f"Loaded {MODEL}@{REVISION} on {device}", flush=True)
    tokenized = tokenizer(edus, truncation=False)
    lengths = np.asarray([len(row) for row in tokenized["input_ids"]])
    if lengths.max() > model.config.max_position_embeddings:
        raise ValueError("An EDU exceeds the encoder's context window")
    order = np.argsort(lengths, kind="stable")
    vectors = np.empty((len(edus), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = tokenizer.pad(
                {name: [rows[i] for i in indices] for name, rows in tokenized.items()},
                return_tensors="pt",
            ).to(device)
            vectors[indices] = model(**batch).last_hidden_state[:, 0].cpu().numpy()
            if start == 0 or start // 4096 != (start + len(indices)) // 4096:
                print(f"Encoded {start + len(indices):,}/{len(edus):,} EDUs", flush=True)
    if not np.isfinite(vectors).all():
        raise ValueError("The encoder produced non-finite vectors")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "encoded"
        staged.mkdir()
        vectors_path = staged / "embeddings.npy"
        np.save(vectors_path, vectors, allow_pickle=False)
        write_json(staged / "manifest.json", {
            "model": MODEL, "revision": REVISION, "pooling": "CLS",
            "scene_order": "rows from scenes.jsonl, then EDUs in scene order",
            "scenes_sha256": scenes_hash, "embeddings_sha256": digest(vectors_path),
            "shape": list(vectors.shape), "max_tokens": int(lengths.max()),
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "device": device, "batch_size": batch_size},
        })
        staged.rename(output)
    print(f"Cached {len(edus):,} EDU vectors in {output}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scenes", type=Path, required=True)
    cli.add_argument("--audit", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--batch-size", type=int, default=64)
    args = cli.parse_args()
    encode(args.scenes, args.audit, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
