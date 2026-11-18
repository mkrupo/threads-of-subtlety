"""Cache one frozen CardiffNLP score vector for each verified SFU EDU."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from .parser_pilot import digest, write_json


RESULTS = Path(__file__).with_name("results")
LABELS = ("negative", "neutral", "positive")
BATCH_SIZE = 64
THREADS = 4
VERSIONS = {"numpy": "2.2.6", "onnxruntime": "1.22.1", "tokenizers": "0.21.4"}
MODEL_FILES = ("model.onnx", "tokenizer.json", "config.json")


def read_scenes(path):
    audit = json.loads((RESULTS / "sfu-full-parser-audit.json").read_text())
    expected = audit["segmentation"]["scenes_sha256"]
    if digest(path) != expected:
        raise ValueError("Shared scene file differs from the verified SFU parse")
    with path.open(encoding="utf-8") as stream:
        scenes = [json.loads(line) for line in stream]
    if any(scene["status"] != "ok" or not scene["edus"] for scene in scenes):
        raise ValueError("Expected successful segmentation for every scene")
    counts = (
        len({scene["document_id"] for scene in scenes}),
        len(scenes),
        sum(len(scene["edus"]) for scene in scenes),
    )
    if counts != (audit["documents"], audit["scenes"], audit["edus"]):
        raise ValueError("Scene or EDU counts differ from the verified SFU parse")
    return scenes, expected


def model_provenance(directory):
    audit = json.loads((RESULTS / "sfu-edu-scorer-check.json").read_text())
    hashes = {filename: digest(directory / filename) for filename in MODEL_FILES}
    for filename, checksum in hashes.items():
        if checksum != audit[f"{filename.split('.')[0]}_sha256"]:
            raise ValueError(f"{filename} differs from the tested scorer")
    return {"id": audit["model"], "revision": audit["revision"], "sha256": hashes}


def normalize(text):
    # Follow the model card's mention and URL normalization.
    return " ".join(
        "@user" if word.startswith("@") and len(word) > 1
        else "http" if word.startswith("http")
        else word
        for word in text.split(" ")
    )


def predict(edus, model_dir):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import tokenizers

    versions = {"numpy": np.__version__, "onnxruntime": ort.__version__,
                "tokenizers": tokenizers.__version__}
    if versions != VERSIONS:
        raise ValueError(f"Use the recorded scorer package versions: {VERSIONS}")
    config = json.loads((model_dir / "config.json").read_text())
    labels = tuple(config["id2label"][str(i)].lower() for i in range(len(LABELS)))
    if labels != LABELS:
        raise ValueError(f"Unexpected scorer classes: {labels}")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    pad_token = "<pad>"
    pad_id = tokenizer.token_to_id(pad_token)
    # Group nearby lengths to reduce padding, then restore input order.
    encoded = tokenizer.encode_batch([normalize(edu) for edu in edus])
    order = sorted(range(len(edus)), key=lambda i: len(encoded[i].ids))
    if len(encoded[order[-1]].ids) > 512:
        raise ValueError("An EDU exceeds the scorer's context window")
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    scores = np.empty((len(edus), len(LABELS)), dtype=np.float32)
    for start in range(0, len(edus), BATCH_SIZE):
        ids = order[start:start + BATCH_SIZE]
        batch = [encoded[index] for index in ids]
        width = len(batch[-1].ids)
        tokens = np.full((len(batch), width), pad_id, dtype=np.int64)
        attention = np.zeros_like(tokens)
        for position, item in enumerate(batch):
            tokens[position, :len(item.ids)] = item.ids
            attention[position, :len(item.ids)] = item.attention_mask
        logits, = session.run(None, {"input_ids": tokens, "attention_mask": attention})
        if logits.shape != (len(batch), len(LABELS)):
            raise ValueError(f"Unexpected model output shape: {logits.shape}")
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores[ids] = probabilities / probabilities.sum(axis=1, keepdims=True)
        completed = start + len(batch)
        if completed // 4096 > start // 4096 or completed == len(edus):
            print(f"Scored {completed:,}/{len(edus):,} EDUs", flush=True)
    return scores


def verify(scenes_path, output):
    import numpy as np

    scenes, scene_hash = read_scenes(scenes_path)
    manifest = json.loads((output / "manifest.json").read_text())
    score_path = output / "scores.jsonl"
    if (scene_hash != manifest["scenes_sha256"]
            or digest(score_path) != manifest["scores_sha256"]
            or manifest["labels"] != list(LABELS)):
        raise ValueError("Score or scene provenance mismatch")
    with score_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    counts = Counter()
    for scene, row in zip(scenes, rows, strict=True):
        if (row["key"], row["document_id"]) != (scene["key"], scene["document_id"]):
            raise ValueError(f"Scene identity mismatch at {scene['key']}")
        values = np.asarray(row["probabilities"])
        if (values.shape != (len(scene["edus"]), len(LABELS))
                or not np.isfinite(values).all()
                or not ((values >= 0) & (values <= 1)).all()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError(f"Invalid probabilities at {scene['key']}")
        counts.update(LABELS[index] for index in values.argmax(axis=1))
    if sum(counts.values()) != manifest["edus"] or dict(counts) != manifest["top_labels"]:
        raise ValueError("Scorer counts differ from the manifest")
    print(f"Verified {sum(counts.values()):,} scores across {len(scenes):,} scenes", flush=True)


def run(scenes_path, model_dir, output):
    output.mkdir(parents=True)
    scenes, scene_hash = read_scenes(scenes_path)
    model = model_provenance(model_dir)
    start = perf_counter()
    scores = predict([edu for scene in scenes for edu in scene["edus"]], model_dir)
    score_path = output / "scores.jsonl"
    offset = 0
    with score_path.open("x", encoding="utf-8") as stream:
        for scene in scenes:
            end = offset + len(scene["edus"])
            row = {"key": scene["key"], "document_id": scene["document_id"],
                   "probabilities": scores[offset:end].tolist()}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            offset = end
    write_json(output / "manifest.json", {
        "model": model,
        "scenes_sha256": scene_hash,
        "scores_sha256": digest(score_path),
        "source_sha256": digest(Path(__file__)),
        "runtime": {"packages": VERSIONS, "provider": "CPUExecutionProvider",
                    "batch_size": BATCH_SIZE, "intra_op_threads": THREADS},
        "labels": list(LABELS),
        "documents": len({scene["document_id"] for scene in scenes}),
        "scenes": len(scenes),
        "edus": len(scores),
        "top_labels": dict(Counter(LABELS[index] for index in scores.argmax(axis=1))),
        "seconds": perf_counter() - start,
    })
    verify(scenes_path, output)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scenes", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-dir", type=Path)
    mode.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    if args.verify:
        verify(args.scenes, args.output)
    else:
        run(args.scenes, args.model_dir, args.output)


if __name__ == "__main__":
    main()
