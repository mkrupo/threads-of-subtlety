"""Score every EDU of a verified parse with the frozen CardiffNLP model.

score_sfu.py checks its inputs against the SFU parser audit. This wrapper uses
the same model checks, normalization and batching for another cohort and
verifies the scene file against that cohort's own parser audit instead.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from .parser_pilot import digest, write_json
from .score_sfu import BATCH_SIZE, LABELS, THREADS, VERSIONS, model_provenance, predict


def run(args):
    audit = json.loads(args.audit.read_text())
    scenes_path = args.trees / "scenes.jsonl"
    if digest(scenes_path) != audit["segmentation"]["scenes_sha256"]:
        raise ValueError("The scene file differs from the parser audit")
    with scenes_path.open(encoding="utf-8") as stream:
        scenes = [json.loads(line) for line in stream]
    if any(scene["status"] != "ok" or not scene["edus"] for scene in scenes):
        raise ValueError("Expected successful segmentation for every scene")
    if (len({s["document_id"] for s in scenes}), len(scenes), sum(len(s["edus"]) for s in scenes)) != (
            audit["documents"], audit["scenes"], audit["edus"]):
        raise ValueError("Scene or EDU counts differ from the parser audit")
    model = model_provenance(args.model_dir)
    args.output.mkdir(parents=True)
    start = perf_counter()
    scores = predict([edu for scene in scenes for edu in scene["edus"]], args.model_dir)
    path = args.output / "scores.jsonl"
    offset = 0
    with path.open("x", encoding="utf-8") as stream:
        for scene in scenes:
            end = offset + len(scene["edus"])
            stream.write(json.dumps({"key": scene["key"], "document_id": scene["document_id"],
                                     "probabilities": scores[offset:end].tolist()}, allow_nan=False) + "\n")
            offset = end
    write_json(args.output / "manifest.json", {
        "model": model, "scenes_sha256": audit["segmentation"]["scenes_sha256"],
        "parser_audit_sha256": digest(args.audit), "scores_sha256": digest(path),
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("score_cohort.py", "score_sfu.py")},
        "runtime": {"packages": VERSIONS, "provider": "CPUExecutionProvider", "batch_size": BATCH_SIZE,
                    "intra_op_threads": THREADS},
        "labels": list(LABELS), "documents": audit["documents"], "scenes": len(scenes), "edus": len(scores),
        "top_labels": dict(Counter(LABELS[i] for i in scores.argmax(axis=1))),
        "seconds": perf_counter() - start,
    })
    print(f"Scored {len(scores):,} EDUs in {len(scenes):,} scenes", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trees", type=Path, required=True, help="Verified parse folder with scenes.jsonl")
    cli.add_argument("--audit", type=Path, required=True, help="That parse's parser-audit.json")
    cli.add_argument("--model-dir", type=Path, required=True, help="Staged CardiffNLP model.onnx, tokenizer and config")
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
