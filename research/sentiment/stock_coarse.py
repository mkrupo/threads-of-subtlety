"""Parse shared PCC EDUs with the released coarse GUM and RST-DT configurations."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from .parse_sfu import parse_with_supplied_edu_text, read_jsonl, save_rows
from .parser_pilot import digest, tree_data, write_json


HERE = Path(__file__).parent
INVENTORIES = ("eng.erst.gum", "eng.rst.rstdt")
AUDITS = {
    "SFU": "results/sfu-full-parser-audit.json",
    "Amazon": "replication/results/amazon/parser-audit.json",
}
SNAPSHOT = "b99b1c0f39435971b9daaa450e8476ef7b1a554a"


def run(args):
    import torch
    from isanlp_rst.parser import Parser

    paths = {"SFU": args.sfu_scenes, "Amazon": args.amazon_scenes}
    audits = {corpus: json.loads((HERE / path).read_text()) for corpus, path in AUDITS.items()}
    source = {}
    for corpus, path in paths.items():
        actual = digest(path)
        if actual != audits[corpus]["segmentation"]["scenes_sha256"]:
            raise ValueError(f"The {corpus} EDUs differ from the verified segmentation")
        source[corpus] = actual
    snapshot = args.stock_snapshot
    model_hashes = {name: digest(snapshot / name) for name in (
        "best_weights.pt", "config.json", f"relation_table_{args.inventory}.txt",
    )}
    for audit in audits.values():
        reference = audit["inventory_reports"]["deu.rst.pcc"]
        if (model_hashes["best_weights.pt"], model_hashes["config.json"]) != (
                reference["weights_sha256"], reference["config_sha256"]):
            raise ValueError("The stock weights or config differ from the verified UniRST release")

    args.output.mkdir(parents=True)
    torch.manual_seed(42)
    start = perf_counter()
    parser = Parser(model_dir=str(snapshot), hf_model_version="unirst",
                    relinventory=args.inventory, cuda_device=args.cuda_device)
    predictor = parser.predictor
    table = list(predictor.relation_tables[predictor.relinventory_idx])
    allowed = {label.lower() for label in table}
    print(f"Loaded stock {args.inventory} after {perf_counter() - start:.1f}s", flush=True)
    results = {}
    for corpus, path in paths.items():
        scenes = read_jsonl(path)
        output = args.output / f"{corpus}.jsonl"
        retries = Counter()
        saved = []
        prior_path = None
        if args.resume_from:
            for candidate in (args.resume_from / output.name, args.resume_from / output.with_suffix(".partial").name):
                if candidate.exists():
                    prior_path = candidate
                    saved = read_jsonl(candidate)
                    break
            if len(saved) > len(scenes):
                raise ValueError(f"Extra saved trees for {corpus}")
        controls = {1, len(saved) // 2, len(saved)}
        start = perf_counter()

        def parse_scene(scene):
            edus = scene["edus"]
            try:
                prediction = parser.from_edus(edus)
            except (ValueError, RuntimeError) as error:
                if not str(error).startswith(("Cannot align segment ",
                                              "The produced segmentation does not match the provided EDUs.")):
                    raise
                prediction = parse_with_supplied_edu_text(parser, edus)
                retries[type(error).__name__] += 1
            root, = prediction["rst"]
            nodes, _ = tree_data(root, edus, allowed)
            return {"key": scene["key"], "document_id": scene["document_id"],
                    "status": "ok", "root": root.id, "nodes": {str(key): value for key, value in nodes.items()}}

        def parsed_rows():
            for i, scene in enumerate(scenes, start=1):
                if scene["status"] != "ok":
                    raise ValueError(f"Scene {scene['key']} is missing its verified EDUs")
                if i <= len(saved):
                    row = saved[i - 1]
                    if (row["key"], row["document_id"], row["status"]) != (scene["key"], scene["document_id"], "ok"):
                        raise ValueError(f"Saved tree differs from scene {scene['key']}")
                    if i in controls and row != parse_scene(scene):
                        raise ValueError(f"Saved tree could not be reproduced for scene {scene['key']}")
                else:
                    row = parse_scene(scene)
                if i % 100 == 0 or i == len(scenes):
                    print(f"{args.inventory} {corpus}: {i}/{len(scenes)}", flush=True)
                yield row

        save_rows(output, parsed_rows())
        results[corpus] = {
            "scenes": len(scenes), "edus": sum(len(row["edus"]) for row in scenes),
            "scenes_sha256": source[corpus], "trees_sha256": digest(output),
            "alignment_retries_this_attempt": sum(retries.values()), "parse_seconds": perf_counter() - start,
            "resumed_scenes": len(saved), "resume_sha256": digest(prior_path) if prior_path else None,
        }
    write_json(args.output / "parser.json", {
        "inventory": args.inventory, "stock_snapshot": SNAPSHOT, "model_hashes": model_hashes,
        "ordered_relation_table": table, "corpora": results,
        "source_hashes": {name: digest(HERE / name)
                          for name in ("stock_coarse.py", "parse_sfu.py", "parser_pilot.py")},
        "runtime": {"torch": torch.__version__, "transformers": version("transformers"),
                    "device": torch.cuda.get_device_name(args.cuda_device) if args.cuda_device >= 0 else "cpu"},
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sfu-scenes", type=Path, required=True)
    cli.add_argument("--amazon-scenes", type=Path, required=True)
    cli.add_argument("--stock-snapshot", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--inventory", choices=INVENTORIES, required=True)
    cli.add_argument("--cuda-device", type=int, default=-1)
    cli.add_argument("--resume-from", type=Path, help="Prior output for this inventory")
    run(cli.parse_args())


if __name__ == "__main__":
    main()
