"""Cache shared PCC EDUs and native UniRST trees for the frozen SFU reviews."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .parser_pilot import (
    INVENTORIES, RESEARCH_REVISION, canonical_maps, digest, tree_data, write_json,
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def save_rows(path, rows):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(prepared, destination):
    provenance = json.loads((prepared / "provenance.json").read_text())
    inputs = provenance["outputs_sha256"]
    for name, checksum in inputs.items():
        if digest(prepared / name) != checksum:
            raise ValueError(f"Prepared corpus checksum mismatch: {name}")
    reviews = read_jsonl(prepared / "reviews.jsonl")
    metadata = read_jsonl(prepared / "manifest.jsonl")
    if len(reviews) != 400 or len({row["document_id"] for row in reviews}) != 400:
        raise ValueError("Expected the frozen 400-review corpus")
    documents = []
    for review, row in zip(reviews, metadata, strict=True):
        if review["document_id"] != row["document_id"]:
            raise ValueError("Review order differs from the prepared manifest")
        documents.append({**row, "text": review["text"]})
    write_json(destination, {
        "documents": documents,
        "archive_sha256": provenance["archive_sha256"],
        "inputs_sha256": inputs,
        "research_revision": RESEARCH_REVISION,
        "maps": canonical_maps(),
    })


def segment(sample, parser, output):
    from tos.unirst import SceneSplitter, UniRSTAdapter

    splitter = SceneSplitter()
    start = perf_counter()
    scenes = []
    for index, document in enumerate(sample["documents"]):
        base = {"document_id": document["document_id"]}
        try:
            texts = splitter.split(document["text"])
            if not texts or "".join("".join(t.split()) for t in texts) != "".join(document["text"].split()):
                raise ValueError("Scene splitting did not preserve the review text")
        except (ValueError, RuntimeError) as error:
            scenes.append({**base, "key": f"d{index:03}-s00", "status": "split_error",
                           "error_type": type(error).__name__, "error": str(error)})
            continue
        for scene_index, text in enumerate(texts):
            row = {**base, "key": f"d{index:03}-s{scene_index:02}", "text": text,
                   "tokens": len(splitter.tokenizer.tokenize(text))}
            try:
                edus = UniRSTAdapter._segment_with_unirst_predictor(parser.predictor, text)
                edus = UniRSTAdapter._normalise_edus(edus)
                if "".join("".join(edu.split()) for edu in edus) != "".join(text.split()):
                    raise ValueError("PCC segmentation did not preserve scene text")
                row.update(edus=edus, status="ok")
            except (ValueError, RuntimeError, IndexError) as error:
                row.update(status="segmentation_error", error_type=type(error).__name__, error=str(error))
            scenes.append(row)
        if (index + 1) % 50 == 0:
            print(f"Segmented {index + 1}/{len(sample['documents'])} reviews", flush=True)

    save_rows(output / "scenes.jsonl", scenes)
    write_json(output / "segmentation.json", {
        "scene_token_target": splitter.max_tokens,
        "seconds": perf_counter() - start,
        "scene_status": dict(Counter(row["status"] for row in scenes)),
        "scenes_sha256": digest(output / "scenes.jsonl"),
    })
    return scenes


def parse_inventory(args):
    import torch
    from isanlp_rst.parser import Parser

    inventory = args.inventory
    output = args.output
    report_path = output / f"{inventory}.json"
    tree_path = output / f"{inventory}.jsonl"
    if tree_path.exists() or report_path.exists():
        raise FileExistsError(f"Outputs already exist for {inventory}")
    sample = json.loads(args.sample.read_text())
    torch.manual_seed(42)
    start = perf_counter()
    artifact = INVENTORIES[inventory]
    directory = args.checkpoints / artifact if artifact else args.stock_snapshot
    if artifact:
        parser = Parser(relation_head_dir=str(directory), hf_model_version="unirst",
                        cuda_device=args.cuda_device)
        predictor = parser.predictor
        table = predictor.model.relation_tables[predictor.model.dataset2classifier[predictor.relinventory_idx]]
    else:
        parser = Parser(model_dir=str(directory), hf_model_version="unirst", relinventory=inventory,
                        cuda_device=args.cuda_device)
        predictor = parser.predictor
        table = predictor.relation_tables[predictor.relinventory_idx]
    load_seconds = perf_counter() - start
    allowed = {label.lower() for label in table}
    print(f"Loaded {inventory} in {load_seconds:.1f}s", flush=True)

    if inventory == "deu.rst.pcc" and not (output / "scenes.jsonl").exists():
        scenes = segment(sample, parser, output)
    else:
        scenes = read_jsonl(output / "scenes.jsonl")

    start = perf_counter()
    counts = Counter()
    def parsed_rows():
        for index, scene in enumerate(scenes, start=1):
            row = {"key": scene["key"], "document_id": scene["document_id"]}
            if scene["status"] != "ok":
                row["status"] = scene["status"]
            else:
                try:
                    root, = parser.from_edus(scene["edus"])["rst"]
                    nodes, relations = tree_data(root, scene["edus"], allowed)
                    counts.update(relations)
                    row.update(status="ok", root=root.id, nodes=nodes)
                except (ValueError, RuntimeError, IndexError, TypeError, RecursionError) as error:
                    row.update(status="parse_error", error_type=type(error).__name__, error=str(error))
            if index % 100 == 0:
                print(f"{inventory} {index}/{len(scenes)} scenes", flush=True)
            yield row

    save_rows(tree_path, parsed_rows())
    write_json(report_path, {
        "sample_sha256": digest(args.sample),
        "scenes_sha256": digest(output / "scenes.jsonl"),
        "trees_sha256": digest(tree_path),
        "source_sha256": digest(Path(__file__)),
        "weights_sha256": digest(directory / "best_weights.pt"),
        "config_sha256": digest(directory / "config.json"),
        "table_sha256": digest(directory / f"relation_table_{inventory}.txt"),
        "ordered_relation_table": list(table),
        "load_seconds": load_seconds,
        "parse_seconds": perf_counter() - start,
        "device": torch.cuda.get_device_name(args.cuda_device) if args.cuda_device >= 0 else "cpu",
        "observed_relations": dict(counts),
        "unmapped_observed": {label: counts[label] for label in sorted(counts.keys() - sample["maps"][inventory].keys())},
    })


def verify(args):
    sample = json.loads(args.sample.read_text())
    output = args.output
    segmentation = json.loads((output / "segmentation.json").read_text())
    scenes_hash = digest(output / "scenes.jsonl")
    if scenes_hash != segmentation["scenes_sha256"]:
        raise ValueError("Segmentation checksum mismatch")
    scenes = read_jsonl(output / "scenes.jsonl")
    documents = {row["document_id"]: row for row in sample["documents"]}
    scene_ids = [(row["key"], row["document_id"]) for row in scenes]
    if len(scene_ids) != len(set(scene_ids)) or set(documents) != {doc for _, doc in scene_ids}:
        raise ValueError("Segmentation identity mismatch")
    scene_texts = {}
    for scene in scenes:
        if scene["status"] != "split_error":
            scene_texts.setdefault(scene["document_id"], []).append(scene["text"])
        if scene["status"] == "ok" and "".join("".join(e.split()) for e in scene["edus"]) != "".join(scene["text"].split()):
            raise ValueError(f"EDU text differs from scene {scene['key']}")
    for doc_id, texts in scene_texts.items():
        text_hash = sha256(" ".join(documents[doc_id]["text"].split()).encode()).hexdigest()
        if text_hash != documents[doc_id]["text_sha256"] or "".join("".join(t.split()) for t in texts) != "".join(documents[doc_id]["text"].split()):
            raise ValueError(f"Scene text differs from document {doc_id}")

    complete = set(documents)
    inventories = {}
    sample_hash = digest(args.sample)
    for inventory in INVENTORIES:
        report = json.loads((output / f"{inventory}.json").read_text())
        tree_path = output / f"{inventory}.jsonl"
        if (report["sample_sha256"], report["scenes_sha256"], report["trees_sha256"]) != (
            sample_hash, scenes_hash, digest(tree_path)
        ):
            raise ValueError(f"Artifact checksum mismatch for {inventory}")
        trees = read_jsonl(tree_path)
        if scene_ids != [(row["key"], row["document_id"]) for row in trees]:
            raise ValueError(f"Scene identity/order mismatch for {inventory}")
        statuses = Counter()
        failures = Counter()
        observed = Counter()
        examples = {}
        for scene, tree in zip(scenes, trees, strict=True):
            statuses[tree["status"]] += 1
            if tree["status"] != "ok":
                complete.discard(tree["document_id"])
                failures[tree["document_id"]] += 1
                examples.setdefault(tree["status"], {k: tree.get(k) for k in ("key", "error_type", "error")})
                continue
            nodes = tree["nodes"]
            leaves = []
            pending = [str(tree["root"])]
            visited = set()
            while pending:
                key = pending.pop()
                if key in visited:
                    raise ValueError(f"Repeated tree node for {inventory}/{scene['key']}")
                visited.add(key)
                node = nodes[key]
                if "relation" in node:
                    pending.extend((str(node["right"]), str(node["left"])))
                    observed[node["relation"].lower()] += 1
                else:
                    leaves.append(node["text"])
            if leaves != scene["edus"] or visited != nodes.keys():
                raise ValueError(f"Tree coverage mismatch for {inventory}/{scene['key']}")
        if dict(observed) != report["observed_relations"]:
            raise ValueError(f"Relation counts mismatch for {inventory}")
        inventories[inventory] = {**report, "scene_status": dict(statuses),
                                  "documents_with_incomplete_trees": len(failures),
                                  "first_failure_by_status": examples}
    summary = {
        "inventory_reports": inventories,
        "archive_sha256": sample["archive_sha256"],
        "prepared_inputs_sha256": sample["inputs_sha256"],
        "research_revision": sample["research_revision"],
        "sample_sha256": sample_hash,
        "segmentation": segmentation,
        "documents": len(documents),
        "scenes": len(scenes),
        "edus": sum(len(scene.get("edus", ())) for scene in scenes),
        "max_scene_tokens": max(scene.get("tokens", 0) for scene in scenes),
        "scenes_over_token_target": sum(scene.get("tokens", 0) > segmentation["scene_token_target"] for scene in scenes),
        "complete_documents_across_inventories": len(complete),
        "incomplete_documents_by_domain_and_label": dict(Counter(
            f"{row['domain']}/{row['label']}" for doc_id, row in documents.items() if doc_id not in complete
        )),
    }
    write_json(args.report, summary)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prepared", type=Path, required=True)
    prep.add_argument("--sample", type=Path, required=True)
    run = sub.add_parser("run")
    report = sub.add_parser("verify")
    for command in (run, report):
        command.add_argument("--sample", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    run.add_argument("--inventory", choices=INVENTORIES, required=True)
    run.add_argument("--checkpoints", type=Path, required=True)
    run.add_argument("--stock-snapshot", type=Path, required=True)
    run.add_argument("--cuda-device", type=int, default=-1)
    report.add_argument("--report", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.prepared, args.sample)
    elif args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        parse_inventory(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
