"""Check the four UniRST configurations on shared PCC EDUs."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import xml.etree.ElementTree as ET


RESEARCH_REVISION = "f06309b62cbc53dce0c98668cac225666dfb64d7"
INVENTORIES = {
    "deu.rst.pcc": None,
    "nld.rst.nldt": None,
    "eng.erst.gum": "erst",
    "eng.rst.rstdt": "rstdt",
}
FIXTURES = [
    {"document_id": "fixture-one-edu", "text": "Excellent!"},
    {"document_id": "fixture-multi-edu", "text": (
        "I liked the screen, but the battery died by noon. "
        "I asked for a replacement because I needed a phone for work. "
        "The replacement lasted all day."
    )},
]


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def canonical_maps():
    def source(path):
        return subprocess.check_output(["git", "show", f"{RESEARCH_REVISION}:{path}"], text=True)

    def mapping(path, variable):
        for node in ast.parse(source(path)).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable for target in node.targets
            ):
                return ast.literal_eval(node.value)
        raise ValueError(f"Missing {variable} in {path}")

    pcc_nldt = mapping("scripts/canonical/ablate_m3_granularity.py", "RELATION_MAP")
    return {
        "deu.rst.pcc": pcc_nldt,
        "nld.rst.nldt": pcc_nldt,
        "eng.erst.gum": mapping(
            "scripts/canonical/ablate_full_inventory_granularity.py", "FINE_TO_PARENT"
        ),
        "eng.rst.rstdt": json.loads(source("docs/eng.rst.rstdt.new_to_coarse.json")),
    }


def prepare(path, prepared):
    documents = FIXTURES
    corpus = None
    if prepared:
        corpus = json.loads((prepared / "provenance.json").read_text())
        reviews_path = prepared / "reviews.jsonl"
        if digest(reviews_path) != corpus["outputs_sha256"]["reviews.jsonl"]:
            raise ValueError("Prepared review checksum differs from provenance")
        with reviews_path.open(encoding="utf-8") as stream:
            reviews = [json.loads(line) for line in stream]
        documents = []
        for label in (0, 1):
            ordered = sorted(
                (row for row in reviews if row["label"] == label),
                key=lambda row: (row["whitespace_tokens"], row["document_id"]),
            )
            for quantile in (10, 50, 90):
                row = ordered[(len(ordered) - 1) * quantile // 100]
                documents.append({**row, "length_percentile": quantile})
    write_json(path, {
        "documents": documents,
        "archive_sha256": corpus["archive_sha256"] if corpus else None,
        "research_revision": RESEARCH_REVISION,
        "maps": canonical_maps(),
    })


def tree_data(root, edus, allowed):
    leaves, relations, nodes = [], Counter(), {}

    def visit(node):
        if node.id in nodes:
            raise ValueError(f"Repeated discourse unit ID {node.id}")
        record = {}
        nodes[node.id] = record
        if node.left is None and node.right is None:
            leaves.append(node.text)
            record["text"] = node.text
            return
        label = f"{node.relation}_{node.nuclearity}"
        if node.left is None or node.right is None or label.lower() not in allowed:
            raise ValueError(f"Invalid internal node {node.id}: {label}")
        record.update(left=node.left.id, right=node.right.id,
                      relation=node.relation, nuclearity=node.nuclearity)
        relations[node.relation.lower()] += 1
        visit(node.left)
        visit(node.right)

    visit(root)
    if leaves != edus or sum(relations.values()) != len(edus) - 1:
        raise ValueError("Binary parser tree differs from the supplied EDUs")
    return nodes, relations


def verify_rs3(path, root_id, nodes, edus):
    xml = ET.parse(path).getroot()
    body = list(xml.find("body"))
    elements = {int(element.attrib["id"]) - 1: element for element in body}
    if len(elements) != len(body) or set(elements) != set(nodes):
        raise ValueError("RS3 node IDs differ from the parser tree")
    if [element.text for element in body if element.tag == "segment"] != edus:
        raise ValueError("RS3 segment text/order differs from the supplied EDUs")
    if "parent" in elements[root_id].attrib:
        raise ValueError("RS3 root has a parent")
    for node_id, node in nodes.items():
        element = elements[node_id]
        if "text" in node:
            if element.tag != "segment" or element.text != node["text"]:
                raise ValueError(f"RS3 segment {node_id} differs from the parser tree")
            continue
        nuclearity = node["nuclearity"]
        kind = "multinuc" if nuclearity == "NN" else "span"
        if element.tag != "group" or element.attrib["type"] != kind:
            raise ValueError(f"RS3 group {node_id} differs from the parser tree")
        for side, other, role in zip(("left", "right"), ("right", "left"), nuclearity):
            parent = node[other] if role == "S" else node_id
            relation = node["relation"] if role == "S" or nuclearity == "NN" else "span"
            child = elements[node[side]]
            if int(child.attrib["parent"]) != parent + 1 or child.attrib["relname"] != relation:
                raise ValueError(f"RS3 attachment differs at node {node[side]}")


def run(args):
    import torch
    from isanlp_rst.parser import Parser
    from tos import unirst

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    torch.manual_seed(42)
    splitter = unirst.SceneSplitter()
    scenes = []
    for document_index, document in enumerate(sample["documents"]):
        texts = splitter.split(document["text"])
        if "".join("".join(text.split()) for text in texts) != "".join(document["text"].split()):
            raise ValueError(f"Scene splitting lost text in {document['document_id']}")
        for scene_index, text in enumerate(texts):
            tokens = len(splitter.tokenizer.tokenize(text))
            scenes.append({
                "key": f"d{document_index:02}-s{scene_index:02}",
                "document_id": document["document_id"],
                "tokens": tokens,
                "text": text,
            })

    results = {}
    for inventory, artifact in INVENTORIES.items():
        print(f"Loading {inventory}", flush=True)
        start = perf_counter()
        if artifact:
            artifact_dir = args.checkpoints / artifact
            parser = Parser(
                relation_head_dir=str(artifact_dir), hf_model_version="unirst",
                cuda_device=args.cuda_device,
            )
        else:
            artifact_dir = args.stock_snapshot
            parser = Parser(
                model_dir=str(artifact_dir), hf_model_version="unirst",
                relinventory=inventory, cuda_device=args.cuda_device,
            )
        predictor = parser.predictor
        index = predictor.model.dataset2classifier[predictor.relinventory_idx]
        table = (predictor.model.relation_tables[index] if artifact
                 else predictor.relation_tables[predictor.relinventory_idx])
        allowed = {label.lower() for label in table}
        load_seconds = perf_counter() - start

        if inventory == "deu.rst.pcc":
            start = perf_counter()
            for scene in scenes:
                edus = unirst.UniRSTAdapter._segment_with_unirst_predictor(predictor, scene["text"])
                if "".join("".join(edu.split()) for edu in edus) != "".join(scene["text"].split()):
                    raise ValueError(f"PCC segmenter lost text in {scene['key']}")
                scene["edus"] = unirst.UniRSTAdapter._normalise_edus(edus)
            segmentation_seconds = perf_counter() - start
            write_json(args.output / "scenes.json", scenes)

        directory = args.output / inventory
        directory.mkdir()
        observed = Counter()
        records = []
        for scene in scenes:
            start = perf_counter()
            root, = parser.from_edus(scene["edus"])["rst"]
            tree, relations = tree_data(root, scene["edus"], allowed)
            file = directory / f"{scene['key']}.rs3"
            root.to_rs3(str(file))
            verify_rs3(file, root.id, tree, scene["edus"])
            write_json(file.with_suffix(".json"), {"root": root.id, "nodes": tree})
            records.append({
                "scene": scene["key"], "relations": dict(relations),
                "seconds": perf_counter() - start, "rs3_sha256": digest(file),
                "tree_sha256": digest(file.with_suffix(".json")),
            })
            observed.update(relations)

        mapping = sample["maps"][inventory]
        missing = set(observed) - mapping.keys()
        coarse = Counter()
        for relation in observed.keys() & mapping.keys():
            coarse[mapping[relation]] += observed[relation]
        results[inventory] = {
            "load_seconds": load_seconds,
            "weights_sha256": digest(artifact_dir / "best_weights.pt"),
            "config_sha256": digest(artifact_dir / "config.json"),
            "table_sha256": digest(artifact_dir / f"relation_table_{inventory}.txt"),
            "ordered_relation_table": list(table),
            "unmapped_inventory_labels": sorted(
                {label.rsplit("_", 1)[0] for label in allowed} - mapping.keys()
            ),
            "unmapped_observed": {label: observed[label] for label in sorted(missing)},
            "observed_relations": dict(observed),
            "coarse_counts": dict(coarse) if not missing else None,
            "scenes": records,
        }
        del predictor, parser
        if args.cuda_device >= 0:
            torch.cuda.empty_cache()
        print(f"Verified {len(records)} {inventory} trees", flush=True)

    write_json(args.output / "report.json", {
        "sample_sha256": digest(args.sample),
        "pilot_source_sha256": digest(Path(__file__)),
        "adapter_source_sha256": digest(Path(unirst.__file__)),
        "archive_sha256": sample["archive_sha256"],
        "research_revision": sample["research_revision"],
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "huggingface-hub", "isanlp", "isanlp-rst",
            "rstconverter", "sentsplit",
        )},
        "device": torch.cuda.get_device_name(args.cuda_device) if args.cuda_device >= 0 else "cpu",
        "scene_limit": splitter.max_tokens,
        "scene_count": len(scenes),
        "segmentation_seconds": segmentation_seconds,
        "segmentation_sha256": digest(args.output / "scenes.json"),
        "documents": [
            {key: value for key, value in row.items() if key not in {"text", "text_normalized"}}
            for row in sample["documents"]
        ],
        "inventories": results,
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("prepare")
    sample.add_argument("--prepared", type=Path)
    sample.add_argument("--sample", type=Path, required=True)
    pilot = commands.add_parser("run")
    for name in ("sample", "output", "checkpoints", "stock-snapshot"):
        pilot.add_argument(f"--{name}", type=Path, required=True)
    pilot.add_argument("--cuda-device", type=int, default=-1)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.sample, args.prepared)
    else:
        run(args)


if __name__ == "__main__":
    main()
