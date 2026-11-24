"""Compare native coarse UniRST predictions with projected fine trees."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .export_rs3 import STANDARDS, export_tree, multinuclear_aliases
from .parse_sfu import read_jsonl
from .parser_pilot import RESEARCH_REVISION, canonical_maps, digest, write_json
from .stock_coarse import AUDITS, HERE, INVENTORIES, SNAPSHOT


def tree_splits(tree, edus, mapping=None):
    """Index binary splits by their EDU spans, independent of node IDs."""
    next_edu = 0
    splits = {}

    def visit(key):
        nonlocal next_edu
        node = tree["nodes"][str(key)]
        if "text" in node:
            if next_edu >= len(edus) or node["text"] != edus[next_edu]:
                raise ValueError(f"Tree leaves differ from the EDUs in {tree['key']}")
            next_edu += 1
            return
        start = next_edu
        visit(node["left"])
        middle = next_edu
        visit(node["right"])
        relation = node["relation"].casefold()
        if mapping is not None:
            relation = mapping[relation].casefold()
        splits[start, next_edu] = (middle, node["nuclearity"], relation)

    visit(tree["root"])
    if next_edu != len(edus) or len(splits) != len(edus) - 1:
        raise ValueError(f"Tree structure differs from the EDUs in {tree['key']}")
    return splits


def compare_trees(stock, fine, scenes, mapping):
    totals = Counter()
    labels = {"stock": Counter(), "mapped": Counter()}
    example = None
    for original, projected, scene in zip(stock, fine, scenes, strict=True):
        for tree in (original, projected):
            if (tree["key"], tree["document_id"], tree["status"]) != (
                    scene["key"], scene["document_id"], "ok"):
                raise ValueError("A tree differs from the verified scene")
        edus = scene["edus"]
        a, b = tree_splits(original, edus), tree_splits(projected, edus, mapping)
        labels["stock"].update(label for _, _, label in a.values())
        labels["mapped"].update(label for _, _, label in b.values())
        shapes, roles, full = 0, 0, 0
        for span in a.keys() & b.keys():
            left, right = a[span], b[span]
            shapes += left[0] == right[0]
            roles += left[:2] == right[:2]
            full += left == right
        totals.update(scenes=1, internal_splits=len(a), same_split=shapes,
                      same_split_and_nuclearity=roles, same_split_nuclearity_and_label=full)
        if len(edus) == 1:
            totals["single_edu_scenes"] += 1
            continue
        totals.update(multiedu_scenes=1, same_shape_scenes=shapes == len(a),
                      same_shape_and_nuclearity_scenes=roles == len(a),
                      identical_scenes=full == len(a))
        if 3 <= len(edus) <= 5 and a[0, len(edus)][0] != b[0, len(edus)][0]:
            if example is None or (len(edus), scene["key"]) < (example["edus"], example["scene"]):
                example = {"scene": scene["key"], "edus": len(edus),
                           "stock_root_split_after_edu": a[0, len(edus)][0],
                           "mapped_root_split_after_edu": b[0, len(edus)][0]}
    return {"counts": dict(totals), "observed_labels": labels,
            "observed_only_in_stock": sorted(labels["stock"].keys() - labels["mapped"].keys()),
            "observed_only_in_mapped": sorted(labels["mapped"].keys() - labels["stock"].keys()),
            "example": example}


def export_stock(folder, trees, scenes):
    from isanlp.annotation_rst import DiscourseUnit

    aliases = multinuclear_aliases(trees, None)
    folder.mkdir(parents=True)
    for tree, scene in zip(trees, scenes, strict=True):
        export_tree(folder / f"{scene['key']}.rs3", tree, scene, None, aliases, DiscourseUnit)
    return {alias: label for label, alias in aliases.items()}


def checked_file(path, sha256):
    if digest(path) != sha256:
        raise ValueError(f"Checksum mismatch for {path}")
    return read_jsonl(path)


def compare(args):
    if args.report.exists() or (args.rs3_output and args.rs3_output.exists()):
        raise FileExistsError("Expected fresh comparison outputs")
    maps = canonical_maps()
    stock_reports = {name: json.loads((args.stock / name / "parser.json").read_text()) for name in INVENTORIES}
    for name, report in stock_reports.items():
        if (report["inventory"], report["stock_snapshot"]) != (name, SNAPSHOT):
            raise ValueError(f"Unexpected stock run for {name}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.rs3_output:
        args.rs3_output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with TemporaryDirectory(dir=args.rs3_output.parent if args.rs3_output else None) as temporary:
        staged = Path(temporary) / "export"
        for corpus, trees_path in (("SFU", args.sfu_trees), ("Amazon", args.amazon_trees)):
            audit = json.loads((HERE / AUDITS[corpus]).read_text())
            if audit["research_revision"] != RESEARCH_REVISION:
                raise ValueError("The fine trees use a different mapping revision")
            scenes_hash = audit["segmentation"]["scenes_sha256"]
            scenes = checked_file(trees_path / "scenes.jsonl", scenes_hash)
            results[corpus] = {}
            aliases = {}
            for name in INVENTORIES:
                reference = audit["inventory_reports"][name]
                fine = checked_file(trees_path / f"{name}.jsonl", reference["trees_sha256"])
                stock_meta = stock_reports[name]["corpora"][corpus]
                if stock_meta["scenes_sha256"] != scenes_hash:
                    raise ValueError("Stock parser and mapped trees have different segmentation")
                stock = checked_file(args.stock / name / f"{corpus}.jsonl", stock_meta["trees_sha256"])
                results[corpus][name] = compare_trees(stock, fine, scenes, maps[name])
                results[corpus][name]["fine_trees_sha256"] = reference["trees_sha256"]
                if args.rs3_output:
                    aliases[STANDARDS[name]] = export_stock(staged / corpus / STANDARDS[name] / "stock-coarse",
                                                           stock, scenes)
            if args.rs3_output:
                write_json(staged / corpus / "rs3-relation-aliases.json", aliases)
                with (staged / corpus / "scene-index.tsv").open("x", encoding="utf-8", newline="") as stream:
                    index = csv.writer(stream, delimiter="\t", lineterminator="\n")
                    index.writerow(("scene_id", "document_id", "edu_count"))
                    index.writerows((row["key"], row["document_id"], len(row["edus"])) for row in scenes)
        if args.rs3_output:
            (staged / "README.md").write_text(
                "# Stock-coarse UniRST parses\n\n"
                "These GUM/eRST and RST-DT parses use the stock UniRST release, with the same PCC EDUs "
                "as the sentiment experiment. Each file covers one scene. An SFU review can have "
                "several scenes. The index gives the review for each file.\n\n"
                "The alias list reverses any extra multinuclear label suffixes used by the RS3 exporter.\n",
                encoding="utf-8",
            )
            staged.rename(args.rs3_output)
    report = {"mapping_revision": RESEARCH_REVISION, "stock_parsers": stock_reports, "comparisons": results,
              "comparison_source_sha256": digest(Path(__file__))}
    write_json(args.report, report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stock", type=Path, required=True)
    cli.add_argument("--sfu-trees", type=Path, required=True)
    cli.add_argument("--amazon-trees", type=Path, required=True)
    cli.add_argument("--report", type=Path, required=True)
    cli.add_argument("--rs3-output", type=Path)
    compare(cli.parse_args())


if __name__ == "__main__":
    main()
