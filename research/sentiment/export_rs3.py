"""Export verified UniRST scene trees with the pilot's RS3 serializer."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from .parse_sfu import read_jsonl
from .parser_pilot import RESEARCH_REVISION, canonical_maps, digest, verify_rs3, write_json


STANDARDS = {"deu.rst.pcc": "PCC", "nld.rst.nldt": "NLDT",
             "eng.erst.gum": "GUM-eRST", "eng.rst.rstdt": "RST-DT"}
CORPORA = {
    "SFU": "https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html",
    "Amazon": "https://amazon-reviews-2023.github.io/",
    "UCI": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
}


def projected_name(label, mapping):
    return label if mapping is None else mapping[label.lower()]


def multinuclear_aliases(trees, mapping):
    roles = defaultdict(set)
    for tree in trees:
        for node in tree["nodes"].values():
            if "relation" in node:
                label = projected_name(node["relation"], mapping)
                roles[label].add(node["nuclearity"] == "NN")
    aliases = {}
    taken = {label.casefold() for label in roles}
    for label, types in sorted(roles.items()):
        if len(types) == 2:
            alias = f"{label}-multinuc"
            while alias.casefold() in taken:
                alias += "-multinuc"
            aliases[label] = alias
            taken.add(alias.casefold())
    return aliases


def restore_unit(node_id, nodes, unit_class):
    node = nodes[node_id]
    if "text" in node:
        return unit_class(id=node_id, text=node["text"], relation="elementary")
    return unit_class(
        id=node_id, relation=node["relation"], nuclearity=node["nuclearity"],
        left=restore_unit(node["left"], nodes, unit_class),
        right=restore_unit(node["right"], nodes, unit_class),
    )


def export_tree(path, tree, scene, mapping, aliases, unit_class):
    nodes, relations = {}, {}
    for node_id, node in tree["nodes"].items():
        if "relation" in node:
            label = projected_name(node["relation"], mapping)
            kind = "multinuc" if node["nuclearity"] == "NN" else "rst"
            if kind == "multinuc":
                label = aliases.get(label, label)
            node = {**node, "relation": label}
            relations[label] = kind
        nodes[int(node_id)] = node

    root = restore_unit(tree["root"], nodes, unit_class)
    root.to_rs3(str(path))
    # Declare every used label with its actual type. The serializer adds a
    # default multinuclear antithesis even in inventories with rst antithesis.
    xml = ET.parse(path)
    header = xml.find("./header/relations")
    header.clear()
    for label, kind in sorted(relations.items()):
        ET.SubElement(header, "rel", name=label, type=kind)
    ET.indent(xml, space="\t")
    xml.write(path, encoding="utf-8")
    verify_rs3(path, tree["root"], nodes, scene["edus"])


def export_corpus(corpus, trees_dir, audit_path, output_dir):
    from isanlp.annotation_rst import DiscourseUnit

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scenes_path = trees_dir / "scenes.jsonl"
    if (audit["research_revision"] != RESEARCH_REVISION
            or digest(scenes_path) != audit["segmentation"]["scenes_sha256"]):
        raise ValueError("Scene file or relation map revision differs from the parser audit")
    scenes = read_jsonl(scenes_path)
    scene_ids = [(row["key"], row["document_id"]) for row in scenes]
    if (len({key for key, _ in scene_ids}) != audit["scenes"]
            or len(scenes) != audit["scenes"] or any(row["status"] != "ok" for row in scenes)):
        raise ValueError("Expected all verified scenes with unique keys")

    destination = output_dir / corpus
    if destination.exists():
        raise FileExistsError(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = canonical_maps()
    with TemporaryDirectory(dir=output_dir) as work_dir:
        staged = Path(work_dir) / corpus
        used_aliases = {}
        for key, standard in STANDARDS.items():
            trees_path = trees_dir / f"{key}.jsonl"
            if digest(trees_path) != audit["inventory_reports"][key]["trees_sha256"]:
                raise ValueError(f"Tree file differs from the parser audit for {standard}")
            trees = read_jsonl(trees_path)
            if ([(row["key"], row["document_id"]) for row in trees] != scene_ids
                    or any(row["status"] != "ok" for row in trees)):
                raise ValueError(f"Expected complete scene alignment for {standard}")

            used_aliases[standard] = {}
            for granularity, mapping in (("fine", None), ("coarse", maps[key])):
                folder = staged / standard / granularity
                folder.mkdir(parents=True)
                aliases = multinuclear_aliases(trees, mapping)
                used_aliases[standard][granularity] = {alias: label for label, alias in aliases.items()}
                for tree, scene in zip(trees, scenes, strict=True):
                    export_tree(folder / f"{scene['key']}.rs3", tree, scene, mapping, aliases, DiscourseUnit)

        write_json(staged / "rs3-relation-aliases.json", used_aliases)
        write_json(staged / "fine-to-coarse.json", maps)
        with (staged / "scene-index.tsv").open("x", encoding="utf-8", newline="") as stream:
            index = csv.writer(stream, delimiter="\t", lineterminator="\n")
            index.writerow(("scene_id", "document_id", "edu_count"))
            index.writerows((row["key"], row["document_id"], len(row["edus"])) for row in scenes)
        (staged / "README.md").write_text(
            f"# {corpus} UniRST scene trees\n\n"
            f"{audit['documents']:,} documents, {len(scenes):,} scenes per RST standard, and "
            f"{len(scenes) * len(STANDARDS) * 2:,} RS3 files across fine and coarse. "
            f"[Corpus source]({CORPORA[corpus]})\n\n"
            "Each RS3 file covers one scene. The scene index links filenames to reviews or sentences. "
            "Fine and coarse versions share the same EDU text, links, and nuclearity.\n\n"
            "For compatibility with RS3 readers that assign one type per relation name, "
            "a label used with both types takes a `-multinuc` suffix for its two-nucleus use. "
            "Use `rs3-relation-aliases.json` to recover the experimental label. "
            "The `fine-to-coarse.json` map gives the original label projection.\n",
            encoding="utf-8",
        )
        staged.rename(destination)
    return len(scenes) * len(STANDARDS) * 2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--corpus", choices=CORPORA, required=True)
    cli.add_argument("--trees-dir", type=Path, required=True)
    cli.add_argument("--audit", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    count = export_corpus(args.corpus, args.trees_dir, args.audit, args.output_dir)
    print(f"Wrote and verified {count:,} RS3 files in {args.output_dir / args.corpus}")


if __name__ == "__main__":
    main()
