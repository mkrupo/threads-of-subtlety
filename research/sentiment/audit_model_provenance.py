"""Statically compare recorded RST-DT projection and pinned training sources.

No parser packages are imported and no model is run. Pass copies of the two
upstream source files at UPSTREAM_REF; their hashes are recorded in the output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


UPSTREAM_REF = "c51480a10a6636ae621fcc4676b12c0099f7a0f9"
RESEARCH_REF = "f06309b62cbc53dce0c98668cac225666dfb64d7"
MAPPING_PATH = "docs/eng.rst.rstdt.new_to_coarse.json"
FINE_TUNING_PATH = "isanlp_rst/universal_parser/relation_finetuning.py"
TABLES_PATH = "isanlp_rst/universal_parser/src/parser/data.py"


def assignments(source: str) -> dict:
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
    return result


def initializer_mapping(source: str) -> dict[str, str]:
    assignment = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_RSTDT_FINE_TO_COARSE"
                for t in node.targets)
    )
    if not isinstance(assignment.value, ast.Dict):
        raise ValueError("Unexpected upstream mapping schema")
    mapping = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        if key is not None:
            mapping[ast.literal_eval(key)] = ast.literal_eval(value)
            continue
        if not isinstance(value, ast.DictComp) or len(value.generators) != 1:
            raise ValueError("Unexpected upstream dictionary expansion")
        generator = value.generators[0]
        if (not isinstance(generator.target, ast.Name)
                or not isinstance(value.key, ast.Name)
                or generator.target.id != value.key.id or generator.ifs):
            raise ValueError("Unexpected upstream dictionary comprehension")
        for name in ast.literal_eval(generator.iter):
            mapping[name] = ast.literal_eval(value.value)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-tuning-source", type=Path, required=True)
    parser.add_argument("--relation-tables-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    serialized = subprocess.check_output(
        ["git", "show", f"{RESEARCH_REF}:{MAPPING_PATH}"], cwd=repo
    )
    canonical = json.loads(serialized)
    fine_bytes = args.fine_tuning_source.read_bytes()
    table_bytes = args.relation_tables_source.read_bytes()
    mapping = initializer_mapping(fine_bytes.decode("utf-8"))
    tables = assignments(table_bytes.decode("utf-8"))
    native = set(tables["_RSTDT_FINE_MULTINUC"]) | set(tables["_RSTDT_FINE_RST"])
    # These two aliases normalize names only, without changing group membership.
    aliases = {"cause": "causal", "contrast": "adversative"}
    differences, unmapped = [], []
    for fine, coarse in sorted(canonical.items()):
        if fine == "__OTHER__":
            continue
        base = fine.removesuffix("-mn")
        if base.startswith("elaboration-") or base in {"example", "definition"}:
            parent = "elaboration"
        elif base in mapping:
            parent = mapping[base]
        else:
            unmapped.append(fine)
            continue
        if coarse != aliases.get(parent, parent):
            differences.append({"relation": fine, "canonical_projection": coarse,
                                "upstream_initialization_parent": parent})
    report = {
        "kind": "static_source_audit_not_model_validation",
        "research_commit": RESEARCH_REF,
        "upstream_commit": UPSTREAM_REF,
        "sources": [
            {"path": MAPPING_PATH, "sha256": hashlib.sha256(serialized).hexdigest()},
            {"path": FINE_TUNING_PATH, "sha256": hashlib.sha256(fine_bytes).hexdigest()},
            {"path": TABLES_PATH, "sha256": hashlib.sha256(table_bytes).hexdigest()},
        ],
        "gum_fine": {
            "relation_nuclearity_rows": len(tables["RelationTableGUMFine"]),
            "native_relation_names": len({
                label.rsplit("_", 1)[0] for label in tables["RelationTableGUMFine"]
            }),
        },
        "rstdt_fine": {
            "relation_nuclearity_rows": len(tables["_RSTDT_FINE_MULTINUC"]) + sum(
                1 if label in tables["_RSTDT_FINE_NS_ONLY"] else 2
                for label in tables["_RSTDT_FINE_RST"]
            ),
            "native_relation_names": len(native),
            "canonical_projection_native_names": len(canonical) - 1,
            "native_names_missing_from_canonical_projection": sorted(native - set(canonical)),
            "canonical_names_missing_from_native_table": sorted(set(canonical) - native - {"__OTHER__"}),
            "canonical_names_without_initialization_parent": unmapped,
            "comparison_name_aliases": aliases,
            "different_group_assignments": differences,
        },
        "interpretation": (
            "Initialization and downstream coarsening serve different purposes. "
            "Differences need explanation or a declared sensitivity, not automatic correction. "
            "Training code does not identify the checkpoint used for historical .new inputs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gum": report["gum_fine"], "rstdt": report["rstdt_fine"]}, indent=2))


if __name__ == "__main__":
    main()
