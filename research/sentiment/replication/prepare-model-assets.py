"""Stage the pinned public model files and verify all sentiment checkpoints."""

import argparse
from functools import cache
from hashlib import sha256
import json
from pathlib import Path

from huggingface_hub import snapshot_download

RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_result(name):
    return json.loads((RESULTS / name).read_text())


@cache
def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path, expected):
    if file_sha256(path.resolve(strict=True)) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}")


def verify_parser(directory, inventory, reference):
    for filename, key in (
        ("best_weights.pt", "weights_sha256"),
        ("config.json", "config_sha256"),
        (f"relation_table_{inventory}.txt", "table_sha256"),
    ):
        verify_sha256(directory / filename, reference[key])


def link(path, source):
    if path.exists() or path.is_symlink():
        if path.resolve(strict=True) != source.resolve(strict=True):
            raise FileExistsError(f"{path} points to a different asset")
        return
    path.symlink_to(source, target_is_directory=source.is_dir())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", required=True, type=Path)
    cli.add_argument("--checkpoints", required=True, type=Path, help="Parent of erst/ and rstdt/")
    cli.add_argument("--offline", action="store_true")
    args = cli.parse_args()
    assets = args.assets.resolve()
    cache = assets / "hf" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    audit = read_result("sfu-full-parser-audit.json")["inventory_reports"]
    revisions = read_result("unirst-parser-pilot.json")["source_revisions"]
    scorer = read_result("sfu-edu-scores-v1.json")["model"]

    for subdir, inventory in (("erst", "eng.erst.gum"), ("rstdt", "eng.rst.rstdt")):
        verify_parser(args.checkpoints / subdir, inventory, audit[inventory])

    def snapshot(repo, revision, files):
        return Path(snapshot_download(
            repo, revision=revision, cache_dir=cache, allow_patterns=files,
            local_files_only=args.offline,
        ))

    parser_files = ["best_weights.pt", "config.json", "relation_table_*.txt"]
    stock = snapshot("tchewik/isanlp_rst_v3", revisions["unirst_stock"], parser_files)
    fine_base = snapshot("tchewik/isanlp_rst_v3", revisions["unirst_fine_base"], parser_files)
    for directory in (stock, fine_base):
        for inventory in ("deu.rst.pcc", "nld.rst.nldt"):
            verify_parser(directory, inventory, audit[inventory])
    link(assets / "stock", stock)

    tokenizer_files = ["config.json", "tokenizer.json", "tokenizer_config.json",
                       "sentencepiece.bpe.model", "special_tokens_map.json"]
    for size in ("base", "large"):
        revision = revisions[f"xlm_roberta_{size}"]
        alias = f"xlm-roberta-{size}"
        snapshot(alias, revision, tokenizer_files)
        main_ref = cache / f"models--{alias}" / "refs" / "main"
        main_ref.parent.mkdir(parents=True, exist_ok=True)
        if main_ref.exists() and main_ref.read_text().strip() != revision:
            raise ValueError(f"{main_ref} already selects a different snapshot")
        if not main_ref.exists():
            main_ref.write_text(revision, encoding="ascii")

    scorer_files = {"config.json": "config.json", "tokenizer.json": "tokenizer.json",
                    "model.onnx": "onnx/model_quantized.onnx"}
    model = snapshot(scorer["id"], scorer["revision"], list(scorer_files.values()))
    (assets / "cardiff").mkdir(exist_ok=True)
    for filename, relative_path in scorer_files.items():
        source = model / relative_path
        verify_sha256(source, scorer["sha256"][filename])
        link(assets / "cardiff" / filename, source)

    print(f"Verified and staged parser and scorer assets in {assets}")


if __name__ == "__main__":
    main()
