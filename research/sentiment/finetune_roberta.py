"""Fine-tune roberta-base on full review texts as a neural baseline for a prepared cohort.

The baseline uses the cohort's frozen five outer folds and fixed settings. There
is no hyperparameter search, early stopping or checkpoint selection: each fold
trains for a fixed number of epochs on its outer training reviews and then
predicts its test reviews. One process trains one fold, so folds can share GPUs.

Run in a separate environment pinned in finetune-requirements.txt (PyTorch,
Transformers), not the locked CPU project environment.

``train``  Train one outer fold and write its held-out probabilities.
``merge``  Combine the five fold outputs, check coverage and write a report.
"""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter

SETTINGS = {
    "max_length": 256, "batch_size": 16, "epochs": 3, "learning_rate": 2e-5, "weight_decay": 0.01,
    "warmup_fraction": 0.06, "max_grad_norm": 1.0, "precision": "fp32", "seed": 42,
    "selection": "none: the model after the last epoch predicts the test fold",
}
CONDITION = "roberta-base"


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_cohort(prepared):
    """Reviews, manifest and folds, checked against the cohort's provenance hashes."""
    provenance = json.loads((prepared / "provenance.json").read_text())
    for name in ("reviews.jsonl", "manifest.jsonl", "splits.json"):
        if digest(prepared / name) != provenance["outputs_sha256"][name]:
            raise ValueError(f"{name} differs from the cohort provenance")
    reviews = {row["document_id"]: row for row in read_jsonl(prepared / "reviews.jsonl")}
    splits = json.loads((prepared / "splits.json").read_text())
    return reviews, splits, dict(provenance["outputs_sha256"])


def model_identity(model_dir):
    """The Hub snapshot revision and file hashes of the local model directory."""
    resolved = Path(model_dir).resolve()
    files = sorted(p for p in resolved.iterdir() if p.is_file())
    return {"path_revision": resolved.name, "files_sha256": {p.name: digest(p) for p in files}}


def train(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = SETTINGS["seed"] + args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    reviews, splits, input_hashes = load_cohort(args.prepared)
    fold = next(f for f in splits["outer_folds"] if f["fold"] == args.fold)
    train_ids, test_ids = sorted(fold["train_ids"]), sorted(fold["test_ids"])
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test reviews overlap")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2).to(device)

    def encode(ids):
        texts = [reviews[doc]["text"] for doc in ids]
        full = tokenizer(texts, truncation=False)["input_ids"]
        encoded = tokenizer(texts, truncation=True, max_length=SETTINGS["max_length"])
        return encoded, sum(len(tokens) > SETTINGS["max_length"] for tokens in full), max(map(len, full))

    train_enc, train_truncated, train_longest = encode(train_ids)
    test_enc, test_truncated, test_longest = encode(test_ids)
    labels = torch.tensor([reviews[doc]["label"] for doc in train_ids])

    def batches(encoded, order, size):
        for start in range(0, len(order), size):
            chunk = order[start:start + size]
            padded = tokenizer.pad({"input_ids": [encoded["input_ids"][i] for i in chunk],
                                    "attention_mask": [encoded["attention_mask"][i] for i in chunk]},
                                   return_tensors="pt")
            yield chunk, {key: value.to(device) for key, value in padded.items()}

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": SETTINGS["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=SETTINGS["learning_rate"])
    steps_per_epoch = math.ceil(len(train_ids) / SETTINGS["batch_size"])
    total = steps_per_epoch * SETTINGS["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(SETTINGS["warmup_fraction"] * total), total)
    generator = torch.Generator().manual_seed(seed)
    start = perf_counter()
    epoch_losses = []
    model.train()
    for epoch in range(SETTINGS["epochs"]):
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        losses = []
        for chunk, batch in batches(train_enc, order, SETTINGS["batch_size"]):
            output = model(**batch, labels=labels[chunk].to(device))
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(output.loss))
        epoch_losses.append(sum(losses) / len(losses))
        print(f"fold {args.fold} epoch {epoch + 1}: mean training loss {epoch_losses[-1]:.4f} "
              f"after {perf_counter() - start:.0f}s", flush=True)

    model.eval()
    probabilities = [0.0] * len(test_ids)
    with torch.no_grad():
        for chunk, batch in batches(test_enc, list(range(len(test_ids))), 64):
            positive = torch.softmax(model(**batch).logits.float(), dim=-1)[:, 1].cpu().tolist()
            for index, value in zip(chunk, positive):
                probabilities[index] = value
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"fold-{args.fold}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for doc, value in zip(test_ids, probabilities):
            stream.write(json.dumps({"document_id": doc, "outer_fold": args.fold, "label": reviews[doc]["label"],
                                     "condition": CONDITION, "probability": value}) + "\n")
    write_json(args.output / f"fold-{args.fold}.json", {
        "fold": args.fold, "settings": SETTINGS, "seed": seed, "train_reviews": len(train_ids),
        "test_reviews": len(test_ids), "steps": total, "epoch_mean_training_loss": epoch_losses,
        "truncated_reviews": {"train": train_truncated, "test": test_truncated},
        "longest_tokens": {"train": train_longest, "test": test_longest},
        "seconds": perf_counter() - start, "input_hashes": input_hashes, "model": model_identity(args.model_dir),
        "predictions_sha256": digest(path), "source_sha256": digest(Path(__file__)),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "deterministic_algorithms": "torch.use_deterministic_algorithms(True, warn_only=True)"},
    })


def merge(args):
    reviews, splits, input_hashes = load_cohort(args.prepared)
    rows, reports = [], []
    for fold in splits["outer_folds"]:
        report = json.loads((args.output / f"fold-{fold['fold']}.json").read_text())
        path = args.output / f"fold-{fold['fold']}.jsonl"
        if digest(path) != report["predictions_sha256"] or report["input_hashes"] != input_hashes:
            raise ValueError(f"Fold {fold['fold']} outputs or inputs differ from its report")
        fold_rows = read_jsonl(path)
        if sorted(row["document_id"] for row in fold_rows) != sorted(fold["test_ids"]):
            raise ValueError(f"Fold {fold['fold']} does not cover its test reviews exactly")
        rows.extend(fold_rows)
        reports.append(report)
    if len({row["document_id"] for row in rows}) != len(reviews):
        raise ValueError("Expected one held-out prediction per review")
    if len({json.dumps(r["model"], sort_keys=True) for r in reports}) != 1 or any(
            r["source_sha256"] != reports[0]["source_sha256"] for r in reports):
        raise ValueError("Folds used different model files or training code")
    rows.sort(key=lambda row: row["document_id"])
    path = args.output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    eps = 1e-12
    loss = [-(r["label"] * math.log(min(max(r["probability"], eps), 1 - eps))
              + (1 - r["label"]) * math.log(1 - min(max(r["probability"], eps), 1 - eps))) for r in rows]
    write_json(args.output / "report.json", {
        "protocol": "roberta-base-baseline-v1", "condition": CONDITION, "settings": SETTINGS,
        "reviews": len(rows), "log_loss": sum(loss) / len(loss),
        "accuracy": sum((r["probability"] >= 0.5) == bool(r["label"]) for r in rows) / len(rows),
        "folds": reports, "input_hashes": input_hashes, "predictions_sha256": digest(path),
    })
    print(f"Merged {len(rows)} held-out predictions", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--fold", type=int, required=True)
    fit.add_argument("--model-dir", type=Path, required=True, help="Local roberta-base snapshot")
    fit.add_argument("--device", default="cuda:0")
    join = sub.add_parser("merge")
    for command in (fit, join):
        command.add_argument("--prepared", type=Path, required=True, help="Cohort prepared/ folder")
        command.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "train":
        if (args.output / f"fold-{args.fold}.jsonl").exists():
            raise FileExistsError(f"Fold {args.fold} already has outputs in {args.output}")
        train(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
