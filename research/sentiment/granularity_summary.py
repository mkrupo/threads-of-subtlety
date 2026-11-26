"""Print Markdown tables from saved granularity diagnostic reports.

Usage: python -m research.sentiment.granularity_summary REPORT.json [...]
Recognizes the loss, scramble, split-ablation and confirmation report protocols.
"""

import json
from pathlib import Path
import sys


NAMES = {"deu.rst.pcc": "PCC", "nld.rst.nldt": "NLDT", "eng.erst.gum": "GUM/eRST", "eng.rst.rstdt": "RST-DT"}


def milli(value):
    return f"{1000 * value:+.2f}"


def interval(values):
    return f"[{milli(values[0])}, {milli(values[1])}]"


def loss(report):
    print(f"\n### {report['corpus']}: paired log-loss (×10⁻³; negative favours the left model)\n")
    for family, rows in report["contrasts"].items():
        print(f"\n{family}\n\n| Contrast | Δ log-loss | 95% interval | p | Holm p | Δ macro-F1 (points) |")
        print("|---|---:|---:|---:|---:|---:|")
        for key, row in rows.items():
            print(f"| {key} | {milli(row['log_loss_difference'])} | {interval(row['interval_95'])} | "
                  f"{row['bootstrap_p']:.3f} | {row['holm_p']:.3f} | {100 * row['macro_f1_difference']:+.2f} |")


def scramble(report):
    print(f"\n### {report['corpus']}: scrambled-label control ({report['permutations']} permutations)\n")
    print("| Standard | Fine LL | Coarse LL | Unlabeled LL | Scrambled LL mean [2.5%, 97.5%] | p (fine < scrambled) |"
          " Gate spread real | Scrambled spread mean [2.5%, 97.5%] | p (spread) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for inventory, row in report["inventories"].items():
        real, null, p = row["real"], row["scrambled"], row["permutation_p"]
        q, s = null["log_loss"]["quantiles_2.5_50_97.5"], null["gate_spread"]["quantiles_2.5_50_97.5"]
        print(f"| {NAMES[inventory]} | {real['fine']['log_loss']:.4f} | {real['coarse']['log_loss']:.4f} | "
              f"{real['unlabeled']['log_loss']:.4f} | {null['log_loss']['mean']:.4f} [{q[0]:.4f}, {q[2]:.4f}] | "
              f"{p['fine_log_loss_lower_than_scrambled']:.3f} | {row['real_fine_gate_spread']:.3f} | "
              f"{null['gate_spread']['mean']:.3f} [{s[0]:.3f}, {s[2]:.3f}] | {p['gate_spread_higher_than_scrambled']:.3f} |")
    for inventory, row in report["inventories"].items():
        groups = row["fine_gates_within_coarse_groups"]
        print(f"\n{NAMES[inventory]} fine gates inside coarse groups (mean over folds, min–max):")
        for group, labels in groups.items():
            members = ", ".join(f"{label} {v['mean_gate']:.2f} ({v['min_gate']:.2f}–{v['max_gate']:.2f})"
                                for label, v in sorted(labels.items(), key=lambda item: -item[1]["mean_gate"]))
            print(f"- {group}: {members}")


def split(report):
    for subset, block in report["subsets"].items():
        print(f"\n### Split minus intact, {subset} ({block['reviews']} reviews; log-loss ×10⁻³, positive = split hurts)\n")
        print("| Model | Intact LL | Split LL | Δ | 95% interval | p | Intact F1 | Split F1 | Changed |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, row in block["models"].items():
            print(f"| {name} | {row['intact']['log_loss']:.4f} | {row['split']['log_loss']:.4f} | "
                  f"{milli(row['split_minus_intact_log_loss'])} | {interval(row['interval_95'])} | "
                  f"{row['bootstrap_p']:.3f} | {row['intact']['macro_f1']:.4f} | {row['split']['macro_f1']:.4f} | "
                  f"{row['changed_predictions']} |")


def contrasts(rows, verdicts=False):
    head = "| Contrast | Δ log-loss | 95% interval | p | Holm p | Δ macro-F1 (points) |" + (" Verdict |" if verdicts else "")
    print(head)
    print("|---|---:|---:|---:|---:|---:|" + ("---|" if verdicts else ""))
    for key, row in rows.items():
        print(f"| {key} | {milli(row['log_loss_difference'])} | {interval(row['interval_95'])} | "
              f"{row['bootstrap_p']:.3f} | {row['holm_p']:.3f} | {100 * row['macro_f1_difference']:+.2f} |"
              + (f" {row['verdict']} |" if verdicts else ""))


def conditions(rows):
    print("| Condition | Log-loss | Macro-F1 | Accuracy |\n|---|---:|---:|---:|")
    for name, row in rows.items():
        print(f"| {name} | {row['log_loss']:.4f} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} |")


def primary(report):
    print(f"\n### Confirmation cohort ({report['reviews']} reviews), log-loss ×10⁻³\n")
    conditions(report["conditions"])
    print("\nPrimary: fine minus mapped coarse (Holm over four standards)\n")
    contrasts(report["primary_fine_minus_coarse"], verdicts=True)
    print("\nSecondary 1: labels minus unlabeled links\n")
    contrasts(report["secondary_labels_minus_unlabeled"])


def curve(report):
    print("\n### Learning curve: fine minus coarse log-loss (×10⁻³) by training size\n")
    sizes = [str(size) for size in report["sizes"]]
    print("| Standard | " + " | ".join(sizes) + " | Gap(1,000) − gap(full) | 95% interval | Holm p |")
    print("|---|" + "---:|" * (len(sizes) + 3))
    for inventory, row in report["inventories"].items():
        change = row["gap_at_smallest_minus_gap_at_full"]
        cells = " | ".join(milli(row["by_training_size"][size]["fine_minus_coarse"]) for size in sizes)
        print(f"| {NAMES[inventory]} | {cells} | {milli(change['difference'])} | {interval(change['interval_95'])} | "
              f"{change['holm_p']:.3f} |")
    print("\nUnlabeled / coarse / fine log-loss by size:")
    for inventory, row in report["inventories"].items():
        print(f"- {NAMES[inventory]}: " + "; ".join(
            f"{size}: {row['by_training_size'][size]['unlabeled_log_loss']:.4f} / "
            f"{row['by_training_size'][size]['coarse_log_loss']:.4f} / {row['by_training_size'][size]['fine_log_loss']:.4f}"
            for size in sizes))


def others(report):
    print("\n### Other models, log-loss ×10⁻³\n")
    conditions(report["conditions"])
    print()
    contrasts(report["fine_minus_coarse"])


def main():
    for path in sys.argv[1:]:
        report = json.loads(Path(path).read_text())
        protocol = report["protocol"]
        print(f"\n## {path}\n")
        if protocol == "granularity-loss-v1":
            loss(report)
        elif protocol == "granularity-scramble-v1":
            scramble(report)
        elif protocol == "amazon-split-ablation-v1":
            split(report)
        elif protocol == "granularity-confirmation-v1/primary":
            primary(report)
        elif protocol == "granularity-confirmation-v1/curve":
            curve(report)
        elif protocol == "granularity-confirmation-v1/others":
            others(report)
        else:
            print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
