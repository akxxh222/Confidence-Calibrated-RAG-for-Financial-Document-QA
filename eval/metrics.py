"""
Metrics for the core comparison table (Section 12): per-category
hallucination rate and coverage rate, each with a bootstrap 95% CI.

Definitions used here:
  coverage         = fraction of questions where the system answered
                      (mode != 'refuse')
  hallucination_rate = among answered/hedged responses, fraction judged
                      incorrect against ground truth (was_hallucination)
"""
import numpy as np


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap."""
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)


def compute_category_metrics(results: list[dict]) -> dict:
    """
    results: list of {"mode": ..., "was_hallucination": bool or None}
    was_hallucination is None for refused questions (no answer was given).
    """
    n = len(results)
    if n == 0:
        return {"coverage": (0, 0, 0), "hallucination_rate": (0, 0, 0), "n": 0}

    covered = [1.0 if r["mode"] != "refuse" else 0.0 for r in results]
    coverage_stats = bootstrap_ci(covered)

    answered = [r for r in results if r["mode"] != "refuse"]
    halluc_flags = [1.0 if r["was_hallucination"] else 0.0 for r in answered]
    halluc_stats = bootstrap_ci(halluc_flags) if halluc_flags else (0.0, 0.0, 0.0)

    return {
        "n": n,
        "coverage": coverage_stats,
        "hallucination_rate": halluc_stats,
    }


def format_metrics_table(all_results: dict[str, dict[str, list[dict]]]) -> str:
    """
    all_results: {variant_name: {category: [result_dict, ...]}}
    Produces a markdown table matching Section 12's layout.
    """
    lines = ["| Category | Variant | Coverage (95% CI) | Hallucination Rate (95% CI) | n |",
             "|---|---|---|---|---|"]
    for variant, by_category in all_results.items():
        for category, results in by_category.items():
            m = compute_category_metrics(results)
            cov_mean, cov_lo, cov_hi = m["coverage"]
            hal_mean, hal_lo, hal_hi = m["hallucination_rate"]
            lines.append(
                f"| {category} | {variant} | "
                f"{cov_mean:.2f} [{cov_lo:.2f}, {cov_hi:.2f}] | "
                f"{hal_mean:.2f} [{hal_lo:.2f}, {hal_hi:.2f}] | {m['n']} |"
            )
    return "\n".join(lines)
