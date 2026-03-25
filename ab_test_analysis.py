"""
A/B Testing Analysis — Online Course Multi-Channel Advertising Optimization

Three experiments evaluating advertising strategy for an online learning platform:
1. Ad Creative Test: content-focused vs social-proof creative across channels
2. Landing Page Test: generic vs channel-specific landing pages
3. Budget Allocation Test: Bayesian evaluation of rebalanced channel spend

Synthetic data generated to reflect realistic online education advertising patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS = {
    "control": "#5C7AEA",
    "treatment": "#FF6B6B",
    "neutral": "#A0A0A0",
    "accent": "#2ECC71",
}


# ============================================================
# Statistical Testing Utilities
# ============================================================

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Required sample size per group for a two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    pooled = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    return int(np.ceil(n))


def chi_square_test(c_conv, c_total, t_conv, t_total, alpha=0.05):
    """Chi-square test for two conversion rates."""
    table = np.array([
        [c_conv, c_total - c_conv],
        [t_conv, t_total - t_conv]
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    return {
        "control_rate": c_conv / c_total,
        "treatment_rate": t_conv / t_total,
        "lift": (t_conv / t_total - c_conv / c_total) / (c_conv / c_total),
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def welch_t_test(control_vals, treatment_vals, alpha=0.05):
    """Welch's t-test for continuous metrics."""
    c_mean, t_mean = np.mean(control_vals), np.mean(treatment_vals)
    t_stat, p_value = stats.ttest_ind(control_vals, treatment_vals, equal_var=False)
    pooled_std = np.sqrt((np.std(control_vals, ddof=1)**2 +
                          np.std(treatment_vals, ddof=1)**2) / 2)
    cohens_d = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
    return {
        "control_mean": c_mean,
        "treatment_mean": t_mean,
        "lift": (t_mean - c_mean) / c_mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < alpha,
    }


def bootstrap_ci(control_vals, treatment_vals, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for difference in means."""
    rng = np.random.default_rng(42)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        c_sample = rng.choice(control_vals, size=len(control_vals), replace=True)
        t_sample = rng.choice(treatment_vals, size=len(treatment_vals), replace=True)
        diffs[i] = np.mean(t_sample) - np.mean(c_sample)
    lower = np.percentile(diffs, (1 - ci) / 2 * 100)
    upper = np.percentile(diffs, (1 + ci) / 2 * 100)
    return {"mean_diff": np.mean(diffs), "ci_lower": lower, "ci_upper": upper}


def bayesian_conversion(c_conv, c_total, t_conv, t_total, n_samples=100000):
    """Bayesian estimation: probability that treatment > control."""
    rng = np.random.default_rng(42)
    c_samples = rng.beta(1 + c_conv, 1 + c_total - c_conv, n_samples)
    t_samples = rng.beta(1 + t_conv, 1 + t_total - t_conv, n_samples)
    prob_t_better = np.mean(t_samples > c_samples)
    return {
        "prob_treatment_better": prob_t_better,
        "control_posterior_mean": np.mean(c_samples),
        "treatment_posterior_mean": np.mean(t_samples),
        "expected_lift": np.mean((t_samples - c_samples) / c_samples),
    }


# ============================================================
# Experiment 1: Ad Creative A/B Test
# ============================================================

def run_experiment_1():
    """
    Ad Creative Test across 4 channels.
    Control (A): Course-content focused creative
    Treatment (B): Learner-outcome + social proof creative
    """
    print("=" * 60)
    print("EXPERIMENT 1: Ad Creative A/B Test")
    print("=" * 60)

    np.random.seed(101)
    channels = ["Search", "Feed", "KOL", "In-App"]
    baselines = {
        "Search":  {"impressions": 50000, "ctr_a": 0.042, "ctr_b": 0.045,
                     "cvr_a": 0.068, "cvr_b": 0.079},
        "Feed":    {"impressions": 80000, "ctr_a": 0.018, "ctr_b": 0.023,
                     "cvr_a": 0.035, "cvr_b": 0.041},
        "KOL":     {"impressions": 30000, "ctr_a": 0.031, "ctr_b": 0.038,
                     "cvr_a": 0.052, "cvr_b": 0.058},
        "In-App":  {"impressions": 60000, "ctr_a": 0.025, "ctr_b": 0.028,
                     "cvr_a": 0.082, "cvr_b": 0.094},
    }

    results = []
    for ch in channels:
        b = baselines[ch]
        imp = b["impressions"]
        clicks_a = np.random.binomial(imp // 2, b["ctr_a"])
        clicks_b = np.random.binomial(imp // 2, b["ctr_b"])
        conv_a = np.random.binomial(clicks_a, b["cvr_a"])
        conv_b = np.random.binomial(clicks_b, b["cvr_b"])

        ctr_result = chi_square_test(clicks_a, imp // 2, clicks_b, imp // 2)
        cvr_result = chi_square_test(conv_a, clicks_a, conv_b, clicks_b)

        cpc_a = np.random.normal(2.8, 0.5, clicks_a).clip(0.5)
        cpc_b = np.random.normal(2.6, 0.5, clicks_b).clip(0.5)
        total_cost_a = cpc_a.sum()
        total_cost_b = cpc_b.sum()
        cpa_a = total_cost_a / conv_a if conv_a > 0 else 0
        cpa_b = total_cost_b / conv_b if conv_b > 0 else 0

        results.append({
            "channel": ch, "impressions": imp,
            "ctr_a": ctr_result["control_rate"],
            "ctr_b": ctr_result["treatment_rate"],
            "ctr_lift": ctr_result["lift"],
            "ctr_p": ctr_result["p_value"],
            "ctr_sig": ctr_result["significant"],
            "cvr_a": cvr_result["control_rate"],
            "cvr_b": cvr_result["treatment_rate"],
            "cvr_lift": cvr_result["lift"],
            "cvr_p": cvr_result["p_value"],
            "cvr_sig": cvr_result["significant"],
            "cpa_a": cpa_a, "cpa_b": cpa_b,
            "cpa_lift": (cpa_b - cpa_a) / cpa_a if cpa_a > 0 else 0,
            "conv_a": conv_a, "conv_b": conv_b,
        })

    df = pd.DataFrame(results)
    print(df[["channel", "ctr_a", "ctr_b", "ctr_lift", "ctr_sig",
              "cvr_a", "cvr_b", "cvr_lift", "cvr_sig",
              "cpa_a", "cpa_b", "cpa_lift"]].to_string(index=False, float_format="%.4f"))

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(channels))
    w = 0.32

    # Panel 1: CVR comparison
    ax = axes[0]
    ax.bar(x - w/2, [r["cvr_a"] * 100 for r in results], w,
           label="Control (Content)", color=COLORS["control"], edgecolor="white")
    ax.bar(x + w/2, [r["cvr_b"] * 100 for r in results], w,
           label="Treatment (Social Proof)", color=COLORS["treatment"], edgecolor="white")
    for i, r in enumerate(results):
        sig = "*" if r["cvr_sig"] else ""
        ax.text(i + w/2, r["cvr_b"] * 100 + 0.2,
                f'+{r["cvr_lift"]:.0%}{sig}', ha="center", fontsize=9,
                fontweight="bold", color=COLORS["treatment"])
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Signup Conversion Rate by Channel")
    ax.legend(loc="upper left", fontsize=9)

    # Panel 2: CPA comparison
    ax = axes[1]
    ax.bar(x - w/2, [r["cpa_a"] for r in results], w,
           label="Control", color=COLORS["control"], edgecolor="white")
    ax.bar(x + w/2, [r["cpa_b"] for r in results], w,
           label="Treatment", color=COLORS["treatment"], edgecolor="white")
    for i, r in enumerate(results):
        pct = r["cpa_lift"]
        ax.text(i + w/2, r["cpa_b"] + 0.5, f'{pct:+.0%}', ha="center",
                fontsize=9, fontweight="bold",
                color=COLORS["accent"] if pct < 0 else COLORS["neutral"])
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("CPA ($)")
    ax.set_title("Cost Per Acquisition by Channel")
    ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Experiment 1: Ad Creative A/B Test — Social Proof vs Content-Focused",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp1_ad_creative.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved {OUTPUT_DIR}/exp1_ad_creative.png")
    return df


# ============================================================
# Experiment 2: Landing Page A/B Test
# ============================================================

def run_experiment_2():
    """
    Landing Page Test.
    Control: Generic landing page (same for all channels)
    Treatment: Channel-specific landing page
      - Search: course syllabus & instructor credentials
      - Feed: limited-time offer & enrollment count
      - KOL: mirrors KOL talking points & discount code
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Landing Page Optimization")
    print("=" * 60)

    np.random.seed(202)
    channels = ["Search", "Feed", "KOL"]
    configs = {
        "Search": {"n": 6000, "cvr_generic": 0.058, "cvr_custom": 0.072},
        "Feed":   {"n": 9000, "cvr_generic": 0.032, "cvr_custom": 0.039},
        "KOL":    {"n": 4000, "cvr_generic": 0.045, "cvr_custom": 0.056},
    }

    all_results = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ch in enumerate(channels):
        cfg = configs[ch]
        n_per_group = cfg["n"] // 2

        conv_generic = np.random.binomial(n_per_group, cfg["cvr_generic"])
        conv_custom = np.random.binomial(n_per_group, cfg["cvr_custom"])

        result = chi_square_test(conv_generic, n_per_group, conv_custom, n_per_group)

        generic_arr = np.concatenate([np.ones(conv_generic),
                                       np.zeros(n_per_group - conv_generic)])
        custom_arr = np.concatenate([np.ones(conv_custom),
                                      np.zeros(n_per_group - conv_custom)])
        boot = bootstrap_ci(generic_arr, custom_arr)

        print(f"\n  {ch} Channel:")
        print(f"    Generic CVR:  {result['control_rate']:.3%}")
        print(f"    Custom CVR:   {result['treatment_rate']:.3%}")
        print(f"    Lift:         {result['lift']:+.1%}")
        print(f"    p-value:      {result['p_value']:.4f}  "
              f"{'[SIG]' if result['significant'] else '[NS]'}")
        print(f"    Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        all_results.append({**result, "channel": ch, "boot_ci": boot})

        # Bootstrap distribution subplot
        ax = axes[idx]
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(10000):
            g = rng.choice(generic_arr, size=len(generic_arr), replace=True)
            c = rng.choice(custom_arr, size=len(custom_arr), replace=True)
            diffs.append(np.mean(c) - np.mean(g))
        diffs = np.array(diffs)

        ax.hist(diffs * 100, bins=60, color=COLORS["control"], alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No Effect")
        ax.axvline(boot["ci_lower"] * 100, color=COLORS["accent"], linestyle=":",
                   linewidth=1.5, label="95% CI")
        ax.axvline(boot["ci_upper"] * 100, color=COLORS["accent"], linestyle=":", linewidth=1.5)
        sig_label = "Significant" if result["significant"] else "Not Significant"
        ax.set_title(f"{ch} (p={result['p_value']:.3f}, {sig_label})", fontsize=11)
        ax.set_xlabel("CVR Difference (pp)")
        if idx == 0:
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

    plt.suptitle("Experiment 2: Bootstrap CI — Channel-Specific vs Generic Landing Page",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp2_landing_page.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved {OUTPUT_DIR}/exp2_landing_page.png")
    return all_results


# ============================================================
# Experiment 3: Budget Reallocation (Bayesian)
# ============================================================

def run_experiment_3():
    """
    Budget Reallocation Test.
    Control: Equal 25/25/25/25 budget split
    Treatment: Data-driven reallocation (Search 30%, In-App 35%, Feed 20%, KOL 15%)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Budget Reallocation (Bayesian)")
    print("=" * 60)

    np.random.seed(303)
    total_budget = 40000

    control_alloc = {"Search": 0.25, "Feed": 0.25, "KOL": 0.25, "In-App": 0.25}
    treatment_alloc = {"Search": 0.30, "Feed": 0.20, "KOL": 0.15, "In-App": 0.35}

    channel_params = {
        "Search":  {"cpa_mean": 38, "cpa_std": 8},
        "Feed":    {"cpa_mean": 52, "cpa_std": 12},
        "KOL":     {"cpa_mean": 45, "cpa_std": 10},
        "In-App":  {"cpa_mean": 30, "cpa_std": 6},
    }

    def simulate_group(alloc, n_days=30):
        daily_conv = []
        daily_cost = []
        for day in range(n_days):
            day_conv = 0
            day_cost = 0
            for ch, pct in alloc.items():
                budget = total_budget * pct / n_days
                cpa = max(5, np.random.normal(channel_params[ch]["cpa_mean"],
                                               channel_params[ch]["cpa_std"]))
                conv = int(budget / cpa)
                day_conv += conv
                day_cost += budget
            daily_conv.append(day_conv)
            daily_cost.append(day_cost)
        return np.array(daily_conv), np.array(daily_cost)

    ctrl_conv, ctrl_cost = simulate_group(control_alloc)
    treat_conv, treat_cost = simulate_group(treatment_alloc)

    ctrl_total = ctrl_conv.sum()
    treat_total = treat_conv.sum()
    ctrl_cpa = ctrl_cost.sum() / ctrl_total
    treat_cpa = treat_cost.sum() / treat_total

    t_result = welch_t_test(ctrl_conv.astype(float), treat_conv.astype(float))

    # Bayesian: Gamma-Poisson model for daily conversions
    ctrl_alpha = 1 + ctrl_conv.sum()
    ctrl_beta = 1 + len(ctrl_conv)
    treat_alpha = 1 + treat_conv.sum()
    treat_beta = 1 + len(treat_conv)

    rng = np.random.default_rng(42)
    ctrl_samples = rng.gamma(ctrl_alpha, 1 / ctrl_beta, 100000)
    treat_samples = rng.gamma(treat_alpha, 1 / treat_beta, 100000)
    prob_better = np.mean(treat_samples > ctrl_samples)

    print(f"\n  Control  — Conversions: {ctrl_total}, CPA: ${ctrl_cpa:.2f}")
    print(f"  Treatment — Conversions: {treat_total}, CPA: ${treat_cpa:.2f}")
    print(f"  Conversion Lift: {(treat_total - ctrl_total) / ctrl_total:+.1%}")
    print(f"  CPA Change:      {(treat_cpa - ctrl_cpa) / ctrl_cpa:+.1%}")
    print(f"  Welch's t p-value: {t_result['p_value']:.4f}")
    print(f"  P(Treatment > Control): {prob_better:.2%}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Daily conversions
    ax = axes[0, 0]
    days = np.arange(1, 31)
    ax.plot(days, ctrl_conv, color=COLORS["control"],
            label="Control (Equal Split)", linewidth=1.5)
    ax.plot(days, treat_conv, color=COLORS["treatment"],
            label="Treatment (Optimized)", linewidth=1.5)
    ax.fill_between(days, ctrl_conv, alpha=0.15, color=COLORS["control"])
    ax.fill_between(days, treat_conv, alpha=0.15, color=COLORS["treatment"])
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Conversions")
    ax.set_title("Daily Conversions Over 30-Day Test")
    ax.legend(fontsize=9)

    # Budget allocation
    ax = axes[0, 1]
    chs = list(control_alloc.keys())
    x = np.arange(len(chs))
    w = 0.32
    ax.bar(x - w/2, [control_alloc[c] * 100 for c in chs], w,
           label="Control", color=COLORS["control"], edgecolor="white")
    ax.bar(x + w/2, [treatment_alloc[c] * 100 for c in chs], w,
           label="Treatment", color=COLORS["treatment"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(chs)
    ax.set_ylabel("Budget Share (%)")
    ax.set_title("Budget Allocation: Control vs Treatment")
    ax.legend(fontsize=9)

    # Bayesian posterior
    ax = axes[1, 0]
    ax.hist(ctrl_samples, bins=80, alpha=0.6, color=COLORS["control"],
            label=f"Control (mean={np.mean(ctrl_samples):.1f})", density=True,
            edgecolor="white")
    ax.hist(treat_samples, bins=80, alpha=0.6, color=COLORS["treatment"],
            label=f"Treatment (mean={np.mean(treat_samples):.1f})", density=True,
            edgecolor="white")
    ax.set_xlabel("Daily Conversions (posterior)")
    ax.set_ylabel("Density")
    ax.set_title(f"Bayesian Posterior — P(Treatment > Control) = {prob_better:.1%}")
    ax.legend(fontsize=9)

    # Summary table
    ax = axes[1, 1]
    ax.axis("off")
    summary = [
        ["Metric", "Control", "Treatment", "Change"],
        ["Total Conversions", f"{ctrl_total}", f"{treat_total}",
         f"{(treat_total - ctrl_total) / ctrl_total:+.1%}"],
        ["Overall CPA", f"${ctrl_cpa:.2f}", f"${treat_cpa:.2f}",
         f"{(treat_cpa - ctrl_cpa) / ctrl_cpa:+.1%}"],
        ["Daily Avg Conv", f"{ctrl_conv.mean():.1f}", f"{treat_conv.mean():.1f}",
         f"{t_result['lift']:+.1%}"],
        ["P(Treat > Ctrl)", "", "", f"{prob_better:.1%}"],
        ["Welch's t p-value", "", "", f"{t_result['p_value']:.4f}"],
    ]
    table = ax.table(cellText=summary, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for j in range(4):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Experiment Summary", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle("Experiment 3: Budget Reallocation — Bayesian Evaluation",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp3_budget_reallocation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved {OUTPUT_DIR}/exp3_budget_reallocation.png")

    return {
        "ctrl_total": ctrl_total, "treat_total": treat_total,
        "ctrl_cpa": ctrl_cpa, "treat_cpa": treat_cpa,
        "prob_better": prob_better,
    }


# ============================================================
# Sample Size Planning
# ============================================================

def print_sample_size_table():
    """Print required sample sizes for experiment planning."""
    print("=" * 60)
    print("SAMPLE SIZE PLANNING")
    print("=" * 60)
    scenarios = [
        ("Ad Creative CVR", 0.05, 0.15),
        ("Ad Creative CVR", 0.05, 0.20),
        ("Landing Page CVR", 0.04, 0.20),
        ("Landing Page CVR", 0.04, 0.25),
    ]
    rows = []
    for name, baseline, mde in scenarios:
        n = calculate_sample_size(baseline, mde)
        rows.append({"Test": name, "Baseline": f"{baseline:.1%}",
                      "MDE": f"{mde:.0%}", "N per Group": f"{n:,}"})
    print(pd.DataFrame(rows).to_string(index=False))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print_sample_size_table()
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    print("=" * 60)
