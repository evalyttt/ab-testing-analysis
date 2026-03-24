"""
A/B Testing Analysis Framework
Statistical analysis for evaluating experiment results with hypothesis testing,
confidence intervals, and business impact estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Calculate required sample size per group for a two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    pooled = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    return int(np.ceil(n))


def run_conversion_test(control_conversions, control_total,
                        treatment_conversions, treatment_total, alpha=0.05):
    """Run chi-square test for conversion rate comparison."""
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total
    lift = (treatment_rate - control_rate) / control_rate

    contingency = np.array([
        [control_conversions, control_total - control_conversions],
        [treatment_conversions, treatment_total - treatment_conversions]
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'lift': lift,
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < alpha
    }


def run_continuous_test(control_values, treatment_values, alpha=0.05):
    """Run Welch's t-test for continuous metric comparison."""
    control_mean = np.mean(control_values)
    treatment_mean = np.mean(treatment_values)
    lift = (treatment_mean - control_mean) / control_mean

    t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)

    pooled_std = np.sqrt((np.std(control_values)**2 + np.std(treatment_values)**2) / 2)
    cohens_d = (treatment_mean - control_mean) / pooled_std

    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'lift': lift,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < alpha
    }


def plot_conversion_comparison(result, title="Conversion Rate: Control vs Treatment"):
    """Visualize conversion rate comparison with confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = ['Control', 'Treatment']
    rates = [result['control_rate'] * 100, result['treatment_rate'] * 100]
    colors = ['#90CAF9', '#4CAF50' if result['significant'] else '#FFC107']

    bars = ax.bar(groups, rates, color=colors, width=0.5, edgecolor='gray')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rate:.2f}%', ha='center', fontsize=12, fontweight='bold')

    sig_text = f"p = {result['p_value']:.4f} ({'Significant' if result['significant'] else 'Not Significant'})"
    ax.set_title(f"{title}\n{sig_text}", fontsize=13)
    ax.set_ylabel('Conversion Rate (%)')
    plt.tight_layout()
    plt.savefig('visuals/conversion_comparison.png', dpi=150)
    plt.close()


# ============================================================
# Demo: Run Sample A/B Test Analysis
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)

    # 1. Sample Size Planning
    required_n = calculate_sample_size(baseline_rate=0.032, mde=0.15)
    print(f"Required sample size per group: {required_n:,}")

    # 2. Conversion Rate Test
    print("\n=== Conversion Rate Test ===")
    conv_result = run_conversion_test(
        control_conversions=320, control_total=10000,
        treatment_conversions=380, treatment_total=10000
    )
    for k, v in conv_result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 3. Average Order Value Test
    print("\n=== Avg Order Value Test ===")
    control_aov = np.random.normal(47.20, 15.0, 10000)
    treatment_aov = np.random.normal(49.80, 16.0, 10000)
    aov_result = run_continuous_test(control_aov, treatment_aov)
    for k, v in aov_result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 4. Summary Table
    print("\n=== Results Summary ===")
    summary = pd.DataFrame([
        {'Metric': 'Conversion Rate', 'Control': f"{conv_result['control_rate']:.1%}",
         'Treatment': f"{conv_result['treatment_rate']:.1%}",
         'Lift': f"+{conv_result['lift']:.1%}", 'p-value': f"{conv_result['p_value']:.3f}",
         'Significant': conv_result['significant']},
        {'Metric': 'Avg Order Value', 'Control': f"${aov_result['control_mean']:.2f}",
         'Treatment': f"${aov_result['treatment_mean']:.2f}",
         'Lift': f"+{aov_result['lift']:.1%}", 'p-value': f"{aov_result['p_value']:.3f}",
         'Significant': aov_result['significant']},
    ])
    print(summary.to_string(index=False))
