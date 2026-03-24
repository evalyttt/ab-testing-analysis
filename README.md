# A/B Testing Analysis Framework

Statistical analysis framework for evaluating A/B test results, including sample size calculation, significance testing, and business impact estimation.

## Overview
Built a reusable A/B testing analysis pipeline that automates hypothesis testing, calculates confidence intervals, and generates executive-ready visualizations for marketing and product experiments.

## Features
- Sample size calculator for experiment planning
- Chi-square and t-test for conversion rate and continuous metric analysis
- Confidence interval visualization
- Effect size estimation (Cohen's d)
- Sequential testing support for early stopping decisions

## Example Results
| Metric | Control | Treatment | Lift | p-value | Significant? |
|--------|---------|-----------|------|---------|--------------|
| Conversion Rate | 3.2% | 3.8% | +18.7% | 0.023 | Yes |
| Avg Order Value | $47.20 | $49.80 | +5.5% | 0.081 | No |
| Revenue/User | $1.51 | $1.89 | +25.2% | 0.008 | Yes |

## Tools & Technologies
- **Python**: scipy, statsmodels, pandas, matplotlib, seaborn
- **Statistical Methods**: Chi-square test, Welch's t-test, Bootstrap CI, Bayesian estimation

## Project Structure
```
├── ab_test_analysis.py    # Core analysis module
├── README.md
└── visuals/               # Output charts
```

## How to Run
```bash
pip install pandas numpy scipy matplotlib seaborn
python ab_test_analysis.py
```
