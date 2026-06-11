import numpy as np
from scipy import stats
from itertools import combinations

# --- Raw Results from Experiment ---
results = {
    0.1: [18.50, 17.92, 20.28, 20.48, 16.83, 20.49, 12.73, 25.34, 17.97, 17.60],
    0.3: [19.52, 19.63, 21.38, 21.57, 18.82, 21.57, 17.75, 26.83, 19.62, 18.97],
    0.5: [20.69, 21.21, 22.22, 21.72, 20.59, 23.15, 16.34, 27.81, 21.48, 20.03],
    0.7: [20.85, 20.43, 23.29, 24.38, 20.23, 24.10, 18.84, 28.15, 24.71, 20.10],
    1.0: [22.52, 23.96, 27.81, 26.69, 24.72, 25.62, 23.32, 27.53, 27.94, 22.10],
}

groups = list(results.keys())
data   = list(results.values())

# --- Summary Statistics ---
print("=" * 55)
print("SUMMARY STATISTICS")
print("=" * 55)
print(f"{'Sep. Weight':<15} {'Mean':>8} {'Std Dev':>10} {'Min':>8} {'Max':>8}")
print("-" * 55)
for g, d in zip(groups, data):
    print(f"{g:<15} {np.mean(d):>8.2f} {np.std(d):>10.2f} {np.min(d):>8.2f} {np.max(d):>8.2f}")

# --- One-Way ANOVA ---
f_stat, p_value = stats.f_oneway(*data)
print("\n" + "=" * 55)
print("ONE-WAY ANOVA")
print("=" * 55)
print(f"F-statistic : {f_stat:.4f}")
print(f"P-value     : {p_value:.4f}")
if p_value < 0.05:
    print("Result      : SIGNIFICANT (p < 0.05)")
    print("             At least one condition differs significantly.")
else:
    print("Result      : NOT SIGNIFICANT (p >= 0.05)")
    print("             No significant difference found between conditions.")

# --- Tukey HSD (manual implementation using t-tests with Bonferroni correction) ---
# scipy does not have built-in Tukey HSD without statsmodels,
# so we use pairwise t-tests with Bonferroni correction as a robust alternative.
print("\n" + "=" * 55)
print("PAIRWISE T-TESTS (Bonferroni corrected)")
print("=" * 55)

pairs = list(combinations(groups, 2))
n_comparisons = len(pairs)
print(f"Number of comparisons: {n_comparisons}")
print(f"Bonferroni-corrected significance threshold: {0.05 / n_comparisons:.4f}\n")
print(f"{'Pair':<18} {'t-stat':>8} {'p-value':>10} {'Corrected p':>13} {'Significant?':>14}")
print("-" * 68)

for g1, g2 in pairs:
    t_stat, p_val = stats.ttest_ind(results[g1], results[g2])
    corrected_p = min(p_val * n_comparisons, 1.0)  # Bonferroni correction
    sig = "YES *" if corrected_p < 0.05 else "no"
    print(f"{str(g1)+' vs '+str(g2):<18} {t_stat:>8.3f} {p_val:>10.4f} {corrected_p:>13.4f} {sig:>14}")

# --- Conclusion ---
print("\n" + "=" * 55)
print("CONCLUSION")
print("=" * 55)
if p_value < 0.05:
    print("The one-way ANOVA shows a statistically significant")
    print("effect of separation weight on average nearest-")
    print("neighbour distance (F={:.3f}, p={:.4f}).".format(f_stat, p_value))
    print("This supports the hypothesis that higher separation")
    print("weight leads to greater inter-agent spacing.")
else:
    print("The one-way ANOVA does not show a statistically")
    print("significant effect (F={:.3f}, p={:.4f}).".format(f_stat, p_value))
    print("The hypothesis is not supported by these results.")
