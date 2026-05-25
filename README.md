# HC-MoE

This project studies whether the **Higher Criticism (HC)** statistic — applied to per-expert routing p-values in Mixture-of-Experts (MoE) language models — predicts per-token cross-entropy loss and the *intelligence gain* obtained by scaling from a small model to a large one within the same family.

The core question: do routing anomalies (a few experts firing far from their null behaviour) flag tokens that are harder to predict, and do they reveal where a larger model has an advantage?

For a full description of the research goals, nomenclature, findings, and open questions, see the **[research report](HC_MoE_Report.html)** (open in a browser).

## Repo Structure

- `empirical_null.py` — empirical null estimation via central matching (Efron-style).
- `empirical_null_lindsey.py` — GLM / Student's t null estimation (Lindsey-inspired).
- `measurements/` — correlation CSVs and summary tables used in the report.
- `figures/` — generated plots (gitignored by default).
- `HC_MoE_Report.html` — self-contained research report with interactive figures.
- `requirements.txt` — Python dependencies (`pip install -r requirements.txt`).
