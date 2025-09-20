# Stated Preference for Interaction and Continued Engagement (SPICE)

Code and data generation and analysis pipeline for the paper:

> **Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM’s Willingness to Re‑engage in Conversation**  
> Rost, Figlia, Wallraff — arXiv:2509.09043 (https://arxiv.org/abs/2509.09043)

This repository contains the full pipeline to elicit, log, and analyze SPICE (a strict YES/NO stated preference to continue an interaction after reviewing a transcript). The pipeline is **notebook‑first** with reproducible Conda environment and optional `.py` pairing via jupytext.

**AI‑assist disclaimer**: Portions of this repository (including comments) were authored with the assistance of AI. All logic and decisions remain the authors’ responsibility.

---

## Files & notebooks

- `interactions.json` and `question_sequences.json` — inputs 
- `01_SPICE_experiment.ipynb` — runs model elicitation and writes a **pipe‑delimited** results CSV into a timestamped folder under `runs_replication/`.
- `02_analysis_visualisation.ipynb` — frequentist analyses, appendix tables, minimal figures.
- `03_bayesian_appendix.ipynb` — Beta–Binomial posterior summaries and contrasts.

---

## Environment (Conda, reproducible)


```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python -m ipykernel install --user --name spice --display-name "Python (spice)"
```

---

## Backend: Ollama (default)

1) Install Ollama (https://ollama.com) and ensure it’s running locally.  
2) Pull the models used in the paper:
```bash
ollama pull gemma2:9b
ollama pull gemma3:12b
ollama pull llama3.1:8b
ollama pull mistral:7b
```
3) (Optional) set the endpoint explicitly (useful in containers/remote):
```bash
export OLLAMA_API_BASE=http://127.0.0.1:11434
```

---

## Data files

`interactions.json`, `question_sequences.json`.  
These are copied into each run folder for provenance.

---

## Running the pipeline (notebook flow)

Open JupyterLab and run all cells top‑to‑bottom:

1. **01_SPICE_experiment.ipynb** → creates `runs_replication/run_YYYYMMDD_HHMMSS/llm_preference_results.csv`  
2. **02_analysis_visualisation.ipynb** → writes `appendix_tables/`, `robustness_outputs/`, `figures_minimal/` inside the latest run  
3. **03_bayesian_appendix.ipynb** → writes `bayes_outputs/` inside the latest run

### Headless (optional)

```bash
papermill 01_SPICE_experiment.ipynb out/01_experiment_run.ipynb -k "Python (spice)"
papermill 02_analysis_visualisation.ipynb out/02_analysis_run.ipynb -k "Python (spice)"
papermill 03_bayesian_appendix.ipynb out/03_bayes_run.ipynb -k "Python (spice)"
```

---

## Design choices (brief)

- **Strict YES/NO parsing** for SPC/abuse/adequacy; non‑binary outputs are flagged `INVALID` to avoid silent misclassification.
- **No Q/A chaining**: each question runs independently on a fixed baseline.
- **Deterministic decoding** (`temperature=0`, seeds logged) for reproducibility.
- **Clustering on `interaction_id`**: ICC‑based design effects, Rao–Scott corrections, cluster permutation, and cluster‑robust GLMs.
- **Per‑run snapshots**: inputs are copied into each run directory.

---

## License

**CC‑BY‑4.0** (code, notebooks, and repository assets).

---

## Citation (arXiv)

```
@misc{rost2025spice,
  title        = {Stated Preference for Interaction and Continued Engagement (SPICE):
                  Evaluating an LLM's Willingness to Re-engage in Conversation},
  author       = {Thomas Manuel Rost and Martina Figlia and Bernd Wallraff},
  year         = {2025},
  eprint       = {2509.09043},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2509.09043}
}
```

## Troubleshooting

- **No runs found** — run the experiment notebook first; outputs appear under `runs_replication/run_*`.
- **CSV not found** — each analysis notebook looks for `llm_preference_results.csv` in the latest run directory.
- **Backend errors** — ensure Ollama is running and that `OLLAMA_API_BASE` (if set) points to the correct host/port.
- **Model not found** — pull the model via `ollama pull <model:tag>`.
