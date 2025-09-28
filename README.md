# ConveyorAI — Confidence Score via Logistic Regression (Binary LLM Features)

This repo/notebook implements a calibrated **confidence score** for ConveyorAI answers using **binary LLM-derived features** and a **logistic regression** model.

**Pipeline**
1) Load Digisign dataset (`human_eval_data.csv`)
2) One **LLM JSON** call per row to extract **binary failure‑mode features**
3) Label = **1 only** when `conveyor_ai_grade == "perfect"`, else 0
4) Train **Logistic Regression** on all rows to predict **p(accurate) ∈ [0, 1]**
5) Show a **confusion matrix** (threshold 0.5)
6) Plot **predicted p(accurate)** vs original categorical grade

---

## Files

- **Notebook:** `conveyor_lr_binary_features.ipynb`
- **Dataset (expected path by default):** `/mnt/data/human_eval_data.csv`
  - You can change the path in the first code cell (`DATA_PATH`) to e.g. `./human_eval_data.csv`
- **Dependencies:** `requirements.txt`

---

## Installation

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

> If you prefer conda:
> ```bash
> conda create -n conveyor python=3.11 -y
> conda activate conveyor
> pip install -r requirements.txt
> ```

---

## Environment Variables (optional, for live LLM calls)

If you want to use the LLM to generate features (instead of the built‑in stub), set:

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..." 
export LLM_MODEL_NAME="gpt-5-nano"
export LLM_TEMPERATURE=0
export MAX_CONCURRENCY=8

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:LLM_MODEL_NAME="gpt-5-nano"
$env:LLM_TEMPERATURE="0"
$env:MAX_CONCURRENCY="8"
```

- If `OPENAI_API_KEY` is **not** set, the notebook uses a **lightweight heuristic stub** so the full pipeline still runs for demos.

---

## Dataset Schema

Expected columns in `human_eval_data.csv`:

- `id`
- `question`
- `conveyor_ai_answer`
- `conveyor_ai_grade` — values like: `perfect`, `accurate_imperfect`, `inaccurate`, `unanswered`
- `source_1 … source_11` — non‑empty values are passed to the LLM as strings (retrieved sources)

> Labeling rule in this notebook: **only** `perfect` → 1; all others → 0.

---

## How to Run

### A) Jupyter (interactive)
```bash
jupyter notebook
```
Open **`conveyor_lr_binary_features.ipynb`**, then **Run All**.

### B) Headless (no UI)
```bash
jupyter nbconvert --to notebook --execute   --ExecutePreprocessor.timeout=1800   --output executed_conveyor_lr_binary_features.ipynb   conveyor_lr_binary_features.ipynb
```

This executes all cells and writes an executed copy with outputs embedded.

---

## Outputs

- **Confusion Matrix** (printed + plotted) at threshold 0.5
- **Classification report** (precision/recall/F1)
- **Box plot:** `predicted p(accurate)` vs **original** `conveyor_ai_grade`

> Plots are displayed in the notebook. If running headless, they are embedded in the executed notebook output.

---

## Knobs & Notes

- Change model: `LLM_MODEL_NAME` (defaults to `gpt-5-nano` per spec)
- Keep deterministic: `LLM_TEMPERATURE=0`
- Concurrency for LLM calls: `MAX_CONCURRENCY`
- Adjust confusion‑matrix threshold in the cell if you want a different operating point