# DermVLM — Perceive-then-Reason On Device

A multi-dimensional **trustworthiness evaluation** of small, on-device vision-language models (VLMs) and small language models (SLMs) for **dermoscopic skin-lesion assessment**. All experiments are inference-only and run on a single Apple-Silicon machine (Mac mini, M2, 16 GB).

Companion study to **TrustSkin** (supervised CNN/ViT trust evaluation); this work extends the accuracy-vs-trust decoupling question to zero-shot generalist VLMs.

## Models (4-bit MLX)
Qwen3.5-0.8B, Qwen3.5-2B, Gemma 4 E2B — via `mlx-vlm` 0.6.17 / `mlx-lm` 0.31.3.

## Data
- **HAM10000** (7 classes), leakage-controlled split from TrustSkin (test n=2014, val n=1020)
- **Tschandl lesion segmentations** — counterfactual faithfulness (n=242 stratified)
- **ISIC-2019**, de-contaminated external set (stratified subset n=1735)
- **Derm7pt** (395 dermoscopic test cases) — perception accuracy vs seven-point checklist

## Conditions
- **A** — end-to-end VLM diagnosis (JSON: diagnosis, verbalized confidence, rationale); 3 paraphrases + abstention variant
- **B** — perceive→reason: VLM emits a structured feature description (B1), a separate SLM diagnoses from text only (B2)
- **Control** — same model reasons over its own description (diagonal of the 3×3 grid), isolating decomposition from model change

## Trust axes
Balanced accuracy, macro-F1, MCC, melanoma recall; calibration (ECE/Brier/NLL on verbalized confidence); selective prediction (AURC, risk@coverage); paraphrase consistency; counterfactual faithfulness (lesion vs same-size control mask); Derm7pt perception; on-device cost (latency). Bootstrap 95% CIs.

## Headline findings
- **Accuracy and trust decouple**: the best-accuracy model (Qwen3.5-2B, bal-acc ≈0.41) is also best-calibrated, but ranking on trust axes does not follow accuracy across models.
- **Prompt fragility scales with size**: unanimous-label rate across paraphrases — Qwen3.5-2B 76%, Gemma4-E2B 18%, Qwen3.5-0.8B 2%.
- **Faithfulness separates models**: lesion-vs-control flip delta — Qwen3.5-2B 0.56, Qwen3.5-0.8B 0.24, Gemma4-E2B 0.02 (Gemma barely uses the lesion).
- **Decomposition trades accuracy for calibration**; the control shows gains come from the reasoner, not decomposition itself; some pairings collapse (degenerate melanoma recall).
- **Calibration collapses under shift** (HAM→ISIC): ECE roughly doubles across models while accuracy holds up better.
- **Perception is criterion-dependent**: blue-white veil recognized best; dots/globules and streaks near-random.

## Reproduce
```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -U mlx-vlm mlx-lm pandas numpy scikit-learn pillow tqdm pyyaml matplotlib kaggle
python src/make_manifest.py --img_root ~/Desktop/data/ham10000 --mask_dir <tschandl_masks>
python src/make_external_subset.py
python src/make_counterfactuals.py
python src/make_derm7pt_manifest.py
python src/run_vlm.py --model <repo> --manifest data/ham_test_manifest.csv --prompt_id A_v1
python src/run_slm.py --model <repo> --b1_jsonl outputs/<...>__B1_v1__ham_test_manifest.jsonl --prompt_id B2_v1
python src/reparse.py && python src/analyze.py
python src/score_derm7pt.py
python src/make_figures.py
```

## Repo layout
- `src/` — runners (`run_vlm`, `run_slm`), data prep, `analyze.py`, `score_derm7pt.py`, `make_figures.py`
- `configs/prompts.yaml` — frozen prompts (tag `prompts-frozen`)
- `outputs/` — raw per-image JSONL (resumable)
- `results/` — computed metrics (`per_run.csv`, `faithfulness.csv`, `consistency.csv`, `perception_derm7pt.csv`)
- `fig/` — publication figures (PDF + PNG, 300 dpi)

## Integrity
Every number traces to a committed JSONL. Prompts frozen before test. Parse-failure and abstention rates reported, not hidden. Quantization is part of the system under test. Verbalized confidence is a self-report, not a posterior.

## Data licences
HAM10000 CC BY-NC 4.0; Tschandl segmentations CC BY-NC 4.0; ISIC-2019 CC BY-NC 4.0; Derm7pt CC BY-NC-ND 4.0 (Kawahara et al., IEEE JBHI 2019, doi:10.1109/JBHI.2018.2824327). Research/non-commercial use.