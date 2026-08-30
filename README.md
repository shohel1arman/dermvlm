# DermVLM — Perceive-then-Reason on Device 


## 0. Hardware gate
    sysctl -n hw.memsize | awk '{printf "%.0f GB\n", $1/1073741824}'
8 GB -> use only Qwen3.5-0.8B / 2B and gemma-4-E2B. 16 GB -> full model list in configs/models.yaml.

## 1. Environment
    cd ~/Desktop && unzip -o dermvlm.zip && cd dermvlm
    conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
    uv venv --python 3.11 && source .venv/bin/activate
    uv pip install -U mlx-vlm mlx-lm pandas numpy scikit-learn pillow tqdm pyyaml
    python -c "import mlx_vlm, mlx_lm; print(mlx_vlm.__version__, mlx_lm.__version__)"
    python -c "import mlx_vlm, inspect; print(inspect.signature(mlx_vlm.generate))"
    python -c "import mlx_lm, inspect; print(inspect.signature(mlx_lm.generate))"
Paste those three outputs to me -> I adapt the two ADAPT functions in src/run_vlm.py / src/run_slm.py if needed.

## 2. Manifests (from TrustSkin_R splits)
    head -1 ~/Desktop/TrustSkin_R/data/splits/test.csv        # paste header; adjust --img_col/--lbl_col/--id_col if needed
    python src/make_manifest.py --mask_dir /path/to/tschandl_masks   # omit --mask_dir if masks are not on the Mac yet
    python src/make_external_subset.py --cap 300

## 3. Smoke test + timing (do for EVERY model you plan to use)
    IMG=$(python -c "import pandas as pd; print(pd.read_csv('data/ham_test_manifest.csv').image_path[0])")
    /usr/bin/time -l python -m mlx_vlm.generate --model mlx-community/Qwen3.5-2B-4bit --image "$IMG" \
      --prompt "Describe the dermoscopic features of this skin lesion." --max-tokens 200 --temperature 0.0 2>&1 | tail -25
Record seconds/image and "maximum resident set size". This sets the budget.

## 4. Prompt validation on the dev set (val split ONLY), then freeze
    python src/run_vlm.py --model mlx-community/Qwen3.5-2B-4bit --manifest data/ham_val_dev200.csv --prompt_id A_v1 --limit 20
    tail -3 outputs/*.jsonl | python -m json.tool
If parse_fail is high, we fix the prompt wording here — never on test. Then:
    git add -A && git commit -m "prompts frozen" && git tag prompts-frozen

## 5. Main runs (background, resumable). Repeat for each model x prompt_id
    caffeinate -i python src/run_vlm.py --model mlx-community/Qwen3.5-2B-4bit --manifest data/ham_test_manifest.csv --prompt_id A_v1
    ... A_v2, A_v3, A_abstain_v1, B1_v1, B1_v2, B1_v3
A helper loop (edit the model list first):
    for M in mlx-community/Qwen3.5-0.8B-4bit mlx-community/Qwen3.5-2B-4bit; do for P in A_v1 A_v2 A_v3 A_abstain_v1 B1_v1 B1_v2 B1_v3; do
      caffeinate -i python src/run_vlm.py --model $M --manifest data/ham_test_manifest.csv --prompt_id $P; done; done

## 6. Stage 2 (SLM over each B1 output) + decoupling control (same model as SLM)
    caffeinate -i python src/run_slm.py --model mlx-community/Qwen3.5-4B-4bit --b1_jsonl outputs/Qwen3.5-2B-4bit__B1_v1__ham_test_manifest.jsonl --prompt_id B2_v1
    caffeinate -i python src/run_slm.py --model mlx-community/Qwen3.5-2B-4bit --b1_jsonl outputs/Qwen3.5-2B-4bit__B1_v1__ham_test_manifest.jsonl --prompt_id B2_v1   # control C

## 7. Counterfactual faithfulness (needs masks)
    python src/make_counterfactuals.py
    caffeinate -i python src/run_vlm.py --model <M> --manifest data/cf_manifest.csv --prompt_id A_v1
    caffeinate -i python src/run_vlm.py --model <M> --manifest data/cf_manifest.csv --prompt_id B1_v1

## 8. External shift + Derm7pt
    caffeinate -i python src/run_vlm.py --model <M> --manifest data/external_subset.csv --prompt_id A_v1
    (Derm7pt: build data/derm7pt_manifest.csv with image_id,image_path,label; run B1_v1; then python src/score_derm7pt.py <B1 jsonl> <derm7pt meta csv>)

## 9. Analysis (any time — it reads whatever outputs exist)
    python src/analyze.py && ls results/
Send me results/*.csv -> figures, stats and manuscript.

## Self-test (synthetic, never for results)
    python src/selftest.py && BOOT=200 python src/analyze.py && rm -rf outputs/* results/* data/ham_test_manifest.csv
