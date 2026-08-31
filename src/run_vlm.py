"""Batch VLM inference (image -> JSON) with resume + latency logging.
[VERIFY] the two ADAPT functions against your installed mlx-vlm before the full run:
    python -c "import mlx_vlm, inspect; print(mlx_vlm.__version__); print(inspect.signature(mlx_vlm.generate))"
CLI form (confirmed in mlx-vlm docs):  python -m mlx_vlm.generate --model <repo> --image <path> --prompt "<p>" --max-tokens 200 --temperature 0.0
"""
import argparse, json, os, sys, time, yaml
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(__file__))
from common import parse_json, norm_dx, norm_conf, done_ids

CFG = {}
def load_model(repo):                                    # verified against mlx-vlm 0.6.17
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    CFG["config"] = load_config(repo)
    return load(repo)                                    # -> (model, processor)

def run_one(model, processor, image_path, prompt, max_tokens):   # ADAPT
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    fp = apply_chat_template(processor, CFG["config"], prompt, num_images=1, enable_thinking=False)
    out = generate(model, processor, fp, image_path, max_tokens=max_tokens, temperature=0.0,
                   enable_thinking=False, seed=42, verbose=False)
    return out if isinstance(out, str) else getattr(out, "text", str(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt_id", required=True)
    ap.add_argument("--max_tokens", type=int, default=0, help="0 = auto: 220 for A prompts, 400 for B1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    prompt = yaml.safe_load(open("configs/prompts.yaml"))[a.prompt_id]
    if a.max_tokens == 0: a.max_tokens = 400 if a.prompt_id.startswith("B1") else 220
    df = pd.read_csv(a.manifest)
    if a.limit: df = df.head(a.limit)
    tag = a.model.split("/")[-1]
    mani = os.path.basename(a.manifest).replace(".csv", "")
    out_path = f"outputs/{tag}__{a.prompt_id}__{mani}.jsonl"
    done = done_ids(out_path)
    model, processor = load_model(a.model)
    n_fail = 0
    with open(out_path, "a") as f:
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{tag} {a.prompt_id}"):
            if r["image_id"] in done: continue
            t0 = time.time(); err = ""
            try:
                raw = run_one(model, processor, r["image_path"], prompt, a.max_tokens)
            except Exception as e:
                raw, err = "", repr(e)
            p = parse_json(raw)
            rec = dict(image_id=r["image_id"], label=r.get("label", None), model=a.model, prompt_id=a.prompt_id,
                       manifest=mani, raw=raw, parsed=p, parse_fail=int(p is None),
                       diagnosis=norm_dx(p.get("diagnosis")) if p else None,
                       confidence=norm_conf(p.get("confidence")) if p else None,
                       latency_s=round(time.time() - t0, 3), error=err)
            n_fail += rec["parse_fail"]
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
    print(f"done -> {out_path}  parse_fail={n_fail}")
