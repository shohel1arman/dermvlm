"""Stage-2 reasoning: structured description (from a B1 jsonl) -> diagnosis JSON, text-only.
[VERIFY] mlx_lm API:  python -c "import mlx_lm, inspect; print(mlx_lm.__version__); print(inspect.signature(mlx_lm.generate))"
If the reasoning model is a multimodal checkpoint that mlx_lm cannot load, run it through mlx_vlm with no image instead.
"""
import argparse, json, os, sys, time, yaml
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(__file__))
from common import parse_json, norm_dx, norm_conf, done_ids

def load_model(repo):                                    # ADAPT
    from mlx_lm import load
    return load(repo)

def run_one(model, tokenizer, prompt, max_tokens):       # ADAPT
    from mlx_lm import generate
    msgs = [{"role": "user", "content": prompt}]
    try:
        fp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        fp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=fp, max_tokens=max_tokens, verbose=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reasoning model repo")
    ap.add_argument("--b1_jsonl", required=True, help="outputs/<vlm>__B1_vX__<manifest>.jsonl")
    ap.add_argument("--prompt_id", default="B2_v1")
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tmpl = yaml.safe_load(open("configs/prompts.yaml"))[a.prompt_id]
    rows = [json.loads(l) for l in open(a.b1_jsonl) if l.strip()]
    if a.limit: rows = rows[:a.limit]
    vlm_tag = os.path.basename(a.b1_jsonl).split("__")[0]
    mani = os.path.basename(a.b1_jsonl).split("__")[-1].replace(".jsonl", "")
    b1_pid = os.path.basename(a.b1_jsonl).split("__")[1]
    tag = a.model.split("/")[-1]
    out_path = f"outputs/{vlm_tag}__{b1_pid}__{tag}__{a.prompt_id}__{mani}.jsonl"
    done = done_ids(out_path)
    model, tok = load_model(a.model)
    with open(out_path, "a") as f:
        for r in tqdm(rows, desc=f"{tag} over {vlm_tag}"):
            if r["image_id"] in done: continue
            if not r.get("parsed"):                       # no description -> propagate as abstention
                rec = dict(image_id=r["image_id"], label=r.get("label"), vlm=vlm_tag, slm=a.model, prompt_id=a.prompt_id,
                           manifest=mani, raw="", parsed=None, parse_fail=1, diagnosis=None, confidence=None,
                           latency_s=0.0, error="no_description")
                f.write(json.dumps(rec) + "\n"); continue
            desc = {k: v for k, v in r["parsed"].items() if k != "diagnosis"}   # belt-and-braces: strip any leaked dx
            prompt = tmpl.replace("{DESCRIPTION_JSON}", json.dumps(desc))
            t0 = time.time(); err = ""
            try:
                raw = run_one(model, tok, prompt, a.max_tokens)
            except Exception as e:
                raw, err = "", repr(e)
            p = parse_json(raw)
            rec = dict(image_id=r["image_id"], label=r.get("label"), vlm=vlm_tag, slm=a.model, prompt_id=a.prompt_id,
                       manifest=mani, raw=raw, parsed=p, parse_fail=int(p is None),
                       diagnosis=norm_dx(p.get("diagnosis")) if p else None,
                       confidence=norm_conf(p.get("confidence")) if p else None,
                       latency_s=round(time.time() - t0, 3), error=err)
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
    print("done ->", out_path)
