"""Synthetic self-test: fabricates model outputs (clearly labelled fake) to exercise parser + analysis. Never use for results."""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import CLASSES, parse_json, norm_dx, norm_conf
rng = np.random.default_rng(0); os.makedirs("outputs", exist_ok=True); os.makedirs("data", exist_ok=True)
ids = [f"ISIC_{i:07d}" for i in range(300)]; labels = rng.choice(CLASSES, 300, p=[.07,.1,.11,.02,.11,.55,.04])
pd.DataFrame(dict(image_id=ids, image_path="x.jpg", label=labels, mask_path="")).to_csv("data/ham_test_manifest.csv", index=False)
def fake(acc, conf_bias):
    for sysname in ["FAKE-VLM-A", "FAKE-VLM-B"]:
        for pid in ["A_v1", "A_v2", "A_v3"]:
            with open(f"outputs/{sysname}__{pid}__ham_test_manifest.jsonl", "w") as f:
                for i, l in zip(ids, labels):
                    dx = l if rng.random() < acc[sysname] else rng.choice(CLASSES)
                    conf = int(np.clip(rng.normal(70 + conf_bias[sysname], 15), 0, 100))
                    raw = "```json\n" + json.dumps(dict(diagnosis=dx, confidence=conf, rationale="fake")) + "\n```" if rng.random() > .05 else "I cannot tell."
                    p = parse_json(raw)
                    f.write(json.dumps(dict(image_id=i, label=l, model=sysname, prompt_id=pid, raw=raw, parsed=p, parse_fail=int(p is None),
                            diagnosis=norm_dx(p.get("diagnosis")) if p else None, confidence=norm_conf(p.get("confidence")) if p else None,
                            latency_s=float(rng.uniform(2, 6)), error="")) + "\n")
        # counterfactual variants for A_v1 only
        with open(f"outputs/{sysname}__A_v1__cf_manifest.jsonl", "w") as f:
            for i, l in zip(ids[:100], labels[:100]):
                for var, flip_p in [("lesion", .6 if sysname == "FAKE-VLM-A" else .2), ("control", .1), ("background", .3)]:
                    dx = rng.choice(CLASSES) if rng.random() < flip_p else l
                    f.write(json.dumps(dict(image_id=f"{i}__{var}", label=l, model=sysname, prompt_id="A_v1", raw="", parsed={}, parse_fail=0,
                            diagnosis=dx, confidence=float(rng.uniform(40, 90)), latency_s=3.0, error="")) + "\n")
fake(acc={"FAKE-VLM-A": .55, "FAKE-VLM-B": .35}, conf_bias={"FAKE-VLM-A": 0, "FAKE-VLM-B": 20})
# a B1 + B2 pair to exercise the 5-part filename path
with open("outputs/FAKE-VLM-A__B1_v1__ham_test_manifest.jsonl", "w") as f:
    for i, l in zip(ids, labels):
        f.write(json.dumps(dict(image_id=i, label=l, parsed=dict(pigment_network="present_irregular", colors=["brown"]), parse_fail=0, diagnosis=None, confidence=None, latency_s=4.0)) + "\n")
with open("outputs/FAKE-VLM-A__B1_v1__FAKE-SLM__B2_v1__ham_test_manifest.jsonl", "w") as f:
    for i, l in zip(ids, labels):
        dx = l if rng.random() < .45 else rng.choice(CLASSES)
        f.write(json.dumps(dict(image_id=i, label=l, parsed={}, parse_fail=0, diagnosis=dx, confidence=float(rng.uniform(30, 95)), latency_s=1.0)) + "\n")
print("synthetic outputs written")
