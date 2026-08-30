"""Perception accuracy (RQ4): B1 descriptions vs Derm7pt seven-point-checklist labels.
[VERIFY] after cloning github.com/jeremykawahara/derm7pt: the exact CSV column names and label vocab in derm7pt/dataset.py.
Fill MAP below (B1 key -> (Derm7pt column, function mapping our value -> Derm7pt label))."""
import json, sys, pandas as pd
from sklearn.metrics import f1_score
b1_jsonl, derm7pt_csv = sys.argv[1], sys.argv[2]
def tri(v):  # our 3-way value -> Derm7pt-style label; EDIT to match dataset.py vocabulary
    v = str(v).lower()
    return "absent" if "absent" in v else ("atypical" if "irregular" in v else "typical")
MAP = {  # EDIT column names to Derm7pt's meta CSV
    "pigment_network": ("pigment_network", tri), "streaks": ("streaks", tri), "dots_globules": ("dots_and_globules", tri),
    "blue_white_veil": ("blue_whitish_veil", lambda v: "absent" if "absent" in str(v).lower() else "present"),
    "regression_structures": ("regression_structures", lambda v: "absent" if "absent" in str(v).lower() else "present"),
    "vascular_structures": ("vascular_structures", tri),
}
pred = {json.loads(l)["image_id"]: json.loads(l)["parsed"] for l in open(b1_jsonl) if l.strip()}
gt = pd.read_csv(derm7pt_csv).set_index("image_id")   # EDIT index column to whatever you used as image_id in the manifest
rows = []
for key, (col, fn) in MAP.items():
    ids = [i for i in pred if pred[i] and key in pred[i] and i in gt.index]
    y = gt.loc[ids, col].astype(str).str.lower(); yh = [fn(pred[i][key]) for i in ids]
    rows.append(dict(criterion=key, n=len(ids), macro_f1=f1_score(y, yh, average="macro", zero_division=0)))
out = pd.DataFrame(rows); out.loc[len(out)] = dict(criterion="MEAN", n=out.n.sum(), macro_f1=out.macro_f1.mean())
out.to_csv("results/perception_derm7pt.csv", index=False); print(out)
