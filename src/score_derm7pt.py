import json, glob, os, pandas as pd
from sklearn.metrics import f1_score
truth = pd.read_csv("data/derm7pt_truth.csv").set_index("image_id")
def tri(v):
    v=str(v).lower(); return "absent" if "absent" in v else ("atypical" if "irregular" in v else "present")
def gt_tri(v):
    v=str(v).lower(); return "absent" if "absent" in v else ("atypical" if ("atypical" in v or "irregular" in v) else "present")
def bin_(v):
    v=str(v).lower(); return "absent" if "absent" in v else "present"
def gt_bin(v):
    v=str(v).lower(); return "absent" if "absent" in v else "present"
MAP={"pigment_network":("pigment_network",tri,gt_tri),"streaks":("streaks",tri,gt_tri),
     "dots_globules":("dots_and_globules",tri,gt_tri),
     "blue_white_veil":("blue_whitish_veil",bin_,gt_bin),
     "regression_structures":("regression_structures",bin_,gt_bin),
     "vascular_structures":("vascular_structures",tri,gt_tri)}
rows=[]
for f in glob.glob("outputs/*__B1_v1__derm7pt_manifest.jsonl"):
    model=os.path.basename(f).split("__")[0]
    pred={json.loads(l)["image_id"]:json.loads(l).get("parsed") for l in open(f) if l.strip()}
    for key,(col,fn,gfn) in MAP.items():
        ids=[i for i in pred if pred[i] and isinstance(pred[i],dict) and key in pred[i] and i in truth.index]
        if len(ids)<10: continue
        y=[gfn(truth.loc[i,col]) for i in ids]; yh=[fn(pred[i][key]) for i in ids]
        rows.append(dict(model=model,criterion=key,n=len(ids),macro_f1=f1_score(y,yh,average="macro",zero_division=0)))
out=pd.DataFrame(rows); out.to_csv("results/perception_derm7pt.csv",index=False)
print(out.pivot(index="model",columns="criterion",values="macro_f1").round(3))
