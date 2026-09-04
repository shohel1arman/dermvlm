"""Derm7pt test-split manifest (dermoscopic images). Maps their diagnosis to our 7 codes where possible."""
import pandas as pd, os
root = os.path.expanduser("~/Desktop/data/derm7pt/release_v0")
meta = pd.read_csv(f"{root}/meta/meta.csv")
test_idx = pd.read_csv(f"{root}/meta/test_indexes.csv")["indexes"].tolist()
df = meta.iloc[test_idx].copy()
# diagnosis text -> our HAM codes (best-effort; unmapped -> 'other', kept for perception scoring not accuracy)
def code(d):
    d = str(d).lower()
    if "melanoma" in d and "metast" not in d: return "mel"
    if "nevus" in d or "naevus" in d or "nevi" in d: return "nv"
    if "basal cell" in d: return "bcc"
    if "seborrheic keratosis" in d or "lentigo" in d or "dermatofibroma" in d or "keratosis" in d: return "bkl"
    if "vascular" in d or "angioma" in d or "angiokeratoma" in d: return "vasc"
    return "other"
df["image_id"] = "d7_" + df["case_num"].astype(str)
df["image_path"] = df["derm"].map(lambda p: f"{root}/images/{p}")
df["label"] = df["diagnosis"].map(code)
df["mask_path"] = ""
exists = df.image_path.map(os.path.exists)
print("test cases:", len(df), "| derm images found:", int(exists.sum()), "| missing:", int((~exists).sum()))
print("label dist:", df.label.value_counts().to_dict())
df = df[exists]
df[["image_id","image_path","label","mask_path"]].to_csv(os.path.expanduser("~/Desktop/dermvlm/data/derm7pt_manifest.csv"), index=False)
# keep the checklist ground truth alongside, for scoring later
keep = ["case_num","pigment_network","streaks","regression_structures","dots_and_globules","blue_whitish_veil","vascular_structures"]
g = df.copy(); g["image_id"] = "d7_" + g["case_num"].astype(str)
g[["image_id"]+keep[1:]].to_csv(os.path.expanduser("~/Desktop/dermvlm/data/derm7pt_truth.csv"), index=False)
print("wrote data/derm7pt_manifest.csv and data/derm7pt_truth.csv")
