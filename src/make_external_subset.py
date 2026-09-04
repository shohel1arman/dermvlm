"""Stratified external ISIC subset from the de-contaminated TrustSkin split, mapped to local files."""
import argparse, os, glob, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=os.path.expanduser("~/Desktop/TrustSkin_R"))
ap.add_argument("--img_root", default=os.path.expanduser("~/Desktop/data/isic2019"))
ap.add_argument("--per_class", type=int, default=300)   # cap per class; rare classes kept whole
a = ap.parse_args()
idx = {}
for p in glob.glob(os.path.join(a.img_root, "**", "*.jpg"), recursive=True):
    idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
print("local ISIC images indexed:", len(idx))
ext = pd.read_csv(f"{a.ts}/data/splits/external_isic.csv")
ext["local"] = ext.image_id.map(idx)
miss = ext.local.isna().sum()
ext = ext.dropna(subset=["local"])
sub = pd.concat([g.sample(min(a.per_class, len(g)), random_state=42) for _, g in ext.groupby("dx")])
out = pd.DataFrame({"image_id": sub.image_id, "image_path": sub.local, "label": sub.dx, "mask_path": ""})
out.to_csv(os.path.expanduser("~/Desktop/dermvlm/data/external_subset.csv"), index=False)
print("mapped:", len(ext), "| missing:", int(miss), "| subset:", len(out), out.label.value_counts().to_dict())
