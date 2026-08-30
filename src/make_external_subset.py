"""Stratified subsample of the de-contaminated external ISIC split (cap per class). EDIT column names if needed."""
import argparse, os, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--ts", default=os.path.expanduser("~/Desktop/TrustSkin_R"))
ap.add_argument("--cap", type=int, default=300); ap.add_argument("--img_col", default="image_path")
ap.add_argument("--lbl_col", default="label"); ap.add_argument("--id_col", default="image_id")
a = ap.parse_args()
ext = pd.read_csv(f"{a.ts}/data/splits/external_isic.csv")
sub = pd.concat([g.sample(min(a.cap, len(g)), random_state=42) for _, g in ext.groupby(a.lbl_col)])
out = pd.DataFrame({"image_id": sub[a.id_col], "image_path": sub[a.img_col], "label": sub[a.lbl_col], "mask_path": ""})
out.to_csv("data/external_subset.csv", index=False)
print(out.label.value_counts().to_dict(), len(out))
