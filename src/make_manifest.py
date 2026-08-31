"""Build manifests from TrustSkin_R splits, mapping image_id -> local file (Kaggle paths in the CSV are ignored).
usage: python src/make_manifest.py --img_root ~/Desktop/data/ham10000 [--mask_dir /path/to/masks]"""
import argparse, os, glob, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=os.path.expanduser("~/Desktop/TrustSkin_R"))
ap.add_argument("--img_root", required=True); ap.add_argument("--mask_dir", default="")
a = ap.parse_args()
idx = {}
for p in glob.glob(os.path.join(os.path.expanduser(a.img_root), "**", "*.jpg"), recursive=True):
    idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
print("local images indexed:", len(idx))
for split in ["test", "val"]:
    df = pd.read_csv(f"{a.ts}/data/splits/{split}.csv")
    out = pd.DataFrame({"image_id": df.image_id, "image_path": df.image_id.map(idx), "label": df.dx})
    missing = out.image_path.isna().sum(); out = out.dropna(subset=["image_path"])
    out["mask_path"] = out.image_id.map(lambda i: os.path.join(a.mask_dir, f"{i}_segmentation.png")) if a.mask_dir else ""
    if a.mask_dir: out["mask_path"] = out.mask_path.where(out.mask_path.map(os.path.exists), "")
    out.to_csv(f"data/ham_{split}_manifest.csv", index=False)
    print(split, len(out), "missing:", missing, out.label.value_counts().to_dict(), "masks:", int((out.mask_path != "").sum()))
v = pd.read_csv("data/ham_val_manifest.csv")
pd.concat([g.sample(min(30, len(g)), random_state=42) for _, g in v.groupby("label")]).to_csv("data/ham_val_dev200.csv", index=False)
