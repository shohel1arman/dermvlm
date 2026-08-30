"""Build manifests from TrustSkin_R splits. EDIT the three column names if your CSV headers differ.
usage: python src/make_manifest.py [--mask_dir /path/to/tschandl_masks]"""
import argparse, os, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--ts", default=os.path.expanduser("~/Desktop/TrustSkin_R"))
ap.add_argument("--mask_dir", default=""); ap.add_argument("--img_col", default="image_path")
ap.add_argument("--lbl_col", default="label"); ap.add_argument("--id_col", default="image_id")
a = ap.parse_args()
for split in ["test", "val"]:
    df = pd.read_csv(f"{a.ts}/data/splits/{split}.csv")
    out = pd.DataFrame({"image_id": df[a.id_col], "image_path": df[a.img_col], "label": df[a.lbl_col]})
    out["mask_path"] = out.image_id.map(lambda i: os.path.join(a.mask_dir, f"{i}_segmentation.png")) if a.mask_dir else ""
    if a.mask_dir:
        out["mask_path"] = out.mask_path.where(out.mask_path.map(os.path.exists), "")
    out.to_csv(f"data/ham_{split}_manifest.csv", index=False)
    print(split, out.shape, out.label.value_counts().to_dict(), "masks:", (out.mask_path != "").sum())
# small val subset for prompt development only
pd.read_csv("data/ham_val_manifest.csv").groupby("label", group_keys=False).apply(lambda g: g.sample(min(30, len(g)), random_state=42)).to_csv("data/ham_val_dev200.csv", index=False)
