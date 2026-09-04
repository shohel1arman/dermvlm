"""Compute the trust profile from outputs/*.jsonl -> results/*.csv
Axes: accuracy (bal-acc, macro-F1, MCC, mel recall), calibration (ECE/Brier/NLL on verbalized conf),
selective prediction (AURC, risk@80% coverage), abstention, paraphrase consistency, faithfulness, cost.
Bootstrap 95% CIs over images (n=1000, seed 42)."""
import glob, json, os, sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CLASSES
B = int(os.environ.get("BOOT", 1000)); rng = np.random.default_rng(42)

def load_all():
    recs = []
    for p in glob.glob("outputs/*.jsonl"):
        for l in open(p):
            if l.strip():
                d = json.loads(l); d["file"] = os.path.basename(p); recs.append(d)
    df = pd.DataFrame(recs)
    parts = df.file.str.replace(".jsonl", "", regex=False).str.split("__")
    df["run"] = df.file.str.replace(".jsonl", "", regex=False)
    df["stage"] = parts.map(lambda x: "B2" if len(x) == 5 else ("B1" if x[1].startswith("B1") else "A"))
    df["mani"] = parts.map(lambda x: x[-1])
    df["pid"] = parts.map(lambda x: x[-2] if len(x) == 5 else x[1])
    df["system"] = parts.map(lambda x: f"{x[0]}->{x[2]}" if len(x) == 5 else x[0])
    df["valid"] = df.diagnosis.isin(CLASSES)
    df["abstain"] = (df.diagnosis == "uncertain") | df.diagnosis.isna()
    df["correct"] = (df.diagnosis == df.label).astype(float)
    df["p"] = df.confidence / 100.0
    return df

def ece(p, c, bins=15):
    p, c = np.asarray(p), np.asarray(c); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(c[m].mean() - p[m].mean())
    return e

def aurc(p, c):
    """Area under risk-coverage curve, ordering by descending confidence."""
    o = np.argsort(-np.asarray(p)); err = 1 - np.asarray(c)[o]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return risk.mean()

def risk_at(p, c, cov=0.8):
    o = np.argsort(-np.asarray(p)); k = max(1, int(round(cov * len(o))))
    return 1 - np.asarray(c)[o][:k].mean()

def metrics(g):
    v = g[g.valid]
    out = dict(n=len(g), parse_fail_rate=g.parse_fail.mean(), abstain_rate=g.abstain.mean(), coverage=len(v) / len(g))
    if len(v) < 5: return out
    y, yh = v.label, v.diagnosis
    out.update(bal_acc=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro", labels=CLASSES, zero_division=0),
               mcc=matthews_corrcoef(y, yh), mel_recall=recall_score(y == "mel", yh == "mel", zero_division=0),
               acc_on_covered=v.correct.mean())
    vc = v.dropna(subset=["p"])
    if len(vc) >= 5:
        eps = 1e-6; pc = vc.p.clip(eps, 1 - eps)
        out.update(mean_conf=vc.p.mean(), ece=ece(vc.p, vc.correct), brier=((vc.p - vc.correct) ** 2).mean(),
                   nll=-(vc.correct * np.log(pc) + (1 - vc.correct) * np.log(1 - pc)).mean(),
                   aurc=aurc(vc.p, vc.correct), risk_at_80cov=risk_at(vc.p, vc.correct, 0.8),
                   overconf_gap=vc.p.mean() - vc.correct.mean())
    out.update(latency_median=g.latency_s.median(), latency_p95=g.latency_s.quantile(0.95))
    return out

def boot_ci(g, key):
    vals = []
    idx = np.arange(len(g))
    for _ in range(B):
        s = g.iloc[rng.choice(idx, len(idx))]
        m = metrics(s); vals.append(m.get(key, np.nan))
    return np.nanpercentile(vals, [2.5, 97.5])

def main():
    os.makedirs("results", exist_ok=True)
    df = load_all(); print("records:", len(df), "runs:", df.run.nunique())
    main_df = df[~df.mani.str.startswith("cf_")]
    rows = []
    for run, g in main_df.groupby("run"):
        m = metrics(g); m.update(run=run, system=g.system.iloc[0], stage=g.stage.iloc[0], pid=g.pid.iloc[0], mani=g.mani.iloc[0])
        for k in ["macro_f1", "ece", "aurc", "mel_recall"]:
            if k in m: lo, hi = boot_ci(g, k); m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
        rows.append(m)
    per_run = pd.DataFrame(rows); per_run.to_csv("results/per_run.csv", index=False)
    # mean +- sd across paraphrases (the study's "seeds")
    num = per_run.select_dtypes("number").columns.difference(["n"])
    agg = per_run.groupby(["system", "stage", "mani"])[list(num)].agg(["mean", "std"]).round(4)
    agg.to_csv("results/per_system_meansd.csv")
    # paraphrase consistency: unanimous-label rate + mean pairwise agreement per system/stage/manifest
    cons = []
    for (sy, st, ma), g in main_df.groupby(["system", "stage", "mani"]):
        piv = g.pivot_table(index="image_id", columns="pid", values="diagnosis", aggfunc="first")
        if piv.shape[1] < 2: continue
        pairs = [(piv[a] == piv[b]).mean() for a, b in itertools.combinations(piv.columns, 2)]
        cons.append(dict(system=sy, stage=st, mani=ma, n_prompts=piv.shape[1], unanimous_rate=(piv.nunique(axis=1) == 1).mean(), mean_pairwise_agreement=np.mean(pairs)))
    pd.DataFrame(cons).to_csv("results/consistency.csv", index=False)
    # faithfulness: lesion vs control vs background on cf_manifest, paired with the original prediction
    cf = df[df.mani.str.startswith("cf_")].copy()
    if len(cf):
        cf["orig_id"] = cf.image_id.str.split("__").str[0]; cf["variant"] = cf.image_id.str.split("__").str[1]
        base = main_df[main_df.mani.str.startswith("ham_test")][["image_id", "system", "pid", "diagnosis", "p"]].rename(columns={"image_id": "orig_id", "diagnosis": "dx0", "p": "p0"})
        j = cf.merge(base, on=["orig_id", "system", "pid"], how="inner")
        j["flip"] = (j.diagnosis != j.dx0).astype(float); j["dconf"] = j.p - j.p0
        fr = []
        for (sy, pid), g in j.groupby(["system", "pid"]):
            pv = g.pivot_table(index="orig_id", columns="variant", values=["flip", "dconf"], aggfunc="first").dropna()
            if len(pv) < 5 or ("flip", "lesion") not in pv or ("flip", "control") not in pv: continue
            d = pv[("flip", "lesion")] - pv[("flip", "control")]
            bs = [d.iloc[rng.choice(len(d), len(d))].mean() for _ in range(B)]
            def col(a, b): return pv[(a, b)].mean() if (a, b) in pv.columns else np.nan
            fr.append(dict(system=sy, pid=pid, n=len(pv), flip_lesion=col("flip","lesion"), flip_control=col("flip","control"),
                           flip_background=col("flip","background"),
                           faithfulness_delta=d.mean(), delta_lo=np.percentile(bs, 2.5), delta_hi=np.percentile(bs, 97.5),
                           dconf_lesion=col("dconf","lesion"), dconf_control=col("dconf","control")))
        pd.DataFrame(fr).to_csv("results/faithfulness.csv", index=False)
    print(per_run[["run", "n", "parse_fail_rate", "bal_acc", "macro_f1", "ece", "aurc", "mel_recall", "latency_median"]].round(3).to_string() if len(per_run) else "no main runs")

if __name__ == "__main__":
    main()
