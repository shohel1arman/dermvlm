"""Publication figures for DermVLM. Reads results/*.csv + outputs/*.jsonl, writes fig/*.pdf and *.png (300 dpi)."""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"font.family":"DejaVu Sans"})
OUT="fig"; os.makedirs(OUT,exist_ok=True)
CB={"Qwen3.5-0.8B-4bit":"#E69F00","Qwen3.5-2B-4bit":"#0072B2","gemma-4-E2B-it-4bit":"#009E73"}
SHORT={"Qwen3.5-0.8B-4bit":"Qwen3.5-0.8B","Qwen3.5-2B-4bit":"Qwen3.5-2B","gemma-4-E2B-it-4bit":"Gemma4-E2B"}
def save(fig,name):
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/{name}.{ext}",bbox_inches="tight")
    plt.close(fig); print("wrote",name)
pr=pd.read_csv("results/per_run.csv"); pr["model"]=pr["run"].map(lambda r:r.split("__")[0])
def e2e(df):
    m=df["run"].str.contains("ham_test") & df["run"].str.count("__").eq(2) & df["run"].str.contains("__A_")
    return df[m].copy()
def load_run(fn):
    p=f"outputs/{fn}"; return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

# FIG 1 framework
fig,ax=plt.subplots(figsize=(7.2,3.4)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
def box(x,y,w,h,t,fc): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",fc=fc,ec="#333",lw=1.1)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9)
def arr(a,b,c,d): ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle="-|>",mutation_scale=12,lw=1.1,color="#333"))
box(0.2,2.1,1.5,0.9,"Dermoscopic\nimage","#f0f0f0"); box(2.2,3.4,2.3,0.9,"VLM\n(end-to-end, A)","#cde7f7")
box(2.2,0.7,2.3,0.9,"VLM perceiver\n(features only, B1)","#cdeee0"); box(5.2,0.7,2.3,0.9,"SLM/VLM reasoner\n(B2 / control)","#cdeee0")
box(8.0,2.1,1.7,0.9,"Diagnosis +\nconfidence","#f7e0cd")
arr(1.7,2.55,2.2,3.7); arr(1.7,2.55,2.2,1.15); arr(4.5,3.85,8.0,2.75); arr(4.5,1.15,5.2,1.15); arr(7.5,1.15,8.0,2.35)
ax.text(6.35,0.35,"same model on diagonal = decoupling control",ha="center",fontsize=7.5,style="italic",color="#555")
ax.text(3.35,4.5,"Condition A: perceive+reason jointly",ha="center",fontsize=8,color="#0072B2")
ax.text(3.85,2.05,"Condition B: perceive -> reason (decomposed)",ha="center",fontsize=8,color="#009E73")
save(fig,"fig1_framework")

# FIG 2 decoupling
d=e2e(pr).dropna(subset=["bal_acc"])
fig,axs=plt.subplots(1,3,figsize=(9.5,3.2))
for ax,(k,lab) in zip(axs,[("ece","ECE (down=better)"),("aurc","AURC (down=better)"),("mel_recall","Melanoma recall (up=better)")]):
    for _,r in d.iterrows(): ax.scatter(r["bal_acc"],r[k],s=48,color=CB[r["model"]],edgecolor="k",lw=.4,zorder=3)
    ax.set_xlabel("Balanced accuracy"); ax.set_ylabel(lab); ax.grid(alpha=.25,zorder=0)
axs[1].legend(handles=[plt.Line2D([0],[0],marker="o",ls="",mfc=CB[m],mec="k",label=SHORT[m]) for m in CB],
    loc="upper center",bbox_to_anchor=(0.5,1.22),ncol=3,frameon=False)
fig.suptitle("Accuracy does not predict trust (each dot = one model x prompt, HAM test)",y=1.06,fontsize=10)
save(fig,"fig2_decoupling")

# FIG 3 reliability
def relia(ax,recs,color,label):
    rs=[r for r in recs if r.get("diagnosis") in ["akiec","bcc","bkl","df","mel","nv","vasc"] and r.get("confidence") is not None]
    if not rs: return
    p=np.array([r["confidence"]/100 for r in rs]); c=np.array([float(r["diagnosis"]==r["label"]) for r in rs])
    e=np.linspace(0,1,11); xs=[]; ys=[]
    for lo,hi in zip(e[:-1],e[1:]):
        m=(p>lo)&(p<=hi) if lo>0 else (p>=lo)&(p<=hi)
        if m.sum()>0: xs.append(p[m].mean()); ys.append(c[m].mean())
    ax.plot([0,1],[0,1],"--",color="#999",lw=1); ax.plot(xs,ys,"o-",color=color,label=label,ms=4)
fig,axs=plt.subplots(1,3,figsize=(9.5,3.3),sharey=True)
for ax,m in zip(axs,CB):
    relia(ax,load_run(f"{m}__A_v1__ham_test_manifest.jsonl"),CB[m],"in-distribution")
    relia(ax,load_run(f"{m}__A_v1__external_subset.jsonl"),"#333","external (ISIC)")
    ax.set_title(SHORT[m]); ax.set_xlabel("Stated confidence"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=.25)
axs[0].set_ylabel("Empirical accuracy"); axs[0].legend(loc="upper left",frameon=False)
fig.suptitle("Reliability of verbalized confidence: in-distribution vs external shift",y=1.03,fontsize=10)
save(fig,"fig3_reliability")

# FIG 4 risk-coverage
fig,ax=plt.subplots(figsize=(4.6,3.6))
for m in CB:
    rs=[r for r in load_run(f"{m}__A_v1__ham_test_manifest.jsonl") if r.get("diagnosis") in ["akiec","bcc","bkl","df","mel","nv","vasc"] and r.get("confidence") is not None]
    if not rs: continue
    p=np.array([r["confidence"]/100 for r in rs]); c=np.array([float(r["diagnosis"]==r["label"]) for r in rs])
    o=np.argsort(-p); err=1-c[o]; risk=np.cumsum(err)/np.arange(1,len(err)+1); cov=np.arange(1,len(err)+1)/len(err)
    ax.plot(cov,risk,color=CB[m],label=SHORT[m],lw=1.6)
ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error)"); ax.grid(alpha=.25); ax.legend(frameon=False); ax.set_title("Risk-coverage (HAM test, A_v1)")
save(fig,"fig4_risk_coverage")

# FIG 5 faithfulness
fa=pd.read_csv("results/faithfulness.csv"); fa=fa[fa.pid=="A_v1"].copy()
fig,ax=plt.subplots(figsize=(5.2,3.6)); x=np.arange(len(fa)); w=0.38
ax.bar(x-w/2,fa.flip_lesion,w,label="lesion masked",color="#0072B2"); ax.bar(x+w/2,fa.flip_control,w,label="control masked",color="#E69F00")
ax.set_xticks(x); ax.set_xticklabels([SHORT.get(s,s) for s in fa.system]); ax.set_ylabel("Decision-flip rate"); ax.set_ylim(0,1.08); ax.legend(frameon=False)
ax.set_title("Faithfulness: flips under lesion vs control masking")
for i,(_,r) in enumerate(fa.iterrows()): ax.text(i,max(r.flip_lesion,r.flip_control)+0.03,f"d={r.faithfulness_delta:.2f}",ha="center",fontsize=8)
save(fig,"fig5_faithfulness")

# FIG 6 perceive-reason matrices
grid=pr[pr.run.str.count("__").eq(4) & pr.run.str.contains("ham_test")].copy()
grid["perceiver"]=grid.run.map(lambda r:r.split("__")[0]); grid["reasoner"]=grid.run.map(lambda r:r.split("__")[2])
order=list(CB.keys())
def mat(metric):
    M=np.full((3,3),np.nan)
    for _,r in grid.iterrows():
        if r.perceiver in order and r.reasoner in order: M[order.index(r.perceiver),order.index(r.reasoner)]=r[metric]
    return M
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,(metric,title,cmap) in zip(axs,[("bal_acc","Balanced accuracy","viridis"),("ece","ECE (lower=better)","magma_r")]):
    M=mat(metric); im=ax.imshow(M,cmap=cmap,aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[o] for o in order],rotation=25,ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([SHORT[o] for o in order])
    ax.set_xlabel("Reasoner (B2)"); ax.set_ylabel("Perceiver (B1)"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i,j]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if ((metric=="bal_acc" and M[i,j]<0.2) or (metric=="ece" and M[i,j]>0.4)) else "k",fontsize=8)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    for k in range(3): ax.add_patch(plt.Rectangle((k-.5,k-.5),1,1,fill=False,ec="cyan",lw=1.6))
fig.suptitle("Perceive->reason grid (cyan = same-model decoupling control)",y=1.02,fontsize=10)
save(fig,"fig6_pr_matrix")

# FIG 7 perception
if os.path.exists("results/perception_derm7pt.csv"):
    pc=pd.read_csv("results/perception_derm7pt.csv")
    try:
        piv=pc.pivot(index="model",columns="criterion",values="macro_f1")
        fig,ax=plt.subplots(figsize=(7,2.6)); im=ax.imshow(piv.values,cmap="viridis",aspect="auto",vmin=0,vmax=1)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,rotation=30,ha="right")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([SHORT.get(i,i) for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]): ax.text(j,i,f"{piv.values[i,j]:.2f}",ha="center",va="center",fontsize=7,color="w" if piv.values[i,j]<0.4 else "k")
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); ax.set_title("Derm7pt perception: per-criterion macro-F1")
        save(fig,"fig7_perception")
    except Exception as e: print("fig7 skipped:",e)
else: print("fig7 skipped: no perception csv")

# FIG 8 profile + cost
prof=e2e(pr); prof=prof[prof.run.str.contains("__A_v1__ham_test")].set_index("model")
axes=["bal_acc","macro_f1","mel_recall","ece","aurc"]; disp={"bal_acc":"Bal.Acc","macro_f1":"MacroF1","mel_recall":"MelRec","ece":"ECE","aurc":"AURC"}
Z=prof.reindex(list(CB))[axes].astype(float); norm=Z.copy()
for k in axes:
    v=Z[k]; norm[k]=(v-v.min())/(v.max()-v.min()+1e-9)
    if k in ("ece","aurc"): norm[k]=1-norm[k]
fig,axs=plt.subplots(1,2,figsize=(9,3.0),gridspec_kw={"width_ratios":[2.3,1]})
im=axs[0].imshow(norm.values,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
axs[0].set_xticks(range(len(axes))); axs[0].set_xticklabels([disp[a] for a in axes])
axs[0].set_yticks(range(len(CB))); axs[0].set_yticklabels([SHORT[m] for m in CB])
for i,m in enumerate(CB):
    for j,k in enumerate(axes): axs[0].text(j,i,f"{Z.loc[m,k]:.2f}",ha="center",va="center",fontsize=8)
axs[0].set_title("Trust profile (A_v1, HAM test; green=better)")
axs[1].barh([SHORT[m] for m in CB],[prof.loc[m,"latency_median"] for m in CB],color=[CB[m] for m in CB])
axs[1].set_xlabel("Median s/image"); axs[1].set_title("On-device cost"); axs[1].invert_yaxis()
save(fig,"fig8_profile_cost")
print("ALL FIGURES DONE ->",OUT)
