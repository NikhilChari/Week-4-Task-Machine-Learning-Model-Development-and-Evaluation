import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve)

SEED = 42
BASE = Path(__file__).resolve().parent          # folder containing this script
CSV = BASE / "breast_cancer_clean.csv"           # keep the CSV next to pipeline.py
FIG = BASE / "figures"                           # charts are written here
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv(CSV)
R = {}

# ---- 1. Data preparation
R["shape_raw"] = list(df.shape)
R["missing"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
R["class_counts"] = df["diagnosis"].value_counts().to_dict()
yj_cols = [c for c in df.columns if c.endswith("_yj")]
R["n_yj"] = len(yj_cols)
feat_cols = [c for c in df.columns if c != "diagnosis" and not c.endswith("_yj")]
X = df[feat_cols]
y = (df["diagnosis"] == "Malignant").astype(int)   # 1 = malignant (positive class)
R["n_features"] = len(feat_cols)

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
R["n_pairs_gt90"] = int((upper > 0.9).sum().sum())

# Fig 1: class balance + a few feature distributions
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
y.map({0: "Benign", 1: "Malignant"}).value_counts().reindex(["Benign", "Malignant"]).plot.bar(ax=ax[0], color=["#4C78A8", "#E45756"], rot=0)
ax[0].set_title("Class balance"); ax[0].set_ylabel("Samples")
for a, f in zip(ax[1:], ["radius_worst", "concave_points_worst"]):
    a.hist(X.loc[y == 0, f], bins=25, alpha=.7, label="Benign", color="#4C78A8")
    a.hist(X.loc[y == 1, f], bins=25, alpha=.7, label="Malignant", color="#E45756")
    a.set_title(f); a.legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG / "fig1_eda.png", dpi=170); plt.close()

# ---- 2. Split (stratified, hold-out test set untouched until the end)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["split"] = {"train": len(Xtr), "test": len(Xte), "train_malig": int(ytr.sum()), "test_malig": int(yte.sum())}

# ---- 3. Models, compared with 5-fold stratified CV on the training set only
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000, random_state=SEED))]),
    "Decision Tree": Pipeline([("m", DecisionTreeClassifier(random_state=SEED))]),
    "Random Forest": Pipeline([("m", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
R["cv"] = {}
for n, p in models.items():
    s = cross_validate(p, Xtr, ytr, cv=cv, scoring=scoring, return_train_score=True)
    R["cv"][n] = {k: [float(np.mean(s[f"test_{k}"])), float(np.std(s[f"test_{k}"]))] for k in scoring}
    R["cv"][n]["train_acc"] = float(np.mean(s["train_accuracy"]))

# ---- 4. Tune the chosen model (logistic regression) and the tree (to show overfitting control)
grid = GridSearchCV(models["Logistic Regression"], {"m__C": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="roc_auc")
grid.fit(Xtr, ytr)
R["best_C"] = grid.best_params_["m__C"]; R["best_cv_auc"] = float(grid.best_score_)
R["C_curve"] = {str(c): float(s) for c, s in zip([0.001, 0.01, 0.1, 1, 10, 100], grid.cv_results_["mean_test_score"])}
best_lr = grid.best_estimator_
tree_grid = GridSearchCV(models["Decision Tree"], {"m__max_depth": [2, 3, 4, 5, 6, None]}, cv=cv, scoring="roc_auc").fit(Xtr, ytr)
R["tree_best_depth"] = str(tree_grid.best_params_["m__max_depth"])
tuned = {"Logistic Regression": best_lr, "Decision Tree": tree_grid.best_estimator_, "Random Forest": models["Random Forest"].fit(Xtr, ytr)}

# ---- 5. Final test-set evaluation
R["test"] = {}; probs = {}
for n, m in tuned.items():
    pr = m.predict_proba(Xte)[:, 1]; pd_ = (pr >= .5).astype(int); probs[n] = pr
    tn, fp, fn, tp = confusion_matrix(yte, pd_).ravel()
    R["test"][n] = dict(accuracy=accuracy_score(yte, pd_), precision=precision_score(yte, pd_), recall=recall_score(yte, pd_),
                        f1=f1_score(yte, pd_), auc=roc_auc_score(yte, pr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                        train_acc=accuracy_score(ytr, m.predict(Xtr)))
R["test"] = {k: {a: (float(b) if not isinstance(b, int) else b) for a, b in v.items()} for k, v in R["test"].items()}

cols = {"Logistic Regression": "#E45756", "Decision Tree": "#4C78A8", "Random Forest": "#54A24B"}
# Fig 2: confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, (n, m) in zip(ax, tuned.items()):
    ConfusionMatrixDisplay.from_predictions(yte, (probs[n] >= .5).astype(int), display_labels=["Benign", "Malignant"], ax=a, cmap="Blues", colorbar=False)
    a.set_title(n); a.grid(False)
plt.tight_layout(); plt.savefig(FIG / "fig2_confusion.png", dpi=170); plt.close()

# Fig 3: ROC + PR curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for n, pr in probs.items():
    fpr, tpr, _ = roc_curve(yte, pr); ax[0].plot(fpr, tpr, color=cols[n], lw=2, label=f"{n} (AUC={roc_auc_score(yte, pr):.3f})")
    p_, r_, _ = precision_recall_curve(yte, pr); ax[1].plot(r_, p_, color=cols[n], lw=2, label=n)
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves"); ax[0].legend(frameon=False, loc="lower right")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves", ylim=(0.8, 1.01)); ax[1].legend(frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(FIG / "fig3_roc_pr.png", dpi=170); plt.close()

# Fig 4: metric comparison bars (test)
mets = ["accuracy", "precision", "recall", "f1", "auc"]
fig, ax = plt.subplots(figsize=(9, 3.8)); w = .26
for i, (n, v) in enumerate(R["test"].items()):
    ax.bar(np.arange(5) + i * w, [v[k] for k in mets], w, label=n, color=cols[n])
ax.set_xticks(np.arange(5) + w); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]); ax.set_ylim(.85, 1.01); ax.legend(frameon=False, ncol=3, loc="upper center")
ax.set_title("Test-set metrics by model"); plt.tight_layout(); plt.savefig(FIG / "fig4_metrics.png", dpi=170); plt.close()

# Fig 5: learning curve (over/underfitting diagnosis) for LR
sizes, trs, vas = learning_curve(best_lr, Xtr, ytr, cv=cv, train_sizes=np.linspace(.1, 1, 8), scoring="accuracy", random_state=SEED)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(sizes, trs.mean(1), "o-", color="#E45756", label="Training"); ax[0].plot(sizes, vas.mean(1), "o-", color="#4C78A8", label="Cross-validation")
ax[0].fill_between(sizes, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), alpha=.15, color="#4C78A8")
ax[0].set(xlabel="Training samples", ylabel="Accuracy", title="Learning curve: logistic regression"); ax[0].legend(frameon=False)
Cs = list(R["C_curve"].keys()); ax[1].semilogx([float(c) for c in Cs], list(R["C_curve"].values()), "o-", color="#4C78A8")
ax[1].set(xlabel="C (inverse regularisation strength)", ylabel="CV ROC AUC", title="Regularisation sweep")
plt.tight_layout(); plt.savefig(FIG / "fig5_learning.png", dpi=170); plt.close()
R["lc"] = {"train_final": float(trs.mean(1)[-1]), "val_final": float(vas.mean(1)[-1])}

# Fig 6: top coefficients
coef = pd.Series(best_lr.named_steps["m"].coef_[0], index=feat_cols)
top = coef.reindex(coef.abs().sort_values(ascending=False).index[:10])[::-1]
fig, ax = plt.subplots(figsize=(7, 3.8)); ax.barh(top.index, top.values, color=["#E45756" if v > 0 else "#4C78A8" for v in top.values])
ax.set_title("Top 10 standardised coefficients (red = pushes toward malignant)"); ax.set_xlabel("Coefficient"); plt.tight_layout(); plt.savefig(FIG / "fig6_coef.png", dpi=170); plt.close()
R["top_coef"] = {k: float(v) for k, v in top[::-1].items()}

# Threshold analysis on LR (CV-free, test set; illustrative)
R["threshold"] = {}
for t in [0.5, 0.3, 0.2]:
    p_ = (probs["Logistic Regression"] >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yte, p_).ravel()
    R["threshold"][str(t)] = dict(recall=float(recall_score(yte, p_)), precision=float(precision_score(yte, p_)), fn=int(fn), fp=int(fp))

# Bootstrap CI for LR test accuracy / recall
rng = np.random.default_rng(SEED); pl = (probs["Logistic Regression"] >= .5).astype(int); yt = yte.values; accs, recs = [], []
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt)); accs.append((pl[i] == yt[i]).mean())
    if yt[i].sum(): recs.append(recall_score(yt[i], pl[i]))
R["ci_acc"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
R["ci_rec"] = [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))]
json.dump(R, open(BASE / "results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
