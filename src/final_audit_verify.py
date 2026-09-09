"""Read-only model audit; writes only dedicated audit evidence and tables.

Does not train a new candidate, rerun forests/embeddings, or overwrite experiments.
"""
from pathlib import Path
import hashlib
import json
import platform
import zipfile

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from organiser_baseline import prototype_predict
from transfer_ablation_experiment import define_groups, group_shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BUDGETS = [5, 25, 50, 100, 200]


def main():
    files = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*.json")
             if p.name != "final_audit_verification.json"}
    checked, errors = [], []

    def walk(value, path):
        if isinstance(value, dict):
            if "scores" in value and isinstance(value["scores"], list):
                a = np.asarray(value["scores"], dtype=float)
                mk = "mean" if "mean" in value else "macro_f1_mean"
                sk = "std" if "std" in value else "macro_f1_std_population"
                if mk in value and sk in value:
                    delta = max(abs(a.mean() - value[mk]), abs(a.std() - value[sk]))
                    checked.append({"path": path, "n": len(a), "max_error": float(delta)})
                    if delta > 1e-12: errors.append(path)
            for k, v in value.items(): walk(v, path + "/" + k)
    for name, data in files.items(): walk(data, name)
    with np.load(OUT / "features_60.npz", allow_pickle=False) as z:
        f = {k: z[k] for k in z.files}
    with np.load(OUT / "amsterdam_fixed_splits.npz", allow_pickle=False) as z:
        splits = {k: z[k] for k in z.files}
    y, x = f["amsterdam_y"], f["amsterdam_x"]
    rng = np.random.default_rng(42)
    split_checks, raw_reproduction, paired = [], {}, {}
    metric = files["metric_learning_results.json"]
    for n in BUDGETS:
        ps, ls = [], []
        for trial in range(10):
            s, q = splits[f"support_{n}_{trial}"], splits[f"query_{n}_{trial}"]
            generated = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False)
                                        for c in [1, 2, 3, 4]])
            assert np.array_equal(np.sort(generated), s)
            assert len(s) == 4*n and len(q) == len(y)-4*n
            assert not np.intersect1d(s, q).size
            assert np.array_equal(np.sort(np.concatenate([s, q])), np.arange(len(y)))
            assert all((y[s] == c).sum() == n for c in [1, 2, 3, 4])
            split_checks.append([n, trial])
            pp = prototype_predict(x[s], y[s], x[q])
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(x[s], y[s])
            ps.append(float(f1_score(y[q], pp, labels=[1,2,3,4], average="macro")))
            ls.append(float(f1_score(y[q], lr.predict(x[q]), labels=[1,2,3,4], average="macro")))
        raw_reproduction[str(n)] = {}
        for key, vals in [("prototype", ps), ("logistic_regression", ls)]:
            saved = metric["raw_features"][str(n)][key]["scores"]
            err = float(np.max(np.abs(np.array(vals)-saved)))
            raw_reproduction[str(n)][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "max_saved_score_error": err}
            if err > 1e-10: errors.append(f"raw reproduction {n} {key}")
        diff = np.array(ls)-ps
        paired[str(n)] = {"logistic_minus_prototype_mean": float(diff.mean()),
                          "sd_of_paired_differences": float(diff.std()),
                          "logistic_wins_of_10": int((diff>0).sum())}
        print(f"Verified raw methods and ten split sets at {n}/class", flush=True)
    groups, _ = define_groups(f["feature_names"].tolist())
    shifts = group_shift(f["madrid_x"], x, groups)
    a = files["feature_transfer_results.json"]
    shift_error = max(abs(shifts[k]["mean_absolute_smd"]-a["shift_measure"]["groups"][k]["mean_absolute_smd"]) for k in groups)
    assert shift_error < 1e-10
    c = np.asarray(a["strongest_25"]["confusion_counts_across_trials"])
    class_metrics = {}
    for i in range(4):
        class_metrics[str(i+1)] = {"f1": float(2*c[i,i]/(c[i].sum()+c[:,i].sum())),
                                   "recall": float(c[i,i]/c[i].sum())}
        assert abs(class_metrics[str(i+1)]["f1"]-a["strongest_25"]["per_class_aggregate"][str(i+1)]["f1"])<1e-12
    paths = list((ROOT / "src").glob("*.py")) + list((ROOT / "Docs").glob("*"))
    paths += [OUT / name for name in files]
    paths += [OUT / "features_60.npz", OUT / "amsterdam_fixed_splits.npz"]
    paths += [ROOT / "Data" / n for n in ["madrid_train.parquet", "amsterdam_data.parquet"]]
    def sha(p):
        h=hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
        return h.hexdigest()
    with zipfile.ZipFile(ROOT / "Data_Science_Hackathon.zip") as z:
        archive = z.namelist()
        organiser_copy_matches = {}
        for p in (ROOT / "Docs").iterdir():
            matches=[n for n in archive if n.replace("\\", "/").split("/")[-1] == p.name]
            for n in matches: organiser_copy_matches[n] = hashlib.sha256(z.read(n)).hexdigest()==sha(p)
    report = {
        "scope": "Arithmetic, cached-feature raw-method reproduction, split integrity, shift and confusion verification; no new candidate training",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "summary_vectors_checked": checked, "errors": errors,
        "split_metadata": json.loads(str(splits["metadata"])),
        "split_sets_verified": len(split_checks),
        "pulled_sweep_rng_matches_fixed_index_sets_given_identical_row_order": True,
        "pulled_cache_row_order_verified": False,
        "missing_artifacts": [p for p in ["outputs/stage1_model.pkl", "outputs/features_cache.pkl", "src/rf_leaf_transfer.py", "src/hybrid_transfer.py", "src/spatial_leakage_eval.py"] if not (ROOT/p).exists()],
        "raw_reproduction": raw_reproduction, "paired_raw_differences": paired,
        "recomputed_shift": shifts, "max_shift_error": shift_error,
        "all60_logistic_25_aggregate_class_metrics": class_metrics,
        "aggregate_macro_f1": float(np.mean([v["f1"] for v in class_metrics.values()])),
        "pixel_key_unique": {city: len(np.unique(f[city+"_pixel_keys"], axis=0))==len(f[city+"_y"]) for city in ["madrid", "amsterdam"]},
        "all_feature_values_finite": bool(np.isfinite(x).all() and np.isfinite(f["madrid_x"]).all()),
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()},
        "archive_members": [n for n in archive if not n.startswith('.venv/')],
        "archive_contains_venv": any(n.startswith('.venv/') for n in archive),
        "archive_organiser_copy_matches": organiser_copy_matches,
    }
    (OUT / "final_audit_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Verified numerical appendix", "", "Generated by `src/final_audit_verify.py`. All ± values are population SD; these are internal development results. See FINAL_CRITICAL_AUDIT.md for interpretation.", ""]
    def table(title, rows):
        lines.extend(["## "+title,"","| Method | 5/class | 25/class | 50/class | 100/class | 200/class |","|---|---:|---:|---:|---:|---:|"])
        for label, vals in rows:
            cells=[]
            for n in BUDGETS:
                v=vals[str(n)]; cells.append(f"{v.get('macro_f1_mean',v.get('mean')):.4f} ± {v.get('macro_f1_std_population',v.get('std')):.4f}")
            lines.append("| "+label+" | "+" | ".join(cells)+" |")
        lines.append("")
    rows=[]
    for label, data in [("Raw",metric["raw_features"]),("Random-negative embedding",metric["embedding"]),("Hard-negative embedding",files["metric_learning_results_hard_negative_8.json"]["embedding"])]:
        for method in ["prototype","logistic_regression"]: rows.append((label+" "+method,{n:v[method] for n,v in data.items()}))
    table("Shared fixed splits",rows)
    table("Feature subset results",[(rep+" "+method,{n:v[method] for n,v in data.items()}) for rep,data in a["ablations"].items() for method in ["prototype","logistic_regression"]])
    table("Support standardisation",[(m,{n:v[m] for n,v in a["correction"]["results"].items()}) for m in ["prototype","logistic_regression"]])
    table("Pulled results with incomplete provenance",[("Two-tier RF transfer",files["final_pipeline_sweep.json"]["adaptation_sweep"]),("RF/prototype score product",files["bayes_combine_transfer_results.json"]["bayes_combined"]),("Madrid LDA prototype",files["lda_embedding_transfer_results.json"]["lda_embedding_prototype"])])
    (ROOT / "Docs" / "VERIFIED_RESULT_TABLES.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"vectors_checked":len(checked), "errors":errors, "shift_error":shift_error, "paired":paired, "class_metrics":class_metrics, "missing":report["missing_artifacts"]},indent=2),flush=True)
    assert not errors


if __name__ == "__main__":
    main()
