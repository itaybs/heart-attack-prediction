"""
Train & evaluate the Logistic Regression model from the command line.

    python train.py                      # uses data/heart_data.csv
    python train.py --data "<folder>"    # looks for 'Project Heart attack data' (.csv / .xlsx) in a folder
"""
import argparse
import json
from pathlib import Path

from src.heart_model import SAMPLE_PATIENT, find_data_file, predict_with_explanation, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart Attack Risk Prediction - Logistic Regression")
    parser.add_argument("--data", help="Folder containing 'Project Heart attack data' (.csv/.xlsx)")
    args = parser.parse_args()

    data_file = find_data_file(args.data)
    print(f"Dataset: {data_file}")
    bundle = train_model(data_file)
    m = bundle.metrics

    print(f"\nRows: {m['n_rows_raw']} raw -> {m['n_rows']} clean | missing values: {m['missing_values_raw']}"
          f" | duplicates: {m['duplicates_raw']}")
    print(f"Positive class (target=1) rate: {m['positive_rate']:.1%}")
    print(f"Train/Test split (stratified 80/20): {m['n_train']} / {m['n_test']}")
    print(f"Features after One-Hot encoding ({m['n_features']}): {bundle.feature_names}")

    print("\n=== Test-set metrics (threshold 0.5) ===")
    t, tr, b = m["test"], m["train"], m["baseline_majority"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        base = f"  | majority baseline {b[k]:.3f}" if k in b else ""
        print(f"{k:<10}: {t[k]:.3f}  (train {tr[k]:.3f}){base}")
    print(f"5-fold CV  : recall {m['cv_recall_mean']:.3f} ± {m['cv_recall_std']:.3f} | "
          f"F1 {m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}")
    print(f"\nConfusion matrix  [[TN FP] [FN TP]] = [[{t['tn']} {t['fp']}] [{t['fn']} {t['tp']}]]")

    print("\n=== Threshold trade-off ===")
    for row in m["thresholds"]:
        print(f"  th={row['threshold']:.1f}  recall={row['recall']:.3f}  precision={row['precision']:.3f}"
              f"  FN={row['fn']}  FP={row['fp']}")

    print("\n=== Coefficients (sorted by standardized importance) ===")
    print(bundle.coefficients[["feature", "coef_std", "odds_ratio_std", "odds_ratio_step", "step"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n=== Sample prediction ===")
    print(SAMPLE_PATIENT)
    r = predict_with_explanation(bundle, SAMPLE_PATIENT)
    print(f"Predicted class : {r['prediction']}  |  P(target=1) = {r['proba']:.1%}"
          f"  (average patient {r['baseline_proba']:.1%})")
    print("Log-odds contributions vs. average patient:")
    for feat, val in r["contributions"].items():
        print(f"  {feat:<10} {val:+.3f}")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    bundle.coefficients.to_csv(out / "coefficients.csv", index=False)
    print(f"\nSaved reports to {out.resolve()}")


if __name__ == "__main__":
    main()
