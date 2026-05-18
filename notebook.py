import csv
import os
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

from pyspark import SparkConf, SparkContext

try:
    import matplotlib.pyplot as plt
    plt.ioff()
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DATA_PATH = "student_mental_health_burnout_1M.csv"
OUTPUT_PATH = "cleaned_student_data.csv"
TARGET_COLUMN = "risk_level"
NUMERIC_FEATURES = [
    "stress_level",
    "burnout_score",
    "depression_score",
    "anxiety_score",
    "exam_pressure",
    "family_expectation",
    "sleep_hours",
    "social_support",
    "academic_performance",
    "mental_health_index",
]
WEIGHTS = {
    "stress_level": 1.4,
    "burnout_score": 1.5,
    "depression_score": 1.2,
    "anxiety_score": 1.1,
    "exam_pressure": 1.0,
    "family_expectation": 0.7,
    "sleep_hours": -1.2,
    "social_support": -1.4,
    "academic_performance": -1.0,
    "mental_health_index": -1.6,
}


def parse_float(value: str):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def start_spark() -> SparkContext:
    conf = SparkConf()
    conf = conf.setAppName("StudentRiskRDD").setMaster("local[*]")
    conf = conf.set("spark.driver.memory", "4g").set("spark.executor.memory", "2g")
    return SparkContext.getOrCreate(conf=conf)


def parse_csv_line(line: str, header: List[str]) -> Dict[str, str]:
    parsed = next(csv.reader([line]))
    return {col: parsed[i] if i < len(parsed) else "" for i, col in enumerate(header)}


def load_rdd(sc: SparkContext, path: str):
    lines = sc.textFile(path).filter(lambda l: l.strip() != "")
    header_line = lines.first()
    header = next(csv.reader([header_line]))
    data_lines = lines.filter(lambda l: l != header_line).distinct()
    rows_rdd = data_lines.map(lambda line: parse_csv_line(line, header))
    return rows_rdd, header


def compute_column_stats(rows_rdd) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    def row_stats(row):
        for col in NUMERIC_FEATURES:
            value = parse_float(row.get(col, ""))
            if value is not None:
                yield (col, (value, 1, value, value))

    stats = rows_rdd.flatMap(row_stats).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1], min(a[2], b[2]), max(a[3], b[3]))
    ).collectAsMap()

    means = {col: stats[col][0] / stats[col][1] if col in stats and stats[col][1] > 0 else 0.0 for col in NUMERIC_FEATURES}
    mins = {col: stats[col][2] if col in stats else 0.0 for col in NUMERIC_FEATURES}
    maxs = {col: stats[col][3] if col in stats else 0.0 for col in NUMERIC_FEATURES}
    return means, mins, maxs


def fill_missing(row: Dict[str, str], means: Dict[str, float]) -> Dict[str, str]:
    clean_row = row.copy()
    for col in NUMERIC_FEATURES:
        if parse_float(clean_row.get(col, "")) is None:
            clean_row[col] = f"{means[col]:.4f}"
    return clean_row


def write_cleaned_csv_from_rdd(rows_rdd, header: List[str], path: str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()
            for row in rows_rdd.toLocalIterator():
                writer.writerow(row)
        print(f"Cleaned dataset saved to {path}")
    except PermissionError:
        fallback = os.path.splitext(path)[0] + "_new.csv"
        print(f"Permission denied writing {path}. Trying alternate path: {fallback}")
        with open(fallback, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()
            for row in rows_rdd.toLocalIterator():
                writer.writerow(row)
        print(f"Cleaned dataset saved to fallback path: {fallback}")


def normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def score_risk(row: Dict[str, str], mins: Dict[str, float], maxs: Dict[str, float]) -> float:
    score = 0.0
    for col, weight in WEIGHTS.items():
        value = parse_float(row.get(col, ""))
        if value is None:
            continue
        normalized = normalize(value, mins[col], maxs[col])
        score += weight * normalized
    return score


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(p * (len(sorted_values) - 1))
    return sorted_values[index]


def predict_label(score: float, cut1: float, cut2: float) -> str:
    if score <= cut1:
        return "Low"
    if score <= cut2:
        return "Medium"
    return "High"


def evaluate_predictions_rdd(predictions_rdd, total_count: int) -> None:
    correct = predictions_rdd.filter(lambda ap: ap[0] == ap[1]).count()
    accuracy = correct / total_count if total_count else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total_count})")

    classes = ["Low", "Medium", "High"]

    confusion = predictions_rdd.map(
        lambda ap: ((ap[0], ap[1]), 1)
    ).reduceByKey(lambda a, b: a + b).collectAsMap()

    print("\nConfusion Matrix:")
    print("\t" + "\t".join(classes))

    for actual in classes:
        row = "\t".join(
            str(confusion.get((actual, predicted), 0))
            for predicted in classes
        )
        print(f"{actual}\t{row}")

    classification_metrics(confusion, classes)


def classification_metrics(confusion, classes):
    print("\nClassification Report:")
    print("Class\tPrecision\tRecall\t\tF1-Score")

    for cls in classes:
        tp = confusion.get((cls, cls), 0)

        fp = sum(confusion.get((other, cls), 0)
                 for other in classes if other != cls)

        fn = sum(confusion.get((cls, other), 0)
                 for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        print(f"{cls}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")


def plot_cleaned_summary(scores: List[float], actual_distribution: Dict[str, int], cut1: float, cut2: float) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed; skipping plots. Install matplotlib to see graphs.")
        return

    labels = ["Low", "Medium", "High"]
    distribution = [actual_distribution.get(label, 0) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, distribution, color=["#4CAF50", "#FFC107", "#F44336"])
    axes[0].set_title("Actual risk level distribution")
    axes[0].set_xlabel("Risk Level")
    axes[0].set_ylabel("Count")

    axes[1].hist(scores, bins=30, color="#2196F3", edgecolor="black")
    axes[1].axvline(cut1, color="orange", linestyle="--", label="Low/Medium cutoff")
    axes[1].axvline(cut2, color="red", linestyle="--", label="Medium/High cutoff")
    axes[1].set_title("Risk score distribution")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    fig.suptitle("Cleaned Data Summary")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
    distribution,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90
    )

    ax.set_title("Risk Level Percentage Distribution")

    plt.show()
    plt.close(fig2)


def plot_feature_weights():
    if not HAS_MATPLOTLIB:
        return

    features = list(WEIGHTS.keys())
    weights = list(WEIGHTS.values())

    plt.figure(figsize=(10, 5))
    plt.bar(features, weights)

    plt.xticks(rotation=45)
    plt.ylabel("Weight")
    plt.title("Feature Importance in Risk Scoring")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap():
    if not HAS_MATPLOTLIB:
        return

    df = pd.read_csv(OUTPUT_PATH)

    corr = df[NUMERIC_FEATURES].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')

    plt.colorbar()

    plt.xticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES, rotation=45)
    plt.yticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES)

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Input file not found: {DATA_PATH}")
        return

    sc = start_spark()
    try:
        rows_rdd, header = load_rdd(sc, DATA_PATH)
        row_count = rows_rdd.count()
        print(f"Loaded {row_count} unique rows from {DATA_PATH}")

        means, mins, maxs = compute_column_stats(rows_rdd)
        cleaned_rdd = rows_rdd.map(lambda row: fill_missing(row, means)).cache()
        cleaned_count = cleaned_rdd.count()
        print(f"Cleaned and filled missing values for {cleaned_count} rows")

        write_cleaned_csv_from_rdd(cleaned_rdd, header, OUTPUT_PATH)

        scored_rdd = cleaned_rdd.map(lambda row: (row.get(TARGET_COLUMN, "").strip().title(), score_risk(row, mins, maxs)))
        sample_size = min(10000, cleaned_count)
        sampled_scores = scored_rdd.map(lambda ap: ap[1]).takeSample(False, sample_size, seed=42)
        if not sampled_scores:
            sampled_scores = scored_rdd.map(lambda ap: ap[1]).take(1000)

        cut1 = percentile(sampled_scores, 0.33)
        cut2 = percentile(sampled_scores, 0.66)

        predictions_rdd = scored_rdd.map(lambda ap: (ap[0], predict_label(ap[1], cut1, cut2)))
        actual_distribution = predictions_rdd.map(lambda ap: (ap[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()

        print(f"Dataset rows after cleaning: {cleaned_count}")
        print("Target distribution:", dict(actual_distribution))
        print(f"Score cutoffs: Low <= {cut1:.4f}, Medium <= {cut2:.4f}, High > {cut2:.4f}")

        plot_cleaned_summary(sampled_scores, actual_distribution, cut1, cut2)
        evaluate_predictions_rdd(predictions_rdd, cleaned_count)
        plot_feature_weights()
        plot_correlation_heatmap()

        print("\nThis notebook now uses a Spark RDD pipeline for data loading and cleaning.")
        print("The classifier remains a lightweight rule-based scoring model.")
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
    
