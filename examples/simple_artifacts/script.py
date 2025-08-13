#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Global plot style
sns.set(style="whitegrid", context="notebook")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def parse_args():
    parser = argparse.ArgumentParser(description="Focused EDA and clustering analysis on a CSV with species labels.")
    parser.add_argument("--csv", "-c", type=str, default=None, help="Path to the CSV data file.")
    # Accept --data as an alias for --csv to avoid 'unrecognized arguments' errors in some runners
    parser.add_argument("--data", type=str, default=None, help="Alias for --csv: path to the CSV data file.")
    args = parser.parse_args()

    # Resolve csv path precedence: --data overrides --csv if both provided
    if args.data is not None:
        args.csv = args.data
    if args.csv is None:
        # Fallback default if neither provided
        args.csv = "data.csv"
    return args


def detect_label_column(df: pd.DataFrame) -> str:
    # Preferred names for the label column
    preferred = ['species', 'class', 'target', 'label', 'variety']
    for name in preferred:
        for col in df.columns:
            if col.strip().lower() == name:
                return col

    # Otherwise look for a categorical/text column with limited unique values
    cat_like = []
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            if 2 <= df[col].nunique(dropna=True) <= min(20, max(2, len(df) // 5)):
                cat_like.append(col)
    if len(cat_like) == 1:
        return cat_like[0]
    if len(cat_like) > 1:
        # pick the one with fewest unique values but > 1
        return sorted(cat_like, key=lambda c: df[c].nunique(dropna=True))[0]

    # Otherwise consider any low-cardinality integer-coded column
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            nun = df[col].nunique(dropna=True)
            if 2 <= nun <= min(20, max(2, len(df) // 5)):
                return col

    raise ValueError("Could not infer the species/label column. Please name it one of: species, class, target, label, variety.")


def sanitize_feature_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except Exception:
        # Fallback if tabulate is not available
        return df.to_string()


def save_hist_and_box(df, feature, label_col, outpath):
    plt.figure(figsize=(6, 4), dpi=72)
    gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    x = df[feature].dropna()
    ax1.hist(x, bins=15, color="#4c72b0", edgecolor="white", alpha=0.9)
    ax1.set_title(f"Histogram: {feature}")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")

    sns.boxplot(data=df, x=label_col, y=feature, ax=ax2, palette="Set2")
    ax2.set_title(f"By {label_col}")
    ax2.set_xlabel(label_col)
    ax2.set_ylabel("")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def save_corr_heatmap(df_num, outpath):
    corr = df_num.corr(method="pearson")
    plt.figure(figsize=(6, 4), dpi=72)
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, square=False, cbar_kws={"shrink": 0.7})
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    return corr


def save_scatter_pair(df, xcol, ycol, label_col, outpath):
    plt.figure(figsize=(6, 4), dpi=72)
    sns.scatterplot(data=df, x=xcol, y=ycol, hue=label_col, palette="Set2", s=25, edgecolor="none", alpha=0.9)
    plt.title(f"Best 2D Separation: {xcol} vs {ycol}")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def save_pca_scatter(pca_df, label_col, outpath):
    plt.figure(figsize=(6, 4), dpi=72)
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=label_col, palette="Set2", s=25, edgecolor="none", alpha=0.9)
    plt.title("PCA: PC1 vs PC2")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def save_silhouette_over_k_plot(ks, sils, outpath):
    plt.figure(figsize=(6, 4), dpi=72)
    sns.lineplot(x=ks, y=sils, marker="o", color="#4c72b0")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette score")
    plt.title("Silhouette score vs k")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def save_cluster_vs_species_heatmap(df, cluster_col, label_col, outpath):
    ctab = pd.crosstab(df[label_col], df[cluster_col])
    plt.figure(figsize=(6, 4), dpi=72)
    sns.heatmap(ctab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Cluster vs Species (Counts)")
    plt.xlabel("Cluster")
    plt.ylabel(label_col)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def compute_best_pair_logreg_accuracy(df_num, y, feature_names):
    # Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_pair = None
    best_acc = -np.inf
    for xcol, ycol in itertools.combinations(feature_names, 2):
        X_pair = df_num[[xcol, ycol]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pair)
        clf = LogisticRegression(max_iter=2000, multi_class='auto', solver='lbfgs', random_state=RANDOM_STATE)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
            mean_acc = float(np.mean(scores))
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_pair = (xcol, ycol)
        except Exception:
            # Skip pairs that fail due to class imbalance across folds or other issues
            continue
    return best_pair, best_acc


def pairwise_class_mahalanobis(df_num, y):
    # Compute pooled covariance and its inverse
    X = df_num.values
    classes = np.unique(y)
    if len(classes) < 2:
        return {}
    cov = np.cov(X, rowvar=False)
    # regularize for stability
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    # Compute means per class
    y_series = pd.Series(y, index=df_num.index)
    means = {c: df_num[y_series == c].mean().values for c in classes}
    distances = {}
    for c1, c2 in itertools.combinations(classes, 2):
        diff = means[c1] - means[c2]
        d2 = float(np.sqrt(diff.T @ inv_cov @ diff))
        distances[(c1, c2)] = d2
    return distances


def main():
    args = parse_args()
    csv_path = args.csv

    if not os.path.exists(csv_path):
        print(f"# Error\nCSV file not found at path: {csv_path}\nPlease provide a valid path with --csv or --data.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)

    # Detect label column
    try:
        label_col = detect_label_column(df)
    except Exception as e:
        print(f"# Error\n{e}")
        sys.exit(1)

    # Prepare numeric features
    numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        print("# Error\nNeed at least two numeric feature columns besides the label for this analysis.")
        sys.exit(1)

    # Drop rows with missing values in numeric or label
    n_rows_initial = len(df)
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=numeric_cols + [label_col])
    n_dropped = n_rows_initial - len(df_clean)
    df_clean[label_col] = df_clean[label_col].astype("category")
    classes = df_clean[label_col].cat.categories.tolist()

    # Convert labels to strings for downstream sklearn compatibility
    y_labels = df_clean[label_col].astype(str).values

    # Data readiness and basic stats
    print("# Focused EDA and Clustering Analysis")
    print("")
    print("## Data Readiness Check")
    print(f"- CSV path: {csv_path}")
    print(f"- Rows (original): {n_rows_initial}; Rows used after dropping missing in numeric/label: {len(df_clean)}; Dropped: {n_dropped}")
    dtypes_str = ", ".join([f"{c}:{str(df[c].dtype)}" for c in df.columns])
    print(f"- Column dtypes: {dtypes_str}")
    na_counts = df[numeric_cols + [label_col]].isna().sum().to_dict()
    print(f"- Missing values (numeric + label): {na_counts}")
    print(f"- Label column detected: {label_col}; Classes: {classes}")

    # Basic stats
    desc = df_clean[numeric_cols].describe().T
    print("\n### Summary statistics (numeric features)")
    print(df_to_markdown_safe(desc))

    # Group-wise centers/spread
    print("\n### Species-level centers and spread (means and std)")
    group_means = df_clean.groupby(label_col)[numeric_cols].mean()
    group_stds = df_clean.groupby(label_col)[numeric_cols].std()
    print("Means by species:")
    print(df_to_markdown_safe(group_means))
    print("\nStandard deviations by species:")
    print(df_to_markdown_safe(group_stds))

    # Univariate plots per feature (overall and by species)
    print("\n## Univariate Distributions")
    for feat in numeric_cols:
        fname = f"univariate_{sanitize_feature_name(feat)}.png"
        save_hist_and_box(df_clean, feat, label_col, fname)
        print(f"![Univariate {feat}]({fname})")
        print(f"- Takeaway: The histogram shows the overall distribution of {feat}, while the boxplot reveals species-level differences in its center and spread.")

    # Pairwise relationships: correlations
    print("\n## Pairwise Relationships")
    corr_png = "correlation_heatmap.png"
    corr = save_corr_heatmap(df_clean[numeric_cols], corr_png)
    print(f"![Correlation Heatmap]({corr_png})")
    print("- Takeaway: The heatmap highlights the strength and direction of linear relationships among numeric features.")

    # Strongest pairwise Pearson correlation
    corr_vals = corr.copy()
    np.fill_diagonal(corr_vals.values, np.nan)
    abs_corr = corr_vals.abs()
    strongest = abs_corr.unstack().dropna().sort_values(ascending=False).index[0]
    # strongest is a tuple (row_feature, col_feature)
    r_val = corr.loc[strongest[0], strongest[1]]
    print(f"\n- Strongest pairwise linear relationship (Pearson r): {strongest[0]} vs {strongest[1]} with r = {r_val:.3f}")

    # Best 2D separation pair via CV logistic regression accuracy
    best_pair, best_pair_acc = compute_best_pair_logreg_accuracy(df_clean[numeric_cols], y_labels, numeric_cols)
    if best_pair is not None:
        scatter_png = f"best2D_{sanitize_feature_name(best_pair[0])}_vs_{sanitize_feature_name(best_pair[1])}.png"
        save_scatter_pair(df_clean, best_pair[0], best_pair[1], label_col, scatter_png)
        print(f"- Best 2D feature pair for species separation (via 5-fold CV logistic accuracy): {best_pair[0]} + {best_pair[1]} with mean accuracy = {best_pair_acc:.3f}")
        print(f"![Best 2D Separation]({scatter_png})")
        print("- Takeaway: This 2D scatter plot illustrates the feature pair that most cleanly separates species in a plane.")
    else:
        print("- Best 2D feature pair for species separation could not be determined due to model or data constraints.")

    # Single most discriminative individual feature via one-way ANOVA
    print("\n## Discriminative Power (Univariate)")
    anova_results = {}
    for feat in numeric_cols:
        groups = [df_clean[df_clean[label_col] == cls][feat].values for cls in classes]
        # Filter groups with at least 2 observations; if any class has <2, ANOVA may still run but we guard
        try:
            f_stat, p_val = f_oneway(*groups)
        except Exception:
            p_val = 1.0
        anova_results[feat] = p_val
    best_feat = min(anova_results, key=anova_results.get)
    best_p = anova_results[best_feat]
    print(f"- Most discriminative individual feature across species (one-way ANOVA): {best_feat} with p-value = {best_p:.3e}")

    # Species most/least separable via pairwise Mahalanobis distance of class means (on standardized numeric features)
    scaler_for_dist = StandardScaler()
    X_std = scaler_for_dist.fit_transform(df_clean[numeric_cols].values)
    df_std = pd.DataFrame(X_std, columns=numeric_cols, index=df_clean.index)
    distances = pairwise_class_mahalanobis(df_std, y_labels)
    if distances:
        most_sep = max(distances, key=distances.get)
        least_sep = min(distances, key=distances.get)
        print(f"- Most separable species pair (Mahalanobis distance of means): {most_sep[0]} vs {most_sep[1]} (distance = {distances[most_sep]:.2f})")
        print(f"- Least separable species pair (Mahalanobis distance of means): {least_sep[0]} vs {least_sep[1]} (distance = {distances[least_sep]:.2f})")

    # Dimensionality reduction: PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[numeric_cols].values)
    pca = PCA(n_components=min(5, len(numeric_cols)), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    cum_var_2 = float(np.sum(var_ratio[:2]))
    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], label_col: df_clean[label_col].values})
    pca_png = "pca_scatter_pc1_pc2.png"
    save_pca_scatter(pca_df, label_col, pca_png)
    print("\n## Dimensionality Reduction (PCA)")
    print(f"- Cumulative variance explained by the first two PCs: {cum_var_2*100:.2f}%")
    print(f"![PCA Scatter]({pca_png})")
    print("- Takeaway: The PCA scatter plot shows whether species are visually separated along the directions of maximum variance.")

    # Clustering: determine optimal k via silhouette score
    print("\n## Clustering Analysis")
    ks = list(range(2, min(8, len(df_clean) - 1) + 1))
    sil_scores = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_k = kmeans.fit_predict(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sil = silhouette_score(X, labels_k, metric="euclidean")
        sil_scores.append(sil)
    best_idx = int(np.argmax(sil_scores))
    best_k = ks[best_idx]
    best_sil = sil_scores[best_idx]
    sil_png = "silhouette_vs_k.png"
    save_silhouette_over_k_plot(ks, sil_scores, sil_png)
    print(f"- Optimal number of clusters based on average silhouette score: k = {best_k} (silhouette = {best_sil:.3f})")
    print(f"![Silhouette vs k]({sil_png})")
    print("- Takeaway: The curve indicates the k that best balances cohesion and separation according to silhouette score.")

    # Fit final KMeans and evaluate ARI to true labels
    final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X)
    ari = adjusted_rand_score(y_labels, cluster_labels)
    print(f"- Adjusted Rand Index (ARI) between KMeans(k={best_k}) and true species: {ari:.3f}")

    # Cluster vs species heatmap
    df_out = df_clean.copy()
    df_out["cluster"] = cluster_labels
    ctab_png = "cluster_vs_species_heatmap.png"
    save_cluster_vs_species_heatmap(df_out, "cluster", label_col, ctab_png)
    print(f"![Cluster vs Species Heatmap]({ctab_png})")
    print("- Takeaway: The heatmap shows how well clusters align with known species labels (perfect alignment would concentrate counts along one column per species).")

    # Features driving the clusters: inspect standardized cluster centers
    centers = final_kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=numeric_cols)
    # Feature importance proxy: range across cluster centers
    feature_ranges = centers_df.max(axis=0) - centers_df.min(axis=0)
    top_driver_features = feature_ranges.sort_values(ascending=False).head(min(3, len(feature_ranges)))
    top_driver_list = ", ".join([f"{feat} (Δ={delta:.2f} z)" for feat, delta in top_driver_features.items()])
    print(f"- Features most driving clusters (by center range in standardized units): {top_driver_list}")

    # Overlaps/outliers: silhouette per sample for final k
    sil_values = silhouette_samples(X, cluster_labels, metric="euclidean")
    low_sil_count = int(np.sum(sil_values < 0.0))
    low_sil_pct = 100.0 * low_sil_count / len(sil_values)
    print(f"- Overlap/outliers: {low_sil_count} samples ({low_sil_pct:.1f}%) have negative silhouette values, indicating potential overlap or misassignment.")

    # Conclusions
    print("\n## Conclusions")
    print("- Species separability: The most/least separable species pairs are reported above based on Mahalanobis distances of class means.")
    print(f"- Key drivers: The most discriminative single feature is {best_feat} (ANOVA p = {best_p:.3e}); top cluster-driving features are {top_driver_list}.")
    print(f"- Pairwise linearity: Strongest Pearson correlation is {strongest[0]} vs {strongest[1]} with r = {r_val:.3f}.")
    if best_pair is not None:
        print(f"- Best 2D separation: {best_pair[0]} + {best_pair[1]} achieved mean CV accuracy = {best_pair_acc:.3f} for species classification.")
    else:
        print("- Best 2D separation: No robust 2D feature pair identified due to data/model constraints.")
    print(f"- Dimensionality: First two PCs explain {cum_var_2*100:.2f}% of variance and the PCA scatter indicates how well species separate in this space.")
    print(f"- Optimal k: k = {best_k} by silhouette ({best_sil:.3f}); cluster-to-species alignment ARI = {ari:.3f}.")
    print(f"- Overlaps/anomalies: {low_sil_count} items with negative silhouette suggest areas of cluster overlap or ambiguous species boundaries.")

if __name__ == "__main__":
    main()