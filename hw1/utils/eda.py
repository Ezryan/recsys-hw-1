"""Train-only descriptive analysis and plots; no model fitting or test labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.cleaning import clean_reviews
from utils.data import obtain_reviews
from utils.reproducibility import PROJECT_ROOT, load_config
from utils.splitting import temporal_split


def _metadata(
    history: pd.DataFrame, entity: str, cutoff: pd.Timestamp, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile strictly prior event features and the training-end snapshot.

    All interactions at one timestamp are aggregated before cumulative sums;
    subtracting the whole group prevents same-time labels from entering features.
    These descriptive frames do not implement FM encoding/normalization/sampling.
    """
    key = f"{entity}_id"
    work = history[[key, "timestamp", "rating"]].assign(
        positive=history["rating"].ge(config["evaluation"]["relevance_rating_threshold"]).astype(int)
    )
    blocks = work.groupby([key, "timestamp"], sort=True).agg(
        review_count=("rating", "size"), rating_sum=("rating", "sum"), positive_sum=("positive", "sum")
    )
    prior = blocks.groupby(level=0).cumsum() - blocks
    queries = pd.MultiIndex.from_frame(work[[key, "timestamp"]])
    prior = prior.reindex(queries)
    prior.index = history.index
    counts = prior["review_count"]
    denominator = counts.replace(0, np.nan)
    features = config["features"]
    events = pd.DataFrame({
        "log1p_review_count": np.log1p(counts),
        "mean_rating": (prior["rating_sum"] / denominator).fillna(features["empty_history_mean_rating"]),
        "positive_fraction": (prior["positive_sum"] / denominator).fillna(features["empty_history_positive_fraction"]),
        "empty_history": counts.eq(0).astype(int),
    }, index=history.index)
    totals = work.groupby(key).agg(
        review_count=("rating", "size"), mean_rating=("rating", "mean"),
        positive_fraction=("positive", "mean"), first_seen=("timestamp", "min")
    )
    snapshot = totals.assign(log1p_review_count=np.log1p(totals["review_count"]), empty_history=0)
    if entity == "item":
        first_seen = work[key].map(totals["first_seen"])
        age_days = (work["timestamp"] - first_seen).dt.total_seconds() / 86400
        events["log1p_observed_age_days"] = np.log1p(age_days.where(counts.gt(0), 0))
        snapshot["observed_age_days"] = (cutoff - totals["first_seen"]).dt.total_seconds() / 86400
        snapshot["log1p_observed_age_days"] = np.log1p(snapshot["observed_age_days"])
    return events, snapshot


def build_eda_profile(train: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Compute training statistics; reject events at/after the fixed cutoff."""
    cutoff = pd.Timestamp(config["split"]["validation_start_utc"])
    if cutoff.tzinfo is None or not isinstance(train["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("EDA requires timezone-aware timestamps and a fixed cutoff")
    if train.empty or train["timestamp"].isna().any() or not train["timestamp"].lt(cutoff).all():
        raise ValueError("EDA accepts only nonempty training history strictly before validation")
    if train.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Sparsity requires cleaned, unique user-item interactions")
    threshold = config["evaluation"]["relevance_rating_threshold"]
    positive = train.loc[train["rating"].ge(threshold)]
    user_counts = train.groupby("user_id").size()
    item_counts = train.groupby("item_id").size()
    user_positive = positive.groupby("user_id").size().reindex(user_counts.index, fill_value=0)
    item_positive = positive.groupby("item_id").size().reindex(item_counts.index, fill_value=0)
    cells = len(user_counts) * len(item_counts)
    sparsity = pd.DataFrame([
        {"signal": signal, "n_users": len(user_counts), "n_items": len(item_counts),
         "possible_pairs": cells, "nnz": count, "sparsity": 1 - count / cells}
        for signal, count in [("all_reviews", len(train)), ("positive_reviews", len(positive))]
    ])
    activity = pd.DataFrame({
        "users_all_reviews": user_counts.describe(percentiles=[.25, .5, .75, .9, .95, .99]),
        "items_all_reviews": item_counts.describe(percentiles=[.25, .5, .75, .9, .95, .99]),
        "users_positive_reviews": user_positive.describe(percentiles=[.25, .5, .75, .9, .95, .99]),
        "items_positive_reviews": item_positive.describe(percentiles=[.25, .5, .75, .9, .95, .99]),
    }).rename_axis("statistic").reset_index()
    short = pd.DataFrame([
        {"max_reviews": limit, "n_users": int(user_counts.le(limit).sum()),
         "fraction_users": float(user_counts.le(limit).mean())} for limit in [1, 3, 5]
    ])
    missing = pd.DataFrame([
        {"field": name, "missing_rows": int(train[name].isna().sum()),
         "missing_fraction": float(train[name].isna().mean())} for name in train.columns
    ])
    ratings = train["rating"].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
    rating_table = pd.DataFrame({"rating": ratings.index, "n_reviews": ratings.to_numpy(),
                                 "fraction": ratings.to_numpy() / len(train)})
    monthly = train.resample("MS", on="timestamp").size().rename("n_reviews").rename_axis("month_utc").reset_index()
    events, snapshots = {}, {}
    feature_rows = []
    for entity in ("user", "item"):
        events[entity], snapshots[entity] = _metadata(train, entity, cutoff, config)
        for context, values in (("before_training_event", events[entity]), ("at_validation_cutoff", snapshots[entity])):
            for feature in config["features"][entity]:
                column = values[feature]
                feature_rows.append({
                    "entity": entity, "context": context, "feature": feature,
                    "n_values": len(column), "missing_values": int(column.isna().sum()),
                    "zero_values": int(column.eq(0).sum()), "mean": float(column.mean()),
                    "min": float(column.min()), "median": float(column.median()),
                    "p95": float(column.quantile(.95)), "p99": float(column.quantile(.99)), "max": float(column.max()),
                })
    top_count = max(1, math.ceil(len(item_counts) * .01))
    empty_user, empty_item = events["user"]["empty_history"].eq(1), events["item"]["empty_history"].eq(1)
    summary = {
        "analyzed_period": "train", "cutoff_utc": cutoff.isoformat(),
        "n_interactions": len(train), "n_users": len(user_counts), "n_items": len(item_counts),
        "n_positive_interactions": len(positive), "positive_fraction": len(positive) / len(train),
        "user_history_median": float(user_counts.median()), "user_history_max": int(user_counts.max()),
        "item_history_median": float(item_counts.median()), "item_history_max": int(item_counts.max()),
        "fraction_users_with_at_most_3_reviews": float(user_counts.le(3).mean()),
        "n_users_without_positive_history": int(user_positive.eq(0).sum()),
        "fraction_users_without_positive_history": float(user_positive.eq(0).mean()),
        "top_1pct_items_count": top_count,
        "top_1pct_items_interaction_fraction": float(item_counts.nlargest(top_count).sum() / len(train)),
        "sparsity_all_reviews": 1 - len(train) / cells,
        "sparsity_positive_reviews": 1 - len(positive) / cells,
        "sparsity_universe": "same_all_train_users_and_items_for_both_signals",
        "training_events_without_prior_user_history": int(empty_user.sum()),
        "training_events_without_prior_item_history": int(empty_item.sum()),
        "training_events_without_either_history": int((empty_user & empty_item).sum()),
        "fraction_events_without_prior_user_history": float(empty_user.mean()),
        "fraction_events_without_prior_item_history": float(empty_item.mean()),
        "fraction_events_at_utc_midnight": float(train["timestamp"].eq(train["timestamp"].dt.floor("D")).mean()),
        "metadata_missing_values_after_defaults": int(sum(frame.isna().sum().sum() for frame in events.values())),
        "training_feature_policy": "strictly_before_timestamp_excluding_entire_current_time_group",
        "snapshot_feature_policy": "train_only_at_validation_cutoff",
        "test_rating_statistics_included": False,
    }
    return {
        "summary": summary,
        "tables": {"activity_quantiles": activity, "short_histories": short, "missing_values": missing,
                   "ratings": rating_table, "monthly_activity": monthly, "sparsity": sparsity,
                   "metadata_summary": pd.DataFrame(feature_rows)},
        "vectors": {"user_counts": user_counts, "item_counts": item_counts,
                    "user_positive_counts": user_positive, "item_positive_counts": item_positive},
        "event_metadata": events, "snapshot_metadata": snapshots,
    }


def save_eda_tables(profile: dict[str, Any], config: dict[str, Any]) -> Path:
    """Publish aggregates only; entity IDs and per-event features stay in memory."""
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    (results / "eda_summary.json").write_text(
        json.dumps(profile["summary"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, table in profile["tables"].items():
        table.to_csv(results / f"eda_{name}.csv", index=False)
    return results


def plot_eda(profile: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Create compact standalone PNGs and return closed figures for the notebook."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    blue, orange, teal = "#4058AD", "#D87B35", "#16877C"

    def keep(name, fig):
        fig.savefig(output / f"{name}.png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[name] = fig

    vectors, tables, events = profile["vectors"], profile["tables"], profile["event_metadata"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .18, "figure.constrained_layout.use": True}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title, unit in zip(axes, ("user_counts", "item_counts"),
                                        ("Активность пользователей", "Популярность объектов"),
                                        ("Пользователей", "Объектов")):
            values = vectors[key]
            bins = np.geomspace(1, values.max() + 1, 30)
            ax.hist(values, bins=bins, color=blue, alpha=.85, edgecolor="white")
            ax.set(xscale="log", yscale="log", xlabel="Отзывов на train", ylabel=unit, title=title)
            ax.axvline(values.median(), color=orange, linestyle="--", label=f"Медиана: {values.median():.0f}")
            ax.legend()
        keep("01_activity_popularity", fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key, label, color in (("item_counts", "Все отзывы", blue), ("item_positive_counts", "Положительные", teal)):
            counts = vectors[key].sort_values(ascending=False)
            counts = counts[counts > 0].to_numpy()
            axes[0].loglog(np.arange(1, len(counts) + 1), counts, color=color, label=label)
        axes[0].set(xlabel="Ранг объекта (порядок отдельно для каждого сигнала)", ylabel="Отзывов", title="Длинный хвост популярности на train")
        axes[0].legend()
        counts = vectors["item_counts"].sort_values(ascending=False).to_numpy()
        axes[1].plot(np.arange(1, len(counts) + 1) / len(counts), counts.cumsum() / counts.sum(), color=blue)
        axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", label="Равномерная популярность")
        axes[1].set(xlabel="Доля самых популярных объектов", ylabel="Доля отзывов", title="Концентрация взаимодействий")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend()
        keep("02_popularity_concentration", fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rating_table = tables["ratings"]
        axes[0].bar(rating_table["rating"], rating_table["n_reviews"], color=[orange] * 3 + [blue] * 2)
        for row in rating_table.itertuples():
            axes[0].text(row.rating, row.n_reviews, f"{row.fraction:.1%}", ha="center", va="bottom")
        axes[0].set(xlabel="Оценка", ylabel="Отзывов", title="Оценки на train (релевантность ≥ 4)", xticks=range(1, 6))
        axes[0].margins(y=.15)
        counts = vectors["user_positive_counts"]
        axes[1].hist(np.log1p(counts), bins=30, color=teal, edgecolor="white")
        ticks = [0, 1, 3, 5, 10, 30, 100, 500]
        axes[1].set(xticks=np.log1p(ticks), xticklabels=[str(tick) for tick in ticks],
                    xlabel="Положительных отзывов (ось log1p)", ylabel="Пользователей",
                    title="Положительная история, включая нулевую")
        keep("03_ratings_positive_history", fig)

        fig, ax = plt.subplots(figsize=(11, 3.5))
        monthly = tables["monthly_activity"]
        ax.plot(monthly["month_utc"], monthly["n_reviews"], color=blue, linewidth=1.5)
        cutoff = pd.Timestamp(profile["summary"]["cutoff_utc"])
        ax.axvline(cutoff, color=orange, linestyle="--", label="Граница train / validation")
        ax.set(xlabel="Месяц UTC (последний месяц неполный)", ylabel="Отзывов на train", title="Временная динамика train")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend()
        keep("04_train_monthly_activity", fig)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, entity, label, color in ((axes[0, 0], "user", "пользователя", blue), (axes[0, 1], "item", "объекта", teal)):
            ax.hist(events[entity]["mean_rating"], bins=np.linspace(1, 5, 33), color=color)
            ax.set(xlabel="Средняя оценка в прошлом", ylabel="Обучающих событий", title=f"Средняя оценка {label}")
        for entity, label, color in (("user", "Пользователь", blue), ("item", "Объект", teal)):
            axes[0, 2].hist(events[entity]["positive_fraction"], bins=np.linspace(0, 1, 26), histtype="step", linewidth=1.5, color=color, label=label)
            axes[1, 0].hist(events[entity]["log1p_review_count"], bins=30, histtype="step", linewidth=1.5, color=color, label=label)
        axes[0, 2].set(xlabel="Доля положительных в прошлом", ylabel="Обучающих событий", title="Сила положительного сигнала")
        axes[0, 2].legend()
        axes[1, 0].set(xlabel="log1p(число прошлых отзывов)", ylabel="Обучающих событий", title="Активность после log1p")
        axes[1, 0].legend()
        log_age = events["item"]["log1p_observed_age_days"]
        axes[1, 1].hist(np.expm1(log_age), bins=30, color=teal)
        axes[1, 1].set(xlabel="Дни от первого прошлого отзыва", ylabel="Обучающих событий", title="Наблюдаемый возраст объекта")
        axes[1, 2].hist(log_age, bins=30, color=teal)
        axes[1, 2].set(xlabel="log1p(возраст в днях)", ylabel="Обучающих событий", title="Возраст после log1p")
        fig.suptitle("Метапризнаки строго до timestamp; при пустой истории используются заданные константы", fontsize=11)
        keep("05_historical_metadata", fig)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    raw, _ = obtain_reviews(config)
    cleaned, _ = clean_reviews(raw, config)
    periods, _ = temporal_split(cleaned, config)
    profile = build_eda_profile(periods["train"], config)
    results = save_eda_tables(profile, config)
    plot_eda(profile, results / "figures")
    print(json.dumps(profile["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
