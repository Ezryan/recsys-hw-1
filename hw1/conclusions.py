"""Generate evidence-linked conclusions from the frozen experiment tables."""
from pathlib import Path
from textwrap import dedent


def build_conclusions(final_metrics, search_trials, paired_test, search_selected,
                      fm_search_epochs, final_run, test_report, split_summary, config):
    # Числа в выводах берутся из результатов текущего воспроизводимого прогона.
    conclusion_models = final_metrics.set_index("model")
    conclusion_primary = conclusion_models.loc[conclusion_models.role.eq("primary")]
    conclusion_trials = search_trials.set_index("trial_id")
    conclusion_pairs = paired_test.set_index(["comparison", "metric"])
    conclusion_best_fm = search_selected["fm"]
    conclusion_fm_curve = fm_search_epochs.loc[fm_search_epochs.trial_id.eq(conclusion_best_fm["trial_id"])]
    conclusion_fm_selected = int(conclusion_best_fm["params"]["epochs"])
    conclusion_fm_last = int(conclusion_fm_curve.epoch.max())

    def cq(value, digits=5):
        return f"{float(value):.{digits}f}".replace(".", ",")

    def cn(value):
        return f"{int(value):,}".replace(",", " ")

    def cm(model, column):
        return float(conclusion_models.loc[model, column])

    def cv(trial):
        return float(conclusion_trials.loc[trial, "validation_ndcg_at_10"])

    def cd(comparison, metric):
        return float(conclusion_pairs.loc[(comparison, metric), "delta"])

    def cci(comparison, metric):
        row = conclusion_pairs.loc[(comparison, metric)]
        return f"[{cq(row.ci_lower, 6)}; {cq(row.ci_upper, 6)}]"

    conclusion_leader = conclusion_primary.test_ndcg_at_10.idxmax()
    conclusion_recall_leader = conclusion_primary.test_recall_at_10.idxmax()
    assert conclusion_leader == conclusion_recall_leader == "ease"
    assert not conclusion_pairs.loc["custom_vs_binary", "ci_excludes_zero"].any()
    assert conclusion_pairs.loc["fm_metadata_vs_ids", "ci_excludes_zero"].all()
    assert conclusion_fm_selected == 8 and conclusion_fm_last == 11
    assert final_run["model_details"]["fm"]["training_details"]["examples_sha256"] == final_run["model_details"]["fm_ids_only"]["training_details"]["examples_sha256"]

    conclusion_text = f"""
    ### Качество моделей

    Среди пяти основных методов EASE получил на test наибольшие NDCG@10 = **{cq(cm('ease', 'test_ndcg_at_10'))}** и Recall@10 = **{cq(cm('ease', 'test_recall_at_10'))}**. Разница NDCG с iALS — {cq(cm('ease', 'test_ndcg_at_10')-cm('ials', 'test_ndcg_at_10'), 6)}; парный интервал для этой пары не считался. Контроль **только с давностью** дал {cq(cm('custom_ials_recency_only', 'test_ndcg_at_10'))} / {cq(cm('custom_ials_recency_only', 'test_recall_at_10'))}, но он не участвовал в выборе основной пятой модели. [Таблица](results/final_metrics.csv), [график](results/figures/12_final_test_metrics.png).

    ### Выбор параметров

    На validation выбраны kNN с k=200 и shrinkage=10 (NDCG {cq(cv('item_knn_08'))}), iALS с rank=128 и 20 итерациями ({cq(cv('ials_04'))}), EASE с lambda=100 ({cq(cv('ease_03'))}), FM с rank=16 и четырьмя отрицательными примерами ({cq(cv('fm_09'))}) и custom iALS с gamma=1, h=180 дней ({cq(cv('custom_ials_05'))}). Для FM лучшая эпоха — {conclusion_fm_selected}; обучение остановилось на {conclusion_fm_last}-й, затем лучшие веса восстановлены.

    Всего проверены 12 kNN, 12 iALS, 4 EASE, 12 FM и 6 custom-конфигураций. Бюджеты поиска различались, поэтому сравнение ограничено выбранными попытками. На validation лидировал custom iALS, на test — EASE; test для повторного подбора не использовался. [Журнал](results/validation_trials.csv), [эпохи FM](results/fm_validation_epochs.csv), [выбранные настройки](results/best_params.json).

    ### Исторические признаки FM

    При одинаковых примерах, seed и числе эпох FM с историческими признаками улучшила NDCG@10 относительно версии только с ID на **{cq(cd('fm_metadata_vs_ids', 'ndcg@10'), 6)}** (95% CI {cci('fm_metadata_vs_ids', 'ndcg@10')}), Recall@10 — на **{cq(cd('fm_metadata_vs_ids', 'recall@10'), 6)}** (CI {cci('fm_metadata_vs_ids', 'recall@10')}). Оба интервала исключают ноль. Это эффект внутри выбранной FM на данном test; по абсолютному качеству она уступила другим основным методам. [Парные сравнения](results/paired_test_comparisons.csv).

    ### Новый таргет iALS

    Разность custom iALS и бинарного iALS: NDCG@10 **{cq(cd('custom_vs_binary', 'ndcg@10'), 6)}**, CI {cci('custom_vs_binary', 'ndcg@10')}; Recall@10 **{cq(cd('custom_vs_binary', 'recall@10'), 6)}**, CI {cci('custom_vs_binary', 'recall@10')}. Оба интервала включают ноль: улучшение не подтвердилось. В контролях одна сила оценки дала {cq(cd('rating_only_vs_binary', 'ndcg@10'), 6)} NDCG, одна давность — {cq(cd('recency_only_vs_binary', 'ndcg@10'), 6)}. Сочетание уступило одной давности на {cq(-cd('combined_vs_recency', 'ndcg@10'), 6)} NDCG, но его CI также включает ноль. Во всех вариантах одинаковы структурные параметры и confidence; меняется только таргет. [Контроли](results/figures/13_final_test_controls.png), [интервалы](results/paired_test_comparisons.csv).

    ### Время и память

    Для {cn(test_report['n_evaluation_users'])} test-пользователей EASE обучался {cq(cm('ease', 'fit_seconds'), 2)} с, выдавал top-10 за {cq(cm('ease', 'inference_seconds'), 2)} с и использовал до {cq(cm('ease', 'peak_worker_rss_bytes')/2**30, 3)} GiB. Item-kNN обучался {cq(cm('item_knn', 'fit_seconds'), 2)} с при пике {cq(cm('item_knn', 'peak_worker_rss_bytes')/2**30, 3)} GiB. Это измерения одного CPU-прогона; пик включает данные и библиотеки, а не только веса. [Полная таблица](results/final_metrics.csv).

    ### Ограничения

    5-core отбирает активных пользователей и товары; отзывы не равны прослушиваниям, а отсутствие отзыва не означает отрицательный интерес. Семплированные отрицательные пары FM могут быть будущими положительными. Из-за холодного старта исключены {cn(test_report['cold_start']['positive_events_with_either'])} из {cn(test_report['held_out_positive_events'])} положительных test-событий ({cq(test_report['cold_start']['fraction_positive_events_with_either']*100, 2)}%). Результаты относятся к одному временному разбиению и seed=42; bootstrap пользователей не учитывает повторное обучение и не скорректирован за множественные сравнения. [Покрытие test](results/evaluation_test_summary.json), [EDA](results/eda_summary.json).
    """
    return dedent(conclusion_text).lstrip()


def save_conclusions(results_dir, text):
    """Keep links correct both in the root notebook and the results artifact."""
    path = Path(results_dir) / "conclusions.md"
    artifact = "# Выводы по эксперименту\n\n" + text.replace("](results/", "](").replace("### ", "## ")
    path.write_text(artifact, encoding="utf-8")
    return path
