"""Frozen refit/test experiment in isolated workers, with paired user bootstrap.

This orchestration lives outside models/utils so adding final reporting cannot
change the already frozen validation-search implementation identity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import sparse

from utils.cleaning import CORE_COLUMNS
from utils.evaluation import build_evaluation_stage, evaluate_recommendations, save_evaluation_stage
from utils.reproducibility import PROJECT_ROOT, configure_reproducibility, load_config, resolve_data_dir
from utils.search import (MODEL_CLASSES, canonical, fingerprint, generate_search_manifest, search_identity,
                          select_best, _file_sha, _read_verified, _write_verified, _save_json, _save_csv)
from utils.splitting import split_content_sha256, temporal_split

LOGGER = logging.getLogger(__name__)
METRICS = ('ndcg@10', 'recall@10')


def validate_frozen_selection(periods, config):
    """Validate all chosen settings and provenance without inspecting the test frame."""
    results = PROJECT_ROOT / config['storage']['results_dir']
    plan = json.loads((results / 'final_experiment_plan.json').read_text())
    best = json.loads((results / 'best_params.json').read_text())
    summary = json.loads((results / 'search_summary.json').read_text())
    frozen = json.loads((results / 'search_manifest.json').read_text())
    validation = build_evaluation_stage(periods, config, 'validation')
    manifest = generate_search_manifest(config)
    search_key, identity = search_identity(config, validation, manifest)
    if (summary['status'] != 'complete' or search_key != plan['search_key'] or search_key != summary['search_key']
            or frozen['plan_sha256'] != fingerprint(manifest) or frozen['code_sha256'] != identity['code_sha256']):
        raise ValueError('Validation selection or model/utils source differs from the frozen search')
    if (not plan['settings_fixed_before_test'] or plan['seed'] != config['seed']
            or plan['selection_metric'] != 'ndcg@10' or plan['evaluation_protocol'] != config['evaluation']
            or plan['refit_cutoff_utc'] != config['split']['test_start_utc']
            or plan['validation_stage_sha256'] != validation.report['stage_sha256']):
        raise ValueError('Final plan does not match the fixed evaluation protocol')
    directory = PROJECT_ROOT / config['storage']['cache_dir'] / 'validation_search' / search_key / 'trials'
    ledger = pd.read_csv(results / 'validation_trials.csv')
    expected_ids = {trial['trial_id'] for trial in manifest['trials']} | {'control_fm_ids_only', 'control_custom_recency_only'}
    if set(ledger['trial_id']) != expected_ids or len(ledger) != len(expected_ids) or not ledger['status'].eq('ok').all():
        raise ValueError('The validation ledger does not contain the complete declared search')
    records = {}
    for row in ledger.itertuples():
        report = _read_verified(directory / row.trial_id / 'report.json')
        request = {'trial_id': row.trial_id, 'model': report['model'], 'purpose': report['purpose'],
                   'params': report['requested_params']}
        if (report['request_sha256'] != fingerprint({'search_key': search_key, 'request': request})
                or canonical(json.loads(row.selected_params)) != canonical(report['params'])):
            raise ValueError('Validation request or ledger parameters differ from verified results')
        if report['status'] != 'ok' or report['validation']['stage_sha256'] != validation.report['stage_sha256']:
            raise ValueError('Validation result is not a completed common-stage experiment')
        for artifact in report['artifacts']:
            if _file_sha(directory / row.trial_id / artifact['name']) != artifact['sha256']:
                raise ValueError('Validation artifact checksum mismatch')
        records[row.trial_id] = report
    for model in MODEL_CLASSES:
        selected = select_best(list(records.values()), model)
        if (best[model]['trial_id'] != selected['trial_id'] or best[model]['params'] != selected['params']
                or best[model]['validation'] != selected['validation']):
            raise ValueError('Chosen parameters differ from the declared validation selection rule')
    if set(best) != set(MODEL_CLASSES):
        raise ValueError('Five primary configurations are required')
    primary = {name for name, entry in plan['models'].items() if entry['role'] == 'primary'}
    if primary != set(MODEL_CLASSES) or len(plan['models']) != 8:
        raise ValueError('Final plan must contain five primary models and three controls')
    for name, entry in plan['models'].items():
        reference = records[entry['validation_trial_id']]
        if entry['model'] != reference['model'] or canonical(entry['params']) != canonical(reference['params']):
            raise ValueError('A final experiment has parameters not fixed on validation')
        if name in primary and entry['params'] != best[name]['params']:
            raise ValueError('Primary final parameters differ from best_params')
    with_meta, ids = (plan['models'][plan['metadata_ablation'][key]] for key in ('with_metadata', 'IDs_only'))
    if with_meta['params'] != {**ids['params'], 'use_metadata': True} or ids['params']['use_metadata']:
        raise ValueError('FM metadata comparison must change only use_metadata')
    baseline = best['ials']['params']
    for name in plan['target_ablations'].values():
        for key, value in baseline.items():
            if plan['models'][name]['params'][key] != value:
                raise ValueError('Target ablations must retain the selected iALS structure')
    semantic_best = {name: {k: value[k] for k in ('params', 'validation', 'trial_id', 'seed', 'search_key')}
                     for name, value in best.items()}
    comparison = {
        'plan': plan, 'best': semantic_best, 'search_key': search_key,
        'model_utils_code_sha256': identity['code_sha256'], 'dataset_sha256': config['dataset']['sha256'],
        'cleaned_content_sha256': config['split']['boundary_source_content_sha256'],
        'split': config['split'], 'features': config['features'], 'seed': config['seed'],
        'bootstrap': {'seed': config['controls']['paired_bootstrap_seed'],
                      'resamples': config['controls']['paired_bootstrap_resamples'], 'confidence': .95,
                      'method': 'paired_percentile_bootstrap_common_users', 'intervals': 'pointwise'},
    }
    return plan, best, records, comparison


def freeze_test_comparison(comparison, results):
    """Commit the comparison specification before any access to real test labels."""
    path = results / 'test_access_manifest.json'
    digest = fingerprint(comparison)
    if path.exists():
        saved = _read_verified(path)
        if saved['comparison_sha256'] != digest or saved['comparison'] != comparison:
            raise ValueError('The comparison specification changed after test access was recorded')
        return saved
    saved = {'comparison_sha256': digest, 'comparison': comparison,
             'first_runner_code_sha256': _file_sha(Path(__file__)),
             'settings_committed_before_test_access': True, 'test_accessed': True}
    _write_verified(path, saved)
    return saved


def _peak_rss():
    path = Path('/proc/self/status')
    if not path.exists():
        raise RuntimeError('Isolated peak RSS measurement requires Linux /proc/self/status')
    return next(int(line.split()[1]) * 1024 for line in path.read_text().splitlines() if line.startswith('VmHWM:'))


def _save_model_artifacts(model, directory):
    if sparse.issparse(getattr(model, 'similarity_', None)):
        sparse.save_npz(directory / 'similarity.npz', model.similarity_)
    if hasattr(model, 'user_factors_'):
        np.save(directory / 'user_factors.npy', model.user_factors_)
        np.save(directory / 'item_factors.npy', model.item_factors_)
    if hasattr(model, 'save_artifacts'):
        model.save_artifacts(directory)
    if getattr(model, 'loss_history_', None):
        pd.DataFrame(model.loss_history_).to_csv(directory / 'loss.csv', index=False)


def fit_final_model(name, specification, history, stage, config, directory, request_sha256):
    """No validation callback/labels; refit all parameters from history before T2."""
    if stage.period != 'test' or stage.context.cutoff != pd.Timestamp(config['split']['test_start_utc']):
        raise ValueError('Final evaluation requires the T2 test stage')
    directory.mkdir(parents=True, exist_ok=True)
    model = MODEL_CLASSES[specification['model']](**specification['params'])
    actual_config = deepcopy(config)
    actual_config['runtime']['reuse_model_cache'] = False  # A missing final report triggers a fresh measured fit.
    baseline_rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    model.fit(history, actual_config, context=stage.context)
    fit_seconds = time.perf_counter() - start
    if canonical(model.params) != canonical(specification['params']):
        raise ValueError('Final refit changed the selected parameters')
    if getattr(model, 'validation_history_', []):
        raise ValueError('Final FM must train a fixed number of epochs without held-out monitoring')
    start = time.perf_counter()
    recommendations = model.recommend(stage=stage)
    inference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = evaluate_recommendations(recommendations, stage)
    metrics_seconds = time.perf_counter() - start
    np.save(directory / 'recommendations.npy', recommendations)
    result.per_user.to_csv(directory / 'per_user.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    _save_model_artifacts(model, directory)
    details = model.training_summary_
    parameter_bytes = next(details[key] for key in ('factor_storage_bytes', 'weights_storage_bytes', 'similarity_storage_bytes') if key in details)
    artifacts = [{'name': p.name, 'sha256': _file_sha(p)} for p in sorted(directory.iterdir())
                 if p.name not in ('report.json', 'job.json', 'worker.log') and not p.name.endswith('.part')]
    report = {
        'experiment': name, 'model_family': specification['model'], 'role': specification['role'],
        'params': model.params, 'seed': config['seed'], 'status': 'ok',
        'request_sha256': request_sha256, 'history_context_sha256': stage.context.fingerprint,
        'test': result.summary, 'training_details': details,
        'fit_seconds': fit_seconds, 'fit_body_seconds': model.fit_seconds_,
        'fit_compute_seconds': getattr(model, 'training_compute_seconds_', model.fit_seconds_),
        'inference_seconds': inference_seconds, 'metrics_seconds': metrics_seconds,
        'baseline_worker_rss_bytes': baseline_rss, 'peak_worker_rss_bytes': _peak_rss(),
        'parameter_storage_bytes': parameter_bytes, 'cpu_threads': config['runtime']['cpu_threads'],
        'model_fit_reused_cache': bool(getattr(model, 'cache_hit_', False)),
        'artifacts': artifacts, 'test_used': True,
    }
    _write_verified(directory / 'report.json', report)
    return report


def read_final_report(directory, request_hash, stage):
    report = _read_verified(directory / 'report.json')
    if (report['request_sha256'] != request_hash or report['status'] != 'ok'
            or report['test']['stage_sha256'] != stage.report['stage_sha256']):
        raise ValueError('Final result cache differs from the fixed request or test stage')
    for artifact in report['artifacts']:
        if _file_sha(directory / artifact['name']) != artifact['sha256']:
            raise ValueError('Final result artifact checksum mismatch; use --no-cache to recompute')
    per_user = pd.read_csv(directory / 'per_user.csv.gz')
    np.testing.assert_array_equal(per_user['user_index'].to_numpy(), stage.user_indices)
    recomputed = evaluate_recommendations(np.load(directory / 'recommendations.npy', allow_pickle=False), stage)
    for metric in METRICS:
        if (abs(report['test'][metric] - recomputed.summary[metric]) > 1e-14
                or not np.allclose(per_user[metric], recomputed.per_user[metric], atol=1e-14, rtol=0)):
            raise ValueError('Cached recommendations and per-user metrics disagree')
    return report, per_user


def paired_user_bootstrap(left, right, *, seed=42, resamples=2000, confidence=.95, batch_size=64):
    """Same user resamples for both metrics; positive delta means left is better."""
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError('Bootstrap resamples must be a positive integer')
    if not 0 < confidence < 1 or not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_)) or batch_size < 1:
        raise ValueError('Invalid bootstrap confidence or batch size')
    for table in (left, right):
        if set(('user_index', *METRICS)) - set(table.columns) or table.empty or not table.user_index.is_unique:
            raise ValueError('Bootstrap needs nonempty unique common-user rows and both metrics')
        values = table[list(METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Per-user metrics must be finite in [0,1]')
    if not np.array_equal(left.user_index.to_numpy(), right.user_index.to_numpy()):
        raise ValueError('Paired bootstrap requires identical common-user order')
    delta = left[list(METRICS)].to_numpy(dtype=float) - right[list(METRICS)].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    distribution = np.empty((resamples, len(METRICS)))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(len(delta), size=(stop-start, len(delta)))
        distribution[start:stop] = delta[indices].mean(axis=1)
    alpha = (1-confidence)/2
    intervals = np.quantile(distribution, [alpha, 1-alpha], axis=0)
    return pd.DataFrame([{
        'metric': metric, 'n_users': len(delta), 'delta': float(delta[:, j].mean()),
        'ci_lower': float(intervals[0, j]), 'ci_upper': float(intervals[1, j]),
        'confidence': confidence, 'resamples': int(resamples), 'seed': int(seed),
        'ci_excludes_zero': bool(intervals[0, j] > 0 or intervals[1, j] < 0),
    } for j, metric in enumerate(METRICS)])


def _final_table(reports, plan, validation_records, stage):
    rows = []
    for name, report in reports.items():
        reference = validation_records[plan['models'][name]['validation_trial_id']]
        rows.append({
            'model': name, 'model_family': report['model_family'], 'role': report['role'], 'status': report['status'],
            'selected_hyperparameters': canonical(report['params']), 'seed': report['seed'],
            'validation_ndcg_at_10': reference['validation']['ndcg@10'],
            'validation_recall_at_10': reference['validation']['recall@10'],
            'test_ndcg_at_10': report['test']['ndcg@10'], 'test_recall_at_10': report['test']['recall@10'],
            'fit_seconds': report['fit_seconds'], 'inference_seconds': report['inference_seconds'],
            'metrics_seconds': report['metrics_seconds'], 'fit_compute_seconds': report['fit_compute_seconds'],
            'peak_worker_rss_bytes': report['peak_worker_rss_bytes'],
            'baseline_worker_rss_bytes': report['baseline_worker_rss_bytes'],
            'parameter_storage_bytes': report['parameter_storage_bytes'],
            'result_cache_hit': report['cache_hit'], 'model_fit_reused_cache': report['model_fit_reused_cache'],
            'n_users': report['test']['n_users'], 'n_items': stage.report['n_catalog_items'],
            'n_relevant_pairs': report['test']['n_relevant_pairs'],
            'mean_candidates': stage.report['mean_candidates_per_evaluation_user'],
            'cold_positive_event_fraction': stage.report['cold_start']['fraction_positive_events_with_either'],
            'stage_sha256': report['test']['stage_sha256'],
        })
    return pd.DataFrame(rows)


def _bootstrap_comparisons(per_user, plan, config):
    target, meta = plan['target_ablations'], plan['metadata_ablation']
    pairs = [
        ('custom_vs_binary', 'custom_ials', 'ials'),
        ('fm_metadata_vs_ids', meta['with_metadata'], meta['IDs_only']),
        ('rating_only_vs_binary', target['rating_only'], target['binary']),
        ('recency_only_vs_binary', target['recency_only'], target['binary']),
        ('combined_vs_recency', target['rating_and_recency'], target['recency_only']),
        ('combined_vs_rating', target['rating_and_recency'], target['rating_only']),
    ]
    tables = []
    for comparison, left, right in pairs:
        table = paired_user_bootstrap(per_user[left], per_user[right],
                                      seed=config['controls']['paired_bootstrap_seed'],
                                      resamples=config['controls']['paired_bootstrap_resamples'])
        table.insert(0, 'right_model', right)
        table.insert(0, 'left_model', left)
        table.insert(0, 'comparison', comparison)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _run_isolated_worker(job, directory):
    _save_json(directory / 'job.json', job)
    with (directory / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory / 'job.json')],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONHASHSEED': str(job['config']['seed'])}, timeout=1800,
        )
    if process.returncode:
        raise RuntimeError(f"Final worker {job['name']} failed; see {directory / 'worker.log'}")


def run_final_evaluation(periods, config, *, reuse=True):
    """Commit frozen choices, then construct one real test stage and execute eight fits."""
    started = time.perf_counter()
    plan, best, validation_records, comparison = validate_frozen_selection(periods, config)
    bootstrap = comparison['bootstrap']
    if (isinstance(bootstrap['resamples'], bool) or not isinstance(bootstrap['resamples'], int) or bootstrap['resamples'] < 1
            or isinstance(bootstrap['seed'], bool) or not isinstance(bootstrap['seed'], int) or bootstrap['seed'] < 0):
        raise ValueError('Bootstrap seed/count must be declared nonnegative/positive integers')
    results = PROJECT_ROOT / config['storage']['results_dir']
    committed = freeze_test_comparison(comparison, results)  # No real test access occurred above.
    cleaned = pd.concat([periods[name][CORE_COLUMNS] for name in ('train', 'validation', 'test')], ignore_index=True)
    canonical_periods, _ = temporal_split(cleaned, config)
    stage = build_evaluation_stage(canonical_periods, config, 'test')
    if len(stage.user_indices) == 0:
        raise ValueError('The fixed test protocol has no evaluable users')
    save_evaluation_stage(stage, config)
    runner_sha = _file_sha(Path(__file__))
    identity = {'comparison_sha256': committed['comparison_sha256'], 'runner_code_sha256': runner_sha,
                'test_stage_sha256': stage.report['stage_sha256']}
    key = fingerprint(identity)
    root = PROJECT_ROOT / config['storage']['cache_dir'] / 'final_evaluation' / key
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / 'cleaned_interactions.csv.gz'
    cleaned.to_csv(data_path, index=False, compression={'method': 'gzip', 'mtime': 0})
    dataset_file_sha = _file_sha(data_path)
    reports, per_user = {}, {}
    def publish(status, error=None):
        if reports:
            _save_csv(results / 'final_metrics.csv', _final_table(reports, plan, validation_records, stage))
        summary = {'status': status, 'expected_experiments': len(plan['models']), 'completed_experiments': len(reports),
                   'result_cache_hits': sum(r['cache_hit'] for r in reports.values()),
                   'comparison_sha256': committed['comparison_sha256'], 'final_cache_key': key,
                   'runner_code_sha256': runner_sha, 'test_stage_sha256': stage.report['stage_sha256'],
                   'current_invocation_seconds': time.perf_counter() - started, 'test_used': True}
        if error is not None:
            summary['error'] = {'type': type(error).__name__, 'message': str(error)}
        _save_json(results / 'final_run_summary.json', summary)
        return summary
    publish('running')
    for name, specification in plan['models'].items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        request_hash = fingerprint({'identity': identity, 'experiment': name, 'specification': specification})
        LOGGER.info('Final experiment %s: %s', name, canonical(specification['params']))
        try:
            hit = reuse and (directory / 'report.json').exists()
            if not hit:
                job = {'name': name, 'specification': specification, 'config': config,
                       'data_path': str(data_path), 'dataset_file_sha256': dataset_file_sha,
                       'directory': str(directory), 'request_sha256': request_hash,
                       'runner_code_sha256': runner_sha, 'model_utils_code_sha256': comparison['model_utils_code_sha256'],
                       'search_key': plan['search_key'], 'test_stage_sha256': stage.report['stage_sha256']}
                _run_isolated_worker(job, directory)
            report, users = read_final_report(directory, request_hash, stage)
        except Exception as error:
            publish('error', error)
            raise
        reports[name], per_user[name] = {**report, 'cache_hit': bool(hit)}, users
        publish('running')
        LOGGER.info('Completed %s: test NDCG %.6f, Recall %.6f, cache=%s', name,
                    report['test']['ndcg@10'], report['test']['recall@10'], hit)
    fm, ids = (reports[plan['metadata_ablation'][key]] for key in ('with_metadata', 'IDs_only'))
    if fm['training_details']['examples_sha256'] != ids['training_details']['examples_sha256']:
        raise ValueError('Final FM controls use different training examples')
    if fm['training_details']['epochs_trained'] != ids['training_details']['epochs_trained']:
        raise ValueError('Final FM controls use different fixed training durations')
    pattern = reports['ials']['training_details']['observed_pattern_sha256']
    for name in set(plan['target_ablations'].values()):
        if reports[name]['training_details']['observed_pattern_sha256'] != pattern:
            raise ValueError('Final target ablation changed the confidence pattern')
    paired = _bootstrap_comparisons(per_user, plan, config)
    _save_csv(results / 'paired_test_comparisons.csv', paired)
    _save_json(results / 'final_model_details.json', {name: {k: v for k, v in report.items() if k != 'artifacts'}
                                                    for name, report in reports.items()})
    summary = publish('complete')
    return {'metrics': _final_table(reports, plan, validation_records, stage), 'paired_comparisons': paired,
            'test_stage_report': stage.report, 'model_details': reports, 'summary': summary,
            'final_plan': plan, 'test_used': True}


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    config = job['config']
    configure_reproducibility(config)
    if _file_sha(Path(__file__)) != job['runner_code_sha256'] or _file_sha(job['data_path']) != job['dataset_file_sha256']:
        raise ValueError('Worker code or input changed after its request was frozen')
    frame = pd.read_csv(job['data_path'], dtype={'user_id': 'string', 'item_id': 'string'}, parse_dates=['timestamp'])
    periods, _ = temporal_split(frame, config)
    validation = build_evaluation_stage(periods, config, 'validation')
    key, identity = search_identity(config, validation, generate_search_manifest(config))
    if key != job['search_key'] or identity['code_sha256'] != job['model_utils_code_sha256']:
        raise ValueError('Worker model/utils code differs from validation selection')
    stage = build_evaluation_stage(periods, config, 'test')
    if stage.report['stage_sha256'] != job['test_stage_sha256']:
        raise ValueError('Worker constructed a different test cohort')
    history = pd.concat([periods['train'], periods['validation']], ignore_index=True)
    fit_final_model(job['name'], job['specification'], history, stage, config,
                    Path(job['directory']), job['request_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.json')
    parser.add_argument('--no-cache', action='store_true', help='Refit all eight frozen experiments and regenerate results')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    if args.worker:
        worker(args.worker)
        return
    config = load_config(args.config)
    configure_reproducibility(config)
    path = resolve_data_dir(config) / 'cleaned_interactions.csv.gz'
    if path.exists():
        cleaned = pd.read_csv(path, dtype={'user_id': 'string', 'item_id': 'string'}, parse_dates=['timestamp'])
    else:
        from utils.data import obtain_reviews
        from utils.cleaning import clean_reviews
        raw, _ = obtain_reviews(config)
        cleaned, _ = clean_reviews(raw, config)
        cleaned = cleaned[CORE_COLUMNS].copy()
        del raw
        gc.collect()
    periods, _ = temporal_split(cleaned, config)
    result = run_final_evaluation(periods, config, reuse=not args.no_cache)
    print(canonical(result['summary']))


if __name__ == '__main__':
    main()
