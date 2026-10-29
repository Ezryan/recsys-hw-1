"""Frozen validation search, resumable verified reports, and final comparison settings.

No test frame enters this module. Local caches retain per-user validation results;
public outputs contain only aggregates, configuration lists, and provenance hashes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import itertools
import json
import logging
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import psutil
import scipy
import torch

from models import ItemKNN, IALS, EASE, FM, CustomIALS
from utils.evaluation import build_history_context
from utils.reproducibility import PROJECT_ROOT, configure_reproducibility, load_config
from utils.resources import estimate_ease_memory, select_probe_history

LOGGER = logging.getLogger(__name__)
MODEL_CLASSES = {'item_knn': ItemKNN, 'ials': IALS, 'ease': EASE, 'fm': FM, 'custom_ials': CustomIALS}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _save_csv(path, frame):
    temporary = path.with_suffix(path.suffix + '.part')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def generate_search_manifest(config):
    """Freeze concrete candidate lists, with separate seeded streams per random grid."""
    search = config['search']
    if search['selection_metric'] != 'ndcg@10' or search['tie_break'] != 'smaller_rank_or_neighbors_then_lexicographic_config':
        raise ValueError('Unsupported declared validation selection rule')
    trials = []
    for model in MODEL_CLASSES:
        settings = search[model]
        space = settings['space']
        names = sorted(space)
        grid = [dict(zip(names, values)) for values in itertools.product(*(space[name] for name in names))]
        grid.sort(key=canonical)
        if len({canonical(p) for p in grid}) != len(grid):
            raise ValueError('Search grid contains duplicate configurations')
        budget = settings['max_trials']
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(grid):
            raise ValueError('Invalid candidate budget')
        if settings['strategy'] == 'random_without_replacement':
            indices = np.random.default_rng(search['seed']).choice(len(grid), size=budget, replace=False)
            chosen = [grid[int(i)] for i in indices]
        elif settings['strategy'] == 'grid' and budget == len(grid):
            chosen = grid
        else:
            raise ValueError('A grid search must cover its full declared grid')
        for number, candidate in enumerate(chosen, 1):
            params = {('regularization' if key == 'lambda' else key): value for key, value in candidate.items()}
            if model != 'custom_ials':
                params = {**config['implementation_checks'][model], **params}
            if model == 'fm':
                params.update(batch_size=settings['fixed']['batch_size'], epochs=settings['fixed']['max_epochs'], use_metadata=True)
            trial = {'trial_id': f'{model}_{number:02d}', 'model': model, 'purpose': 'search', 'params_template': params}
            if model == 'custom_ials':
                trial['structural_params_from'] = 'best_validation_ials'
            trials.append(trial)
    return {
        'schema_version': 1, 'seed': config['seed'], 'search_seed': search['seed'],
        'selection_metric': 'ndcg@10', 'tie_break': search['tie_break'],
        'fm_epoch_tie_break': 'earliest_epoch', 'fm_patience': search['fm']['fixed']['patience'],
        'primary_trial_count': len(trials), 'trials': trials,
        'controls': {
            'fm_ids_only': 'same selected FM parameters, examples, seed, epoch count; metadata disabled',
            'custom_target': 'binary, rating_only, recency_only, rating_and_recency; frozen best iALS structure',
            'finite_half_life_rule': 'selected custom h if finite; otherwise best finite trial at selected gamma on validation',
            'additional_training_trials': 2,
            'reuse_grid_trials': 'rating_only and rating_and_recency reuse matching custom grid results',
        },
        'test_used': False,
    }


def select_best(records, model):
    candidates = [record for record in records if record['model'] == model and record['purpose'] == 'search' and record['status'] == 'ok']
    if not candidates:
        raise ValueError(f'No successful search candidates for {model}')
    def key(record):
        params = record['params']
        complexity = params.get('rank', params.get('k_neighbors', 0))
        return (-record['validation']['ndcg@10'], complexity, canonical(params))
    return min(candidates, key=key)


def search_identity(config, stage, manifest):
    code = hashlib.sha256()
    for folder in ('models', 'utils'):
        for path in sorted((PROJECT_ROOT / folder).glob('*.py')):
            code.update(str(path.relative_to(PROJECT_ROOT)).encode())
            code.update(path.read_bytes())
    identity = {
        'code_sha256': code.hexdigest(), 'manifest': manifest,
        'raw_archive_sha256': config['dataset']['sha256'], 'history_sha256': stage.context.fingerprint,
        'validation_stage_sha256': stage.report['stage_sha256'],
        'split': config['split'], 'features': config['features'], 'evaluation': config['evaluation'],
        'seed': config['seed'], 'search_settings': config['search'],
        'runtime': {name: config['runtime'][name] for name in ('cpu_threads', 'device', 'deterministic_algorithms', 'user_batch_size', 'item_batch_size')},
        'environment': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
                        'python': platform.python_version(), 'machine': platform.machine()},
    }
    return fingerprint(identity), identity


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified(path):
    wrapper = json.loads(path.read_text(encoding='utf-8'))
    if fingerprint(wrapper['payload']) != wrapper['sha256']:
        raise ValueError(f'Search cache checksum mismatch: {path}; rerun with --no-cache')
    return wrapper['payload']


def _write_verified(path, payload):
    _save_json(path, {'sha256': fingerprint(payload), 'payload': payload})


def _preflight(history, stage, config):
    """Train-only check at largest requested rank and sampling rate, without metrics."""
    memory = estimate_ease_memory(stage.context, config, psutil.virtual_memory().available)
    if not memory['fits_optimized_strategy']:
        raise MemoryError('Full EASE exceeds the declared memory budget before search')
    fragment = select_probe_history(history, config)
    rows = []
    for name, model in (
        ('ials_max_rank', IALS(rank=max(config['search']['ials']['space']['rank']), iterations=1)),
        ('fm_max_rank_q', FM(rank=max(config['search']['fm']['space']['rank']),
                             negatives_per_positive=max(config['search']['fm']['space']['negatives_per_positive']), epochs=1)),
    ):
        start = time.perf_counter()
        model.fit(fragment, config)
        scores = model.score(np.arange(min(8, len(model.context_.user_ids))), np.arange(min(16, len(model.context_.item_ids))))
        assert np.isfinite(scores).all()
        rows.append({'kernel': name, 'params': model.params, 'fit_and_score_seconds': time.perf_counter() - start,
                     'n_history_rows': len(fragment), 'validation_labels_used': False})
    return {'ease_memory': memory, 'train_only_probes': rows, 'budget_reduced': False, 'test_used': False}


def execute_trial(request, history, stage, config, cache_root, search_key, *, reuse=True):
    """Fit/evaluate once or restore a report only after request and artifact verification."""
    if stage.period != 'validation':
        raise ValueError('Hyperparameter search accepts validation only')
    directory = cache_root / request['trial_id']
    path = directory / 'report.json'
    request_hash = fingerprint({'search_key': search_key, 'request': request})
    if reuse and path.exists():
        payload = _read_verified(path)
        if payload['request_sha256'] != request_hash:
            raise ValueError('Cached trial request differs from the frozen request')
        for artifact in payload['artifacts']:
            artifact_path = directory / artifact['name']
            if not artifact_path.is_file() or _file_sha(artifact_path) != artifact['sha256']:
                raise ValueError('Search cache artifact mismatch; rerun with --no-cache')
        if payload['validation']['stage_sha256'] != stage.report['stage_sha256']:
            raise ValueError('Cached validation protocol mismatch')
        return {**payload, 'cache_hit': True}
    model = MODEL_CLASSES[request['model']](**request['params'])
    trial_config = deepcopy(config)
    if not reuse:
        trial_config['runtime']['reuse_model_cache'] = False
    start = time.perf_counter()
    if request['model'] == 'fm' and request['purpose'] == 'search':
        model.fit_with_validation(history, trial_config, stage, patience=config['search']['fm']['fixed']['patience'])
    else:
        model.fit(history, trial_config, context=stage.context)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = model.evaluate(stage)
    evaluation_seconds = time.perf_counter() - start
    if request['model'] == 'fm' and request['purpose'] == 'search':
        selected = model.validation_history_[model.selected_epoch_ - 1]
        if any(abs(result.summary[metric] - selected[metric]) > 1e-14 for metric in ('ndcg@10', 'recall@10')):
            raise ValueError('Restored FM weights do not reproduce the selected validation epoch')
    for metric in ('ndcg@10', 'recall@10'):
        if not np.isfinite(result.summary[metric]) or not 0 <= result.summary[metric] <= 1:
            raise ValueError('Search produced an invalid metric')
    directory.mkdir(parents=True, exist_ok=True)
    result.per_user.to_csv(directory / 'validation_per_user.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    loss_trace = getattr(model, 'loss_history_', [])
    epoch_trace = getattr(model, 'validation_history_', [])
    if loss_trace:
        pd.DataFrame(loss_trace).to_csv(directory / 'loss.csv', index=False)
    if epoch_trace:
        pd.DataFrame(epoch_trace).to_csv(directory / 'epochs.csv', index=False)
    artifacts = [{'name': p.name, 'sha256': _file_sha(p)} for p in sorted(directory.iterdir()) if p.name != 'report.json' and not p.name.endswith('.part')]
    payload = {
        'trial_id': request['trial_id'], 'model': request['model'], 'purpose': request['purpose'],
        'status': 'ok', 'request_sha256': request_hash, 'requested_params': request['params'], 'params': model.params,
        'validation': result.summary, 'training_details': model.training_summary_,
        'fit_seconds': fit_seconds, 'fit_compute_seconds': getattr(model, 'training_compute_seconds_', model.fit_seconds_),
        'model_fit_reused_cache': bool(getattr(model, 'cache_hit_', False)),
        'evaluation_seconds': evaluation_seconds, 'seed': config['seed'],
        'loss_trace': loss_trace, 'epoch_trace': epoch_trace, 'artifacts': artifacts, 'test_used': False,
    }
    _write_verified(path, payload)
    del model
    gc.collect()
    return {**payload, 'cache_hit': False}


def _trial_table(records):
    return pd.DataFrame([{
        'trial_id': r['trial_id'], 'model': r['model'], 'purpose': r['purpose'], 'status': r['status'],
        'requested_params': canonical(r['requested_params']), 'selected_params': canonical(r['params']),
        'validation_ndcg_at_10': r['validation']['ndcg@10'], 'validation_recall_at_10': r['validation']['recall@10'],
        'selected_epoch': r['params'].get('epochs'), 'epochs_completed': r['training_details'].get('epochs_trained'),
        'seed': r['seed'], 'fit_seconds': r['fit_seconds'], 'fit_compute_seconds': r['fit_compute_seconds'],
        'evaluation_seconds': r['evaluation_seconds'], 'result_cache_hit': r['cache_hit'],
        'model_fit_reused_cache': r['model_fit_reused_cache'], 'n_users': r['validation']['n_users'],
        'stage_sha256': r['validation']['stage_sha256'],
        'error_type': r.get('error_type'), 'error_message': r.get('error_message'),
    } for r in records])


def _resolve_controls(records, best, config):
    baseline = best['ials']['params']
    primary = best['custom_ials']['params']
    gamma, h = primary['gamma'], primary['half_life_days']
    if h == 'infinity':
        finite = [r for r in records if r['model'] == 'custom_ials' and r['params']['gamma'] == gamma
                  and r['params']['half_life_days'] != 'infinity']
        h = select_best(finite, 'custom_ials')['params']['half_life_days']
    variants = {
        'binary': best['ials'],
        'rating_only': {**baseline, 'gamma': gamma, 'half_life_days': 'infinity'},
        'recency_only': {**baseline, 'gamma': 0., 'half_life_days': h},
        'rating_and_recency': {**baseline, 'gamma': gamma, 'half_life_days': h},
    }
    for variant in ('rating_only', 'rating_and_recency'):
        variants[variant] = next(r for r in records if r['model'] == 'custom_ials' and canonical(r['params']) == canonical(variants[variant]))
    requests = [
        {'trial_id': 'control_fm_ids_only', 'model': 'fm', 'purpose': 'control',
         'params': {**best['fm']['params'], 'use_metadata': False}},
        {'trial_id': 'control_custom_recency_only', 'model': 'custom_ials', 'purpose': 'control',
         'params': variants['recency_only']},
    ]
    return requests, variants


def _final_settings(best, variants, fm_control, config, search_key, stage):
    runs = {name: {'model': name, 'params': row['params'], 'validation_trial_id': row['trial_id'], 'role': 'primary'}
            for name, row in best.items()}
    runs['fm_ids_only'] = {'model': 'fm', 'params': fm_control['params'], 'validation_trial_id': fm_control['trial_id'], 'role': 'control'}
    mapping = {'binary': 'ials'}
    for variant in ('rating_only', 'recency_only', 'rating_and_recency'):
        row = variants[variant]
        if canonical(row['params']) == canonical(best['custom_ials']['params']):
            mapping[variant] = 'custom_ials'
        else:
            name = f'custom_ials_{variant}'
            mapping[variant] = name
            runs[name] = {'model': 'custom_ials', 'params': row['params'], 'validation_trial_id': row['trial_id'], 'role': 'control'}
    return {
        'schema_version': 1, 'search_key': search_key, 'validation_stage_sha256': stage.report['stage_sha256'],
        'selection_metric': 'ndcg@10', 'settings_fixed_before_test': True,
        'refit_history': 'train+validation', 'refit_cutoff_utc': config['split']['test_start_utc'],
        'evaluation_period': 'test', 'evaluation_protocol': config['evaluation'],
        'recompute_all_history_features_and_targets_at_refit': True,
        'models': runs, 'metadata_ablation': {'with_metadata': 'fm', 'IDs_only': 'fm_ids_only'},
        'target_ablations': mapping, 'seed': config['seed'], 'test_used': False,
    }


def run_validation_search(history, stage, config, *, reuse=True, plan_only=False):
    if stage.period != 'validation' or stage.report['protocol'] != config['evaluation']:
        raise ValueError('Search requires the common validation stage')
    if build_history_context(history, stage.context.cutoff, config).fingerprint != stage.context.fingerprint:
        raise ValueError('Search history differs from the common stage history')
    started = time.perf_counter()
    manifest = generate_search_manifest(config)
    search_key, identity = search_identity(config, stage, manifest)
    public_manifest = {**manifest, 'search_key': search_key, 'plan_sha256': fingerprint(manifest),
                       'history_context_sha256': stage.context.fingerprint,
                       'validation_stage_sha256': stage.report['stage_sha256'], 'code_sha256': identity['code_sha256']}
    results = PROJECT_ROOT / config['storage']['results_dir']
    results.mkdir(parents=True, exist_ok=True)
    _save_json(results / 'search_manifest.json', public_manifest)  # Before probes or model selection.
    if plan_only:
        return {'manifest': public_manifest, 'test_used': False}
    root = PROJECT_ROOT / config['storage']['cache_dir'] / 'validation_search' / search_key
    root.mkdir(parents=True, exist_ok=True)
    preflight_path = root / 'preflight.json'
    if reuse and preflight_path.exists():
        preflight = _read_verified(preflight_path)
    else:
        preflight = _preflight(history, stage, config)
        _write_verified(preflight_path, preflight)
    _save_json(results / 'search_preflight.json', preflight)
    records = []
    def publish(status):
        if records:
            _save_csv(results / 'validation_trials.csv', _trial_table(records))
        summary = {'search_key': search_key, 'status': status,
                   'expected_primary_trials': manifest['primary_trial_count'],
                   'completed_primary_trials': sum(r['purpose'] == 'search' and r['status'] == 'ok' for r in records),
                   'completed_control_trials': sum(r['purpose'] == 'control' and r['status'] == 'ok' for r in records),
                   'result_cache_hits': sum(r['cache_hit'] for r in records),
                   'current_invocation_seconds': time.perf_counter() - started, 'test_used': False}
        _save_json(results / 'search_summary.json', summary)
        return summary
    publish('running')
    def run(request):
        LOGGER.info('Trial %s: %s', request['trial_id'], canonical(request['params']))
        start = time.perf_counter()
        try:
            record = execute_trial(request, history, stage, config, root / 'trials', search_key, reuse=reuse)
        except Exception as error:
            records.append({
                **request, 'requested_params': request['params'], 'status': 'error', 'cache_hit': False,
                'training_details': {}, 'fit_seconds': time.perf_counter() - start, 'fit_compute_seconds': None,
                'model_fit_reused_cache': False, 'evaluation_seconds': None, 'seed': config['seed'],
                'validation': {'ndcg@10': None, 'recall@10': None, 'n_users': len(stage.user_indices),
                               'stage_sha256': stage.report['stage_sha256']},
                'error_type': type(error).__name__, 'error_message': str(error),
            })
            publish('error')
            _save_json(root / f"{request['trial_id']}_failure.json", records[-1])
            raise
        records.append(record)
        publish('running')
        LOGGER.info('Completed %s: NDCG %.6f, Recall %.6f, cache=%s', request['trial_id'],
                    record['validation']['ndcg@10'], record['validation']['recall@10'], record['cache_hit'])
        return record
    for template in manifest['trials']:
        params = deepcopy(template['params_template'])
        if template['model'] == 'custom_ials':
            params = {**select_best(records, 'ials')['params'], **params}
        run({'trial_id': template['trial_id'], 'model': template['model'], 'purpose': 'search', 'params': params})
    best = {name: select_best(records, name) for name in MODEL_CLASSES}
    requests, variants = _resolve_controls(records, best, config)
    public_manifest['resolved_controls'] = requests
    public_manifest['resolved_custom_structural_params'] = best['ials']['params']
    _save_json(results / 'search_manifest.json', public_manifest)
    fm_control = run(requests[0])
    variants['recency_only'] = run(requests[1])
    if best['fm']['training_details']['examples_sha256'] != fm_control['training_details']['examples_sha256']:
        raise ValueError('Controlled FM comparison uses different training examples')
    for row in variants.values():
        if row['training_details']['observed_pattern_sha256'] != best['ials']['training_details']['observed_pattern_sha256']:
            raise ValueError('Target ablation changed the positive confidence pattern')
    settings = _final_settings(best, variants, fm_control, config, search_key, stage)
    chosen = {name: {'params': r['params'], 'validation': r['validation'], 'trial_id': r['trial_id'],
                     'seed': config['seed'], 'search_key': search_key, 'fit_seconds': r['fit_seconds']}
              for name, r in best.items()}
    control_rows = []
    for family, members, reference in (
        ('custom_target', variants, best['ials']),
        ('fm_metadata', {'with_metadata': best['fm'], 'IDs_only': fm_control}, fm_control),
    ):
        for variant, row in members.items():
            control_rows.append({'family': family, 'variant': variant, 'trial_id': row['trial_id'],
                                 'params': canonical(row['params']), 'ndcg@10': row['validation']['ndcg@10'],
                                 'recall@10': row['validation']['recall@10'],
                                 'ndcg_delta_vs_reference': row['validation']['ndcg@10'] - reference['validation']['ndcg@10'],
                                 'recall_delta_vs_reference': row['validation']['recall@10'] - reference['validation']['recall@10'],
                                 'stage_sha256': row['validation']['stage_sha256']})
    _save_csv(results / 'validation_controls.csv', pd.DataFrame(control_rows))
    epoch_rows = [{**epoch, 'trial_id': r['trial_id'], 'params': canonical(r['requested_params']),
                   'selected_epoch': r['params']['epochs']} for r in records for epoch in r.get('epoch_trace', [])]
    epoch_table = pd.DataFrame(epoch_rows, columns=['epoch', 'ndcg@10', 'recall@10', 'evaluation_seconds',
                                                   'improved', 'best_epoch_so_far', 'stage_sha256',
                                                   'trial_id', 'params', 'selected_epoch'])
    _save_csv(results / 'fm_validation_epochs.csv', epoch_table)
    _save_json(results / 'best_params.json', chosen)
    _save_json(results / 'final_experiment_plan.json', settings)
    summary = publish('complete')
    return {'manifest': public_manifest, 'best_params': chosen, 'trials': _trial_table(records),
            'controls': pd.DataFrame(control_rows), 'fm_epochs': epoch_table,
            'final_plan': settings, 'summary': summary, 'test_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.json')
    parser.add_argument('--no-cache', action='store_true', help='Recompute every trial and EASE weights')
    parser.add_argument('--plan-only', action='store_true', help='Freeze concrete candidate lists before training')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    config = load_config(args.config)
    configure_reproducibility(config)
    from utils.data import obtain_reviews
    from utils.cleaning import clean_reviews
    from utils.splitting import temporal_split
    from utils.evaluation import build_evaluation_stage
    raw, _ = obtain_reviews(config)
    cleaned, _ = clean_reviews(raw, config)
    periods, _ = temporal_split(cleaned, config)
    stage = build_evaluation_stage(periods, config, 'validation')
    result = run_validation_search(periods['train'], stage, config, reuse=not args.no_cache, plan_only=args.plan_only)
    print(canonical(result.get('summary', {'primary_trials': result['manifest']['primary_trial_count'], 'search_key': result['manifest']['search_key']})))


if __name__ == '__main__':
    main()
