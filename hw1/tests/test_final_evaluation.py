"""Frozen-selection guards, final refit, isolated workers, and paired bootstrap."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import final_evaluation as final
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config
from utils.search import run_validation_search, _save_json
from utils.splitting import split_content_sha256, temporal_split


def final_fixture(directory):
    config=deepcopy(load_config())
    config['storage'].update(results_dir=str(directory/'results'),cache_dir=str(directory/'cache'),data_dir_default=str(directory/'data'))
    config['split'].update(validation_start_utc='2020-01-02T00:00:00+00:00',test_start_utc='2020-01-03T00:00:00+00:00')
    config['controls']['paired_bootstrap_resamples']=100
    rows=[('private-user','item-00',5),('private-user','item-01',1)]+[('private-seed',f'item-{i:02d}',5) for i in range(13)]
    train=pd.DataFrame([{'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,'timestamp':pd.Timestamp('2020-01-01',tz='UTC')}
                        for j,(u,i,r) in enumerate(rows)])
    validation=pd.DataFrame([{'user_id':'private-user','item_id':'item-02','rating':5.,'source_line':100,'timestamp':pd.Timestamp('2020-01-02',tz='UTC')}])
    test=pd.DataFrame([{'user_id':u,'item_id':i,'rating':5.,'source_line':200+j,'timestamp':pd.Timestamp('2020-01-03',tz='UTC')}
                       for j,(u,i) in enumerate([('private-user','item-03'),('private-cold','item-04'),('private-user','private-future-item')])])
    merged=pd.concat([train,validation,test],ignore_index=True)
    config['split']['boundary_source_content_sha256']=split_content_sha256(merged)
    spaces={
        'item_knn':{'k_neighbors':[3],'shrinkage':[1.]},
        'ials':{'rank':[2],'lambda':[.2],'alpha':[2.],'iterations':[2]},
        'ease':{'lambda':[2.]},
        'fm':{'rank':[2],'learning_rate':[.01],'l2':[.001],'negatives_per_positive':[1]},
        'custom_ials':{'gamma':[1.],'half_life_days':[2.,'infinity']},
    }
    for name,space in spaces.items():
        config['search'][name].update(space=space,max_trials=int(np.prod([len(x) for x in space.values()])),strategy='grid')
    config['search']['fm']['fixed'].update(max_epochs=2,patience=1)
    periods={'train':train,'validation':validation,'test':test}
    stage=build_evaluation_stage(periods,config,'validation')
    with patch('utils.search._preflight',return_value={'train_only':True}):
        run_validation_search(train,stage,config)
    return config,periods


def in_process_worker(job,directory):
    frame=pd.read_csv(job['data_path'],dtype={'user_id':'string','item_id':'string'},parse_dates=['timestamp'])
    periods,_=temporal_split(frame,job['config'])
    stage=build_evaluation_stage(periods,job['config'],'test')
    history=pd.concat([periods['train'],periods['validation']],ignore_index=True)
    final.fit_final_model(job['name'],job['specification'],history,stage,job['config'],directory,job['request_sha256'])


class FinalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.previous_threads)
        self.config,self.periods=final_fixture(self.directory)
        self.results=self.directory/'results'

    def test_selection_validation_does_not_access_test_and_rejects_modified_params(self):
        class Protected(dict):
            def __getitem__(self,key):
                if key=='test':raise AssertionError('test must stay unopened')
                return super().__getitem__(key)
        plan,_,_,_=final.validate_frozen_selection(Protected(self.periods),self.config)
        plan['models']['ials']['params']['rank']=3
        _save_json(self.results/'final_experiment_plan.json',plan)
        with self.assertRaisesRegex(ValueError,'parameters not fixed'):
            final.run_final_evaluation(Protected(self.periods),self.config)
        self.assertFalse((self.results/'test_access_manifest.json').exists())

    def test_comparison_manifest_is_written_before_first_test_access(self):
        results=self.results
        class Observed(dict):
            def __getitem__(self,key):
                if key=='test':
                    assert (results/'test_access_manifest.json').exists()
                return super().__getitem__(key)
        with patch('final_evaluation._run_isolated_worker',side_effect=in_process_worker):
            result=final.run_final_evaluation(Observed(self.periods),self.config)
        self.assertEqual(result['summary']['completed_experiments'],8)

    def test_final_fit_rebuilds_t2_history_and_controlled_examples_and_can_replay(self):
        with patch('final_evaluation._run_isolated_worker',side_effect=in_process_worker) as worker:
            result=final.run_final_evaluation(self.periods,self.config)
            replay=final.run_final_evaluation(self.periods,self.config)
        self.assertEqual(worker.call_count,8)
        self.assertEqual(replay['summary']['result_cache_hits'],8)
        self.assertEqual(result['metrics']['stage_sha256'].nunique(),1)
        self.assertTrue(result['metrics']['n_users'].eq(1).all())
        self.assertTrue(result['metrics']['n_items'].eq(13).all())
        self.assertTrue(result['metrics']['n_relevant_pairs'].eq(1).all())
        self.assertTrue(result['metrics']['test_recall_at_10'].eq(1).all())
        self.assertEqual(len(result['paired_comparisons']),12)
        for name,details in result['model_details'].items():
            self.assertEqual(details['history_context_sha256'],result['test_stage_report']['history_context_sha256'])
            self.assertFalse(details['model_fit_reused_cache'])
            if details['model_family']=='custom_ials':
                self.assertEqual(details['training_details']['target_reference_cutoff_utc'],'2020-01-03T00:00:00+00:00')
            if details['model_family']=='fm':
                self.assertEqual(details['training_details']['max_epochs_requested'],details['params']['epochs'])
                self.assertEqual(details['training_details']['epochs_trained'],details['params']['epochs'])
        fm,ids=(result['model_details'][name]['training_details'] for name in ('fm','fm_ids_only'))
        self.assertEqual(fm['examples_sha256'],ids['examples_sha256'])
        self.assertEqual(fm['n_observed'],16)
        self.assertEqual(result['test_stage_report']['cold_start']['positive_events_with_either'],2)
        for path in self.results.iterdir():
            if path.is_file():self.assertNotIn('private-user',path.read_text())

    def test_changing_bootstrap_or_plan_after_test_access_is_rejected(self):
        with patch('final_evaluation._run_isolated_worker',side_effect=in_process_worker):
            final.run_final_evaluation(self.periods,self.config)
        self.config['controls']['paired_bootstrap_resamples']+=1
        with self.assertRaisesRegex(ValueError,'changed after test access'):
            final.run_final_evaluation(self.periods,self.config)

    def test_corrupted_recommendations_are_detected_and_force_refit_repairs(self):
        with patch('final_evaluation._run_isolated_worker',side_effect=in_process_worker):
            result=final.run_final_evaluation(self.periods,self.config)
        root=self.directory/'cache/final_evaluation'/result['summary']['final_cache_key']
        (root/'item_knn/recommendations.npy').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'artifact checksum mismatch'):
            final.run_final_evaluation(self.periods,self.config)
        with patch('final_evaluation._run_isolated_worker',side_effect=in_process_worker):
            repaired=final.run_final_evaluation(self.periods,self.config,reuse=False)
        np.testing.assert_allclose(repaired['metrics'].test_ndcg_at_10,result['metrics'].test_ndcg_at_10)

    def test_missing_successful_search_candidate_prevents_test_access(self):
        ledger=pd.read_csv(self.results/'validation_trials.csv')
        ledger.iloc[1:].to_csv(self.results/'validation_trials.csv',index=False)
        with self.assertRaisesRegex(ValueError,'complete declared search'):
            final.run_final_evaluation(self.periods,self.config)
        self.assertFalse((self.results/'test_access_manifest.json').exists())

    def test_isolated_worker_executes_exact_t2_request_and_returns_measured_rss(self):
        original=final._run_isolated_worker
        invoked=[]
        def executor(job,directory):
            if not invoked:
                original(job,directory)
            else:in_process_worker(job,directory)
            invoked.append(job['name'])
        with patch('final_evaluation._run_isolated_worker',side_effect=executor):
            result=final.run_final_evaluation(self.periods,self.config)
        self.assertEqual(invoked[0],'item_knn')
        report=result['model_details']['item_knn']
        self.assertGreater(report['peak_worker_rss_bytes'],0)
        self.assertGreaterEqual(report['peak_worker_rss_bytes'],report['baseline_worker_rss_bytes'])
        self.assertGreater(report['parameter_storage_bytes'],0)

    def test_worker_failure_leaves_explicit_incomplete_status(self):
        with patch('final_evaluation._run_isolated_worker',side_effect=RuntimeError('synthetic worker failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic worker failure'):
                final.run_final_evaluation(self.periods,self.config)
        summary=json.loads((self.results/'final_run_summary.json').read_text())
        self.assertEqual(summary['status'],'error')
        self.assertEqual(summary['completed_experiments'],0)
        self.assertEqual(summary['error']['type'],'RuntimeError')


class BootstrapTests(unittest.TestCase):
    def test_identical_predictions_give_exact_zero_interval(self):
        table=pd.DataFrame({'user_index':[0,1,2],'ndcg@10':[0.,.3,1.],'recall@10':[0.,.5,1.]})
        result=final.paired_user_bootstrap(table,table)
        np.testing.assert_array_equal(result[['delta','ci_lower','ci_upper']],np.zeros((2,3)))
        self.assertFalse(result.ci_excludes_zero.any())

    def test_known_two_user_interval_and_metric_pairing(self):
        left=pd.DataFrame({'user_index':[0,1],'ndcg@10':[0.,1.],'recall@10':[0.,.5]})
        right=left.copy();right[['ndcg@10','recall@10']]=0.
        result=final.paired_user_bootstrap(left,right,seed=42,resamples=2000)
        np.testing.assert_allclose(result.delta,[.5,.25])
        np.testing.assert_allclose(result.ci_lower,[0.,0.])
        np.testing.assert_allclose(result.ci_upper,[1.,.5])
        again=final.paired_user_bootstrap(left,right,seed=42,resamples=2000,batch_size=17)
        pd.testing.assert_frame_equal(result,again)

    def test_one_user_constant_positive_and_negative_differences(self):
        left=pd.DataFrame({'user_index':[9],'ndcg@10':[.8],'recall@10':[.2]})
        right=pd.DataFrame({'user_index':[9],'ndcg@10':[.3],'recall@10':[.7]})
        result=final.paired_user_bootstrap(left,right)
        np.testing.assert_allclose(result.delta,[.5,-.5])
        np.testing.assert_allclose(result.ci_lower,result.delta)
        np.testing.assert_allclose(result.ci_upper,result.delta)
        self.assertTrue(result.ci_excludes_zero.all())

    def test_bad_cohorts_metrics_and_bootstrap_settings_are_rejected(self):
        table=pd.DataFrame({'user_index':[0,1],'ndcg@10':[0.,1.],'recall@10':[0.,1.]})
        for right in (table.iloc[::-1],table.iloc[:1],table.iloc[:0],table.assign(**{'ndcg@10':[np.nan,1.]})):
            with self.assertRaises(ValueError):final.paired_user_bootstrap(table,right)
        for settings in ({'resamples':0},{'resamples':True},{'confidence':1},{'batch_size':0}):
            with self.assertRaises(ValueError):final.paired_user_bootstrap(table,table,**settings)


if __name__=='__main__':
    unittest.main()
