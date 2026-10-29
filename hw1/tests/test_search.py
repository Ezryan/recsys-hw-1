"""Search budgets, selection, early stopping, failure handling, and verified replay."""
from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models import FM
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config
from utils.search import (MODEL_CLASSES, canonical, execute_trial, generate_search_manifest,
                          run_validation_search, search_identity, select_best)


def fixture():
    config=deepcopy(load_config())
    config['split'].update(validation_start_utc='2020-01-02T00:00:00+00:00',test_start_utc='2020-01-03T00:00:00+00:00')
    rows=[('A','item-00',5),('A','item-01',1)]+[('seed',f'item-{i:02d}',5) for i in range(13)]
    history=pd.DataFrame([{'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,
                           'timestamp':pd.Timestamp('2020-01-01',tz='UTC')}
                          for j,(u,i,r) in enumerate(rows)])
    future=pd.DataFrame([{'user_id':'A','item_id':'item-02','rating':5.,'source_line':100,'timestamp':pd.Timestamp('2020-01-02',tz='UTC')}])
    stage=build_evaluation_stage({'train':history,'validation':future},config)
    return config,history,stage


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.config,self.history,self.stage=fixture()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.config['storage']['results_dir']=str(Path(self.temp.name)/'results')
        self.config['storage']['cache_dir']=str(Path(self.temp.name)/'cache')
        self.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.previous_threads)

    def tiny_config(self):
        config=deepcopy(self.config)
        spaces={
            'item_knn':{'k_neighbors':[3],'shrinkage':[1.]},
            'ials':{'rank':[2],'lambda':[.2],'alpha':[2.],'iterations':[2]},
            'ease':{'lambda':[2.]},
            'fm':{'rank':[2],'learning_rate':[.01],'l2':[.001],'negatives_per_positive':[1]},
            'custom_ials':{'gamma':[1.],'half_life_days':[2.,'infinity']},
        }
        for model,space in spaces.items():
            config['search'][model].update(space=space,max_trials=np.prod([len(x) for x in space.values()]).item(),strategy='grid')
        config['search']['fm']['fixed'].update(max_epochs=4,patience=2)
        return config

    def test_manifest_has_all_declared_unique_candidates_and_dependency(self):
        manifest=generate_search_manifest(self.config)
        self.assertEqual(manifest['primary_trial_count'],46)
        for model,n in (('item_knn',12),('ials',12),('ease',4),('fm',12),('custom_ials',6)):
            rows=[r for r in manifest['trials'] if r['model']==model]
            self.assertEqual(len(rows),n)
            self.assertEqual(len({canonical(r['params_template']) for r in rows}),n)
            for row in rows:
                if model=='custom_ials':
                    self.assertEqual(row['structural_params_from'],'best_validation_ials')
                    self.assertEqual(set(row['params_template']),{'gamma','half_life_days'})
                if model=='fm':
                    self.assertEqual(row['params_template']['epochs'],20)
                    self.assertTrue(row['params_template']['use_metadata'])
        self.assertEqual(manifest['controls']['additional_training_trials'],2)

    def test_seeded_candidate_lists_reproduce_and_grid_order_is_key_order_independent(self):
        first=generate_search_manifest(self.config)
        config=deepcopy(self.config)
        for model in MODEL_CLASSES:
            config['search'][model]['space']=dict(reversed(list(config['search'][model]['space'].items())))
        self.assertEqual(first,generate_search_manifest(config))
        config['search']['seed']+=1
        other=generate_search_manifest(config)
        self.assertNotEqual([r for r in first['trials'] if r['model']=='ials'],[r for r in other['trials'] if r['model']=='ials'])
        self.assertEqual([r for r in first['trials'] if r['model']=='ease'],[r for r in other['trials'] if r['model']=='ease'])

    def test_selection_uses_ndcg_then_complexity_then_lexical_params_not_recall(self):
        def row(rank,alpha,ndcg,recall):
            return {'model':'ials','purpose':'search','status':'ok','params':{'rank':rank,'alpha':alpha},'validation':{'ndcg@10':ndcg,'recall@10':recall}}
        rows=[row(64,1,.2,.9),row(32,2,.2,.1),row(32,1,.2,.05),row(16,1,.19,1.)]
        self.assertEqual(select_best(rows,'ials'),rows[2])
        failed=row(1,0,1.,1.);failed['status']='error'
        self.assertEqual(select_best([failed,*rows],'ials'),rows[2])
        with self.assertRaises(ValueError):select_best([],'ials')

    def test_identity_changes_with_data_seed_features_and_split(self):
        manifest=generate_search_manifest(self.config)
        key,_=search_identity(self.config,self.stage,manifest)
        for field in ('seed','features','split'):
            config=deepcopy(self.config)
            if field=='seed':config['seed']+=1
            elif field=='features':config['features']['include_month']=True
            else:config['split']['test_start_utc']='2020-01-04T00:00:00+00:00'
            self.assertNotEqual(search_identity(config,self.stage,manifest)[0],key)
        changed=replace(self.stage,report={**self.stage.report,'stage_sha256':'different-labels'})
        self.assertNotEqual(search_identity(self.config,changed,manifest)[0],key)

    def test_trial_cache_verifies_artifacts_and_force_recomputes_corruption(self):
        request={'trial_id':'knn','model':'item_knn','purpose':'search','params':{'k_neighbors':3}}
        root=Path(self.temp.name)/'trials'
        first=execute_trial(request,self.history,self.stage,self.config,root,'key')
        with patch('models.item_knn.ItemKNN.fit',side_effect=AssertionError('cache must skip fitting')):
            second=execute_trial(request,self.history,self.stage,self.config,root,'key')
        self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'])
        self.assertEqual(first['validation'],second['validation'])
        (root/'knn/validation_per_user.csv.gz').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'artifact mismatch'):
            execute_trial(request,self.history,self.stage,self.config,root,'key')
        repaired=execute_trial(request,self.history,self.stage,self.config,root,'key',reuse=False)
        self.assertEqual(first['validation'],repaired['validation'])

    def test_cache_request_and_json_checksum_mismatches_are_rejected(self):
        request={'trial_id':'knn','model':'item_knn','purpose':'search','params':{'k_neighbors':3}}
        root=Path(self.temp.name)/'trials'
        execute_trial(request,self.history,self.stage,self.config,root,'key')
        changed={**request,'params':{'k_neighbors':4}}
        with self.assertRaisesRegex(ValueError,'request differs'):
            execute_trial(changed,self.history,self.stage,self.config,root,'key')
        path=root/'knn/report.json';value=json.loads(path.read_text());value['payload']['seed']=999
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            execute_trial(request,self.history,self.stage,self.config,root,'key')

    def test_fm_early_stopping_restores_earliest_best_epoch_and_matches_fixed_refit(self):
        params={'rank':2,'epochs':20,'batch_size':64,'learning_rate':.01}
        values=[.1,.3,.3,.2,.1]
        outputs=[SimpleNamespace(summary={'ndcg@10':v,'recall@10':v/2,'stage_sha256':self.stage.report['stage_sha256']}) for v in values]
        with patch('models.fm.evaluate_scorer',side_effect=outputs):
            model=FM(**params).fit_with_validation(self.history,self.config,self.stage,patience=3)
        reference=FM(**{**params,'epochs':2}).fit(self.history,self.config,context=self.stage.context)
        self.assertEqual(model.selected_epoch_,2);self.assertEqual(model.epochs_completed_,5)
        self.assertEqual(model.params['epochs'],2)
        self.assertEqual(len(model.validation_history_),5)
        self.assertEqual(model.training_summary_['weights_sha256'],reference.training_summary_['weights_sha256'])
        self.assertEqual(model.training_summary_['final_train_bce'],reference.training_summary_['final_train_bce'])
        np.testing.assert_array_equal(model.score([0,1],np.arange(13)),reference.score([0,1],np.arange(13)))
        # Subsequent ordinary refit must not keep or consult validation callbacks.
        with patch('models.fm.evaluate_scorer',side_effect=AssertionError('No validation during refit')):
            model.fit(self.history,self.config)

    def test_fm_stops_at_max_epochs_if_validation_keeps_improving(self):
        values=[.1,.2,.3]
        outputs=[SimpleNamespace(summary={'ndcg@10':v,'recall@10':v,'stage_sha256':'stage'}) for v in values]
        with patch('models.fm.evaluate_scorer',side_effect=outputs):
            model=FM(rank=2,epochs=3).fit_with_validation(self.history,self.config,self.stage,patience=2)
        self.assertEqual(model.selected_epoch_,3);self.assertEqual(model.epochs_completed_,3)

    def test_test_stage_invalid_patience_and_wrong_history_are_rejected(self):
        wrong=replace(self.stage,period='test')
        with self.assertRaises(ValueError):run_validation_search(self.history,wrong,self.config)
        with self.assertRaises(ValueError):FM().fit_with_validation(self.history,self.config,wrong)
        for patience in (0,True,1.5):
            with self.assertRaises(ValueError):FM().fit_with_validation(self.history,self.config,self.stage,patience=patience)
        history=self.history.copy();history.loc[0,'rating']=4.
        with self.assertRaisesRegex(ValueError,'history differs'):
            run_validation_search(history,self.stage,self.config)

    def test_small_end_to_end_search_resumes_all_trials_and_freezes_final_controls(self):
        config=self.tiny_config()
        with patch('utils.search._preflight',return_value={'train_only':True}) as probe:
            first=run_validation_search(self.history,self.stage,config)
            second=run_validation_search(self.history,self.stage,config)
            probe.assert_called_once()
        self.assertEqual(len(first['trials']),8)
        self.assertEqual(first['summary']['completed_primary_trials'],6)
        self.assertEqual(first['summary']['completed_control_trials'],2)
        self.assertEqual(second['summary']['result_cache_hits'],8)
        self.assertEqual(first['best_params'],second['best_params'])
        self.assertEqual(set(first['best_params']),set(MODEL_CLASSES))
        self.assertEqual(len(first['final_plan']['models']),8)
        self.assertTrue(first['final_plan']['settings_fixed_before_test'])
        self.assertFalse(first['final_plan']['test_used'])
        for key in first['best_params']['ials']['params']:
            self.assertEqual(first['best_params']['ials']['params'][key],first['best_params']['custom_ials']['params'][key])
        for name,row in first['final_plan']['models'].items():
            self.assertEqual(row['params']['epochs'],first['best_params']['fm']['params']['epochs']) if row['model']=='fm' else None
        self.assertEqual(first['controls']['stage_sha256'].nunique(),1)
        self.assertEqual(first['controls'].shape[0],6)
        for path in Path(config['storage']['results_dir']).iterdir():
            content=path.read_text()
            self.assertNotIn('item-02',content)
        epochs=first['fm_epochs']
        selected=int(epochs.sort_values(['ndcg@10','epoch'],ascending=[False,True]).iloc[0]['epoch'])
        self.assertEqual(first['best_params']['fm']['params']['epochs'],selected)

    def test_failed_trial_is_recorded_as_error_with_empty_metrics_and_can_retry(self):
        config=self.tiny_config()
        with patch('utils.search._preflight',return_value={'train_only':True}),patch('utils.search.execute_trial',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                run_validation_search(self.history,self.stage,config)
        rows=pd.read_csv(Path(config['storage']['results_dir'])/'validation_trials.csv')
        self.assertEqual(rows.iloc[0]['status'],'error')
        self.assertTrue(pd.isna(rows.iloc[0]['validation_ndcg_at_10']))
        self.assertEqual(rows.iloc[0]['error_type'],'RuntimeError')
        self.assertFalse((Path(config['storage']['results_dir'])/'best_params.json').exists())
        with patch('utils.search._preflight',return_value={'train_only':True}):
            result=run_validation_search(self.history,self.stage,config)
        self.assertEqual(result['summary']['status'],'complete')
        self.assertTrue(result['trials']['status'].eq('ok').all())


if __name__=='__main__':
    unittest.main()
