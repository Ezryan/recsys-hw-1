"""Custom preference math, unchanged confidence, exact degeneration, and refit cutoff."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from models import IALS, CustomIALS
from models.custom_ials import custom_preferences, summarize_targets
from models.ials import als_update, ials_objective
from utils.evaluation import build_evaluation_stage
from utils.model_checks import run_custom_ials_control_checks
from utils.reproducibility import load_config


def history_frame(rows):
    return pd.DataFrame([
        {'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,
         'timestamp':pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=day)}
        for j,(day,u,i,r) in enumerate(rows)
    ])


class CustomIalsTests(unittest.TestCase):
    def setUp(self):
        self.config=deepcopy(load_config())
        self.config['split'].update(validation_start_utc='2020-01-06T00:00:00+00:00',test_start_utc='2020-01-10T00:00:00+00:00')
        self.history=history_frame([(0,'u0','a',4),(1,'u0','b',1),(2,'u1','b',5),(0,'u1','c',4),(2,'u2','d',2)])

    def target_map(self, model):
        values={}
        for row,user in enumerate(model.context_.user_ids):
            a,b=model.targets_.indptr[row:row+2]
            for item,p in zip(model.targets_.indices[a:b],model.targets_.data[a:b]):
                values[user,model.context_.item_ids[item]]=p
        return values

    def stage(self):
        rows=[(0,'synthetic-user','item-00',5),(0,'synthetic-user','item-01',1)]
        rows += [(0,'synthetic-seed',f'item-{i:02d}',5) for i in range(13)]
        history=history_frame(rows)
        future=pd.DataFrame([{'user_id':'synthetic-user','item_id':'item-02','rating':5.,'source_line':100,
                              'timestamp':pd.Timestamp('2020-01-06',tz='UTC')}])
        return history,build_evaluation_stage({'train':history,'validation':future},self.config)

    def test_hand_formula_uses_rating_strength_and_half_life(self):
        cutoff=pd.Timestamp('2020-01-21',tz='UTC')
        times=pd.to_datetime(['2020-01-11','2020-01-01','2020-01-11','2020-01-01'],utc=True)
        result=custom_preferences([4,5,3,1],times,cutoff,gamma=1,half_life_days=10)
        np.testing.assert_allclose(result,[.25,.25,0,0],atol=1e-15)
        result=custom_preferences([4,5],times[:2],cutoff,gamma=2,half_life_days=10)
        np.testing.assert_allclose(result,[.125,.25],atol=1e-15)

    def test_rating_only_ignores_age_and_recency_only_ignores_positive_rating_strength(self):
        times=pd.to_datetime(['2020-01-01','2020-01-01'],utc=True)
        a=custom_preferences([4,5],times,'2020-01-11T00:00:00+00:00',gamma=1,half_life_days='infinity')
        b=custom_preferences([4,5],times,'2021-01-11T00:00:00+00:00',gamma=1,half_life_days=np.inf)
        np.testing.assert_array_equal(a,[.5,1.]);np.testing.assert_array_equal(a,b)
        recency=custom_preferences([4,5],times,'2020-01-11T00:00:00+00:00',gamma=0,half_life_days=10)
        np.testing.assert_array_equal(recency,[.5,.5])

    def test_binary_limit_reproduces_targets_factors_loss_and_scores_bit_for_bit(self):
        params={'rank':3,'regularization':.2,'alpha':2.5,'iterations':3,'init_std':.02}
        baseline=IALS(**params).fit(self.history,self.config)
        custom=CustomIALS(**params,gamma=0,half_life_days='infinity').fit(self.history,self.config)
        np.testing.assert_array_equal(custom.targets_.data,baseline.targets_.data)
        np.testing.assert_array_equal(custom.user_factors_,baseline.user_factors_)
        np.testing.assert_array_equal(custom.item_factors_,baseline.item_factors_)
        self.assertEqual(custom.loss_history_,baseline.loss_history_)
        np.testing.assert_array_equal(custom.score([0,1],[0,1,2]),baseline.score([0,1],[0,1,2]))
        self.assertTrue(custom.training_summary_['recovers_binary_target'])

    def test_target_coordinates_align_with_csr_after_input_permutation(self):
        first=CustomIALS(rank=2,iterations=2,half_life_days=2).fit(self.history,self.config)
        second=CustomIALS(rank=2,iterations=2,half_life_days=2).fit(self.history.sample(frac=1,random_state=7),self.config)
        expected={('u0','a'):.5*2**(-5/2),('u1','b'):2**(-3/2),('u1','c'):.5*2**(-5/2)}
        self.assertEqual(set(self.target_map(first)),set(expected))
        for key,value in expected.items():self.assertAlmostEqual(self.target_map(first)[key],value)
        np.testing.assert_array_equal(first.targets_.data,second.targets_.data)
        np.testing.assert_array_equal(first.user_factors_,second.user_factors_)
        self.assertEqual(first.training_summary_['factors_sha256'],second.training_summary_['factors_sha256'])

    def test_custom_normal_equations_and_loss_match_dense_unchanged_confidence(self):
        model=CustomIALS(rank=2,iterations=2,alpha=2.5,regularization=.2,half_life_days=2).fit(self.history,self.config)
        observed,target=model.observed_.toarray(),model.targets_.toarray()
        confidence=1+2.5*observed
        y=np.array([[.1,.5],[-.2,.3],[.7,-.1],[.4,.2]])
        expected=np.array([np.linalg.solve(y.T@np.diag(c)@y+.2*np.eye(2),y.T@(c*p)) for c,p in zip(confidence,target)])
        actual=als_update(model.observed_,model.targets_,y,alpha=2.5,regularization=.2)
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        expected_loss=np.sum(confidence*np.square(target-actual@y.T))+.2*(np.square(actual).sum()+np.square(y).sum())
        self.assertAlmostEqual(ials_objective(model.observed_,model.targets_,actual,y,alpha=2.5,regularization=.2),expected_loss)
        self.assertEqual(model.training_summary_['observed_confidence'],3.5)
        self.assertFalse(model.training_summary_['confidence_changed'])

    def test_zero_target_coordinate_still_has_observed_confidence(self):
        observed=sparse.csr_matrix([[1.,1.]])
        target=observed.copy();target.data[:]=[0.,1.]
        result=als_update(observed,target,np.array([[2.],[3.]]),alpha=2,regularization=1)
        self.assertAlmostEqual(result[0,0],9/40)
        self.assertNotAlmostEqual(result[0,0],9/32)

    def test_numerical_decay_underflow_keeps_all_observed_coordinates(self):
        model=CustomIALS(rank=2,iterations=2,half_life_days=1e-300).fit(self.history,self.config)
        baseline=IALS(rank=2,iterations=2).fit(self.history,self.config)
        self.assertEqual(model.targets_.nnz,3)
        np.testing.assert_array_equal(model.targets_.data,np.zeros(3))
        np.testing.assert_array_equal(model.targets_.indices,model.observed_.indices)
        self.assertEqual(model.training_summary_['observed_pattern_sha256'],baseline.training_summary_['observed_pattern_sha256'])
        self.assertEqual(model.training_summary_['target_statistics']['zero_targets'],3)
        np.testing.assert_array_equal(model.user_factors_,np.zeros((3,2)))
        self.assertEqual(model.loss_history_[-1]['objective'],0)

    def test_initialization_is_the_same_seeded_baseline_stream(self):
        model=CustomIALS(rank=2,iterations=2,init_std=.02).fit(self.history,self.config)
        rng=np.random.default_rng(self.config['seed'])
        x=rng.normal(0,.02,(3,2));y=rng.normal(0,.02,(4,2))
        expected=ials_objective(model.observed_,model.targets_,x,y,alpha=10,regularization=.1)
        self.assertEqual(model.loss_history_[0]['objective'],expected)

    def test_every_half_step_decreases_its_own_fixed_target_objective(self):
        for gamma,h in ((1,'infinity'),(0,2),(1,2)):
            model=CustomIALS(rank=2,iterations=4,gamma=gamma,half_life_days=h).fit(self.history,self.config)
            values=np.array([row['objective'] for row in model.loss_history_])
            self.assertEqual(len(values),9)
            self.assertTrue((np.diff(values)<=1e-10*np.maximum(1,values[:-1])).all())
            self.assertLess(values[-1],values[0])

    def test_refit_uses_new_cutoff_and_combined_history_instead_of_old_target_values(self):
        first=CustomIALS(rank=2,iterations=2,half_life_days=2).fit(self.history,self.config)
        extra=pd.DataFrame([{'user_id':'u0','item_id':'e','rating':5.,'source_line':100,'timestamp':pd.Timestamp('2020-01-07',tz='UTC')}])
        combined=pd.concat([self.history,extra],ignore_index=True)
        refit=CustomIALS(rank=2,iterations=2,half_life_days=2).fit(combined,self.config,cutoff=self.config['split']['test_start_utc'])
        old,new=self.target_map(first),self.target_map(refit)
        for key,p in old.items():self.assertAlmostEqual(new[key],p*.25)
        self.assertAlmostEqual(new['u0','e'],2**(-3/2))
        self.assertEqual(refit.training_summary_['target_reference_cutoff_utc'],'2020-01-10T00:00:00+00:00')

    def test_empty_positive_history_has_zero_factors_and_defined_empty_statistics(self):
        history=self.history.copy();history['rating']=1.
        model=CustomIALS(rank=2,iterations=2).fit(history,self.config)
        self.assertEqual(model.targets_.nnz,0)
        self.assertEqual(model.loss_history_[-1]['objective'],0.)
        self.assertEqual(model.training_summary_['target_statistics']['n_values'],0)
        self.assertIsNone(model.training_summary_['target_statistics']['median'])
        json.dumps(model.training_summary_,allow_nan=False)

    def test_common_stage_keeps_binary_relevance_and_masks_low_and_high_history(self):
        history,stage=self.stage()
        model=CustomIALS(rank=2,iterations=2).fit(history,self.config,context=stage.context)
        result=model.decode_items(model.recommend(stage=stage)).ravel().tolist()
        self.assertEqual(len(set(result)),10)
        self.assertNotIn('item-00',result);self.assertNotIn('item-01',result)
        self.assertTrue((stage.ground_truth.data==1).all())
        self.assertEqual(model.evaluate(stage).summary['stage_sha256'],stage.report['stage_sha256'])

    def test_invalid_target_params_or_nonhistorical_timestamps_are_rejected(self):
        for params in ({'gamma':-1},{'gamma':True},{'gamma':np.nan},{'gamma':np.inf},
                       {'half_life_days':0},{'half_life_days':-1},{'half_life_days':-np.inf},
                       {'half_life_days':np.nan},{'half_life_days':True},{'half_life_days':'730'}):
            with self.subTest(params=params),self.assertRaises(ValueError):CustomIALS(**params)
        for ratings,times,cutoff in (([6],pd.to_datetime(['2020-01-01'],utc=True),'2020-01-02T00:00:00+00:00'),
                                     ([5],pd.to_datetime(['2020-01-01']),'2020-01-02T00:00:00+00:00'),
                                     ([5],pd.to_datetime(['2020-01-02'],utc=True),'2020-01-02T00:00:00+00:00'),
                                     ([5],pd.to_datetime(['2020-01-01'],utc=True),'2020-01-02')):
            with self.assertRaises(ValueError):custom_preferences(ratings,times,cutoff)

    def test_target_statistics_include_explicit_zeros(self):
        stats=summarize_targets([0.,.25,.5,1.])
        self.assertEqual(stats['zero_targets'],1)
        self.assertEqual(stats['median'],.375)
        self.assertEqual(stats['mean'],.4375)
        self.assertEqual(stats['n_values'],4)

    def test_control_runner_preserves_history_params_and_publishes_only_aggregates(self):
        history,stage=self.stage()
        self.config['implementation_checks']['ials'].update(rank=2,iterations=2)
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'RECSYS_DATA_DIR':str(Path(directory)/'data')}):
            self.config['storage']['results_dir']=str(Path(directory)/'results')
            models,reports=run_custom_ials_control_checks(history,stage,self.config)
            self.assertEqual(set(models),{'binary','rating_only','recency_only','rating_and_recency'})
            for variant,report in reports.items():
                self.assertFalse(report['test_used']);self.assertFalse(report['hyperparameters_tuned'])
                self.assertEqual(report['training_details']['observed_pattern_sha256'],reports['binary']['training_details']['observed_pattern_sha256'])
                for key in ('rank','alpha','regularization','iterations','init_std'):
                    self.assertEqual(models[variant].params[key],models['binary'].params[key])
            for path in (Path(directory)/'results').iterdir():
                content=path.read_text()
                self.assertNotIn('synthetic-user',content);self.assertNotIn('synthetic-seed',content)
            groups=pd.read_csv(Path(directory)/'results/custom_ials_target_by_rating_age.csv')
            self.assertEqual(groups.n_positive_reviews.sum(),14)

    def test_primary_control_rejects_degenerate_target(self):
        history,stage=self.stage()
        self.config['implementation_checks']['custom_ials']['gamma']=0
        with self.assertRaisesRegex(ValueError,'both rating and finite recency'):
            run_custom_ials_control_checks(history,stage,self.config)


if __name__=='__main__':
    unittest.main()
