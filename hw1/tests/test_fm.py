"""Pair-sum and gradient references, exact snapshot scoring, and controlled FM fits."""
from copy import deepcopy
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from models import FM
from models.fm import FMModule
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config
from test_features import toy_history


class FMTests(unittest.TestCase):
    def setUp(self):
        self.config=deepcopy(load_config())
        self.config['split'].update(validation_start_utc='2020-01-06T00:00:00+00:00', test_start_utc='2020-01-10T00:00:00+00:00')
        self.history=toy_history()
        # Small tests do not need a global runtime configuration or many BLAS threads.
        self.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.previous_threads)

    def test_fast_fm_and_all_weight_gradients_match_explicit_unordered_pair_sum(self):
        module=FMModule(2,3,3,2,seed=42,init_std=.2).double()
        with torch.no_grad():
            module.bias.fill_(.3)
            module.user_linear.copy_(torch.tensor([.1,-.2]))
            module.item_linear.copy_(torch.tensor([.4,-.5,.6]))
            module.numeric_linear.copy_(torch.tensor([.3,.2,-.1]))
        users=torch.tensor([0,1,0]);items=torch.tensor([1,2,0])
        numeric=torch.tensor([[.5,0.,-1.],[1.,2.,.3],[-.2,.4,1.]],dtype=torch.float64)
        actual=module(users,items,numeric)
        z=torch.cat((torch.nn.functional.one_hot(users,2),torch.nn.functional.one_hot(items,3),numeric),dim=1)
        linear=torch.cat((module.user_linear,module.item_linear,module.numeric_linear))
        factors=torch.cat((module.user_factors,module.item_factors,module.numeric_factors),dim=0)
        expected=module.bias+z@linear
        for j in range(z.shape[1]):
            for k in range(j+1,z.shape[1]):
                expected=expected+(factors[j]*factors[k]).sum()*z[:,j]*z[:,k]
        np.testing.assert_allclose(actual.detach(),expected.detach(),atol=1e-14)
        actual.sum().backward()
        gradients={name:p.grad.clone() for name,p in module.named_parameters()}
        module.zero_grad()
        expected.sum().backward()
        for name,p in module.named_parameters():
            torch.testing.assert_close(p.grad,gradients[name],atol=1e-13,rtol=1e-13)

    def test_analytic_gradients_pass_finite_difference_check_for_numeric_fields(self):
        module=FMModule(2,3,2,3,seed=2,init_std=.3).double()
        users,items=torch.tensor([0,1]),torch.tensor([1,2])
        numeric=torch.tensor([[.3,-.2],[.8,.4]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:module(users,items,x),(numeric,)))

    def test_collapsed_snapshot_preserves_within_entity_and_cross_entity_interactions(self):
        for n_user_fields,n_item_fields in ((4,5),(0,0)):
            module=FMModule(3,4,n_user_fields+n_item_fields,3,seed=9,init_std=.3).double()
            generator=torch.Generator().manual_seed(3)
            zu=torch.randn(3,n_user_fields,generator=generator,dtype=torch.float64)
            zi=torch.randn(4,n_item_fields,generator=generator,dtype=torch.float64)
            hu,hi,au,bi=module.collapsed_snapshot(zu,zi)
            collapsed=module.bias+au[:,None]+bi[None,:]+hu@hi.T
            users=torch.arange(3).repeat_interleave(4);items=torch.arange(4).repeat(3)
            direct=module(users,items,torch.cat((zu[users],zi[items]),dim=1)).reshape(3,4)
            torch.testing.assert_close(collapsed,direct,atol=1e-13,rtol=1e-13)

    def test_l2_is_half_global_squared_weight_sum_excluding_global_bias(self):
        module=FMModule(2,3,2,2,seed=2,init_std=.1)
        with torch.no_grad():
            for p in module.parameters():p.fill_(2.)
        count=sum(p.numel() for name,p in module.named_parameters() if name!='bias')
        self.assertEqual(module.penalty().item(),2*count)
        with torch.no_grad():module.bias.fill_(1000.)
        self.assertEqual(module.penalty().item(),2*count)

    def test_fixed_training_reduces_full_sampled_bce_and_objective(self):
        model=FM(rank=3,epochs=20,batch_size=64,learning_rate=.03,l2=1e-5).fit(self.history,self.config)
        self.assertLess(model.loss_history_[-1]['bce'],model.loss_history_[0]['bce'])
        self.assertLess(model.loss_history_[-1]['objective'],model.loss_history_[0]['objective'])
        self.assertEqual(len(model.loss_history_),21)
        self.assertEqual(model.training_summary_['numeric_features_used'],list(model.scaler_.summary()['feature_names']))
        self.assertEqual(len(model.training_summary_['numeric_features_used']),9)

    def test_permuted_history_reproduces_weights_loss_and_samples(self):
        first=FM(rank=2,epochs=3,batch_size=8).fit(self.history,self.config)
        second=FM(rank=2,epochs=3,batch_size=8).fit(self.history.sample(frac=1,random_state=4),self.config)
        self.assertEqual(first.loss_history_,second.loss_history_)
        self.assertEqual(first.training_summary_['weights_sha256'],second.training_summary_['weights_sha256'])
        self.assertEqual(first.examples_report_,second.examples_report_)

    def test_metadata_control_uses_identical_examples_scaling_and_initial_id_weights(self):
        first=FM(rank=2,epochs=2).fit(self.history,self.config)
        second=FM(rank=2,epochs=2,use_metadata=False).fit(self.history,self.config)
        self.assertEqual(first.examples_report_,second.examples_report_)
        np.testing.assert_array_equal(first.scaler_.mean_,second.scaler_.mean_)
        self.assertEqual(second.module_.numeric_factors.shape,(0,2))
        state=torch.random.get_rng_state().clone()
        a=FMModule(3,4,9,2,seed=42,init_std=.1)
        b=FMModule(3,4,0,2,seed=42,init_std=.1)
        torch.testing.assert_close(a.user_factors,b.user_factors,rtol=0,atol=0)
        torch.testing.assert_close(a.item_factors,b.item_factors,rtol=0,atol=0)
        torch.testing.assert_close(state,torch.random.get_rng_state(),rtol=0,atol=0)

    def test_fitted_score_matches_direct_forward_on_cutoff_metadata(self):
        for use_metadata in (True,False):
            model=FM(rank=2,epochs=3,use_metadata=use_metadata).fit(self.history,self.config)
            u=np.array([2,0,1]);i=np.array([4,2,0,3])
            users=np.repeat(u,len(i));items=np.tile(i,len(u))
            raw=np.column_stack((model.snapshot_users_raw_[users],model.snapshot_items_raw_[items]))
            features=model.scaler_.transform(raw) if use_metadata else np.empty((len(users),0),dtype=np.float32)
            with torch.no_grad():direct=model.module_(torch.from_numpy(users),torch.from_numpy(items),torch.from_numpy(features)).numpy().reshape(len(u),len(i))
            np.testing.assert_allclose(model.score(u,i),direct,atol=1e-6,rtol=1e-6)

    def test_common_stage_masks_known_items_and_batches_preserve_ranking(self):
        rows=[('A','item-00',5),('A','item-01',1)]+[('seed',f'item-{i:02d}',5) for i in range(13)]
        history=pd.DataFrame([{'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,'timestamp':pd.Timestamp('2020-01-01',tz='UTC')} for j,(u,i,r) in enumerate(rows)])
        future=pd.DataFrame([{'user_id':'A','item_id':'item-02','rating':5.,'source_line':100,'timestamp':pd.Timestamp('2020-01-06',tz='UTC')}])
        stage=build_evaluation_stage({'train':history,'validation':future},self.config)
        model=FM(rank=2,epochs=2).fit(history,self.config,context=stage.context)
        result=model.recommend(stage=stage,user_batch_size=1,item_batch_size=1)
        np.testing.assert_array_equal(result,model.recommend(stage=stage,user_batch_size=8,item_batch_size=20))
        recommended=model.decode_items(result).ravel().tolist()
        self.assertEqual(len(set(recommended)),10)
        self.assertNotIn('item-00',recommended);self.assertNotIn('item-01',recommended)
        self.assertEqual(model.evaluate(stage).summary['stage_sha256'],stage.report['stage_sha256'])

    def test_saved_weights_and_feature_transform_match_fitted_model(self):
        model=FM(rank=2,epochs=2).fit(self.history,self.config)
        with tempfile.TemporaryDirectory() as directory:
            model.save_artifacts(directory)
            from pathlib import Path
            weights=torch.load(Path(directory)/'weights.pt',weights_only=True)
            for name,p in model.module_.state_dict().items():torch.testing.assert_close(weights[name],p,rtol=0,atol=0)
            with np.load(Path(directory)/'feature_transform.npz') as transform:
                np.testing.assert_array_equal(transform['mean'],model.scaler_.mean_)
                np.testing.assert_array_equal(transform['snapshot_items'],model.snapshot_items_raw_)

    def test_invalid_params_and_external_features_are_rejected(self):
        for params in ({'rank':0},{'rank':True},{'epochs':0},{'batch_size':1.5},{'learning_rate':0},
                       {'l2':-1},{'l2':np.inf},{'init_std':0},{'negatives_per_positive':-1},{'use_metadata':1}):
            with self.subTest(params=params),self.assertRaises(ValueError):FM(**params)
        model=FM(rank=2,epochs=1).fit(self.history,self.config)
        with self.assertRaises(ValueError):model.score([0],[0],features=np.zeros((1,9)))


if __name__=='__main__':
    unittest.main()
