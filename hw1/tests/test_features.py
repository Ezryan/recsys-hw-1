"""Prefix-oracle checks for observed/negative features and causal sampling."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from utils.evaluation import build_history_context
from utils.features import FEATURE_NAMES, FeatureScaler, HistoricalFeatureState, build_fm_examples
from utils.reproducibility import load_config


def toy_history():
    rows = [(0,"s","a",5),(0,"s","b",5),(0,"s","c",1),
            (1,"u","a",4),(1,"u","d",1),(1,"v","b",5),
            (2,"u","b",5),(2,"v","c",4),(2,"w","e",5),
            (3,"u","e",5),(3,"v","d",1)]
    return pd.DataFrame([{'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,
                          'timestamp':pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=t)}
                         for j,(t,u,i,r) in enumerate(rows)])


def prefix_features(prefix, user, item, time):
    result = []
    for entity, value in (("user_id",user),("item_id",item)):
        selected = prefix.loc[prefix[entity].eq(value)]
        n = len(selected)
        values = [np.log1p(n), selected.rating.mean() if n else 3., selected.rating.ge(4).mean() if n else 0.]
        if entity == 'item_id':
            values.append(np.log1p((time-selected.timestamp.min()).total_seconds()/86400) if n else 0.)
        result.extend(values+[float(n==0)])
    return np.array(result)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config=deepcopy(load_config())
        self.config['split']['validation_start_utc']='2020-01-06T00:00:00+00:00'
        self.history=toy_history()
        self.context=build_history_context(self.history,self.config['split']['validation_start_utc'],self.config)

    def build(self, q=2, history=None):
        return build_fm_examples(self.history if history is None else history,self.context,self.config,negatives_per_positive=q)

    def test_every_observed_and_sampled_row_matches_strict_prefix_oracle(self):
        data=self.build()
        for u,i,t,features in zip(data.users,data.items,data.timestamps_ns,data.raw_features):
            time=pd.Timestamp(t,tz='UTC')
            prefix=self.history.loc[self.history.timestamp.lt(time)]
            expected=prefix_features(prefix,self.context.user_ids[u],self.context.item_ids[i],time)
            np.testing.assert_allclose(features,expected,atol=2e-7)

    def test_samples_use_only_prior_positive_catalog_and_exclude_all_prior_seen(self):
        data=self.build()
        for u,i,t,line in zip(data.users[data.sampled],data.items[data.sampled],data.timestamps_ns[data.sampled],data.anchor_source_lines[data.sampled]):
            time=pd.Timestamp(t,tz='UTC')
            prefix=self.history.loc[self.history.timestamp.lt(time)]
            item,user=self.context.item_ids[i],self.context.user_ids[u]
            self.assertIn(item,set(prefix.loc[prefix.rating.ge(4),'item_id']))
            self.assertNotIn(item,set(prefix.loc[prefix.user_id.eq(user),'item_id']))
            anchor=self.history.set_index('source_line').loc[line]
            self.assertNotEqual(item,anchor.item_id)
            self.assertEqual(user,anchor.user_id)
            self.assertEqual(time,anchor.timestamp)
            self.assertGreaterEqual(anchor.rating,4)
        for line in np.unique(data.anchor_source_lines[data.sampled]):
            values=data.items[data.sampled & (data.anchor_source_lines==line)]
            self.assertEqual(len(values),len(set(values)))

    def test_future_positive_is_allowed_as_a_sampled_negative(self):
        data=self.build()
        # u's first positive at day 1 has only b as an eligible candidate;
        # u reviews b positively the next day. Looking ahead must not exclude it.
        anchor=4
        self.assertEqual([self.context.item_ids[i] for i in data.items[data.sampled & (data.anchor_source_lines==anchor)]],['b'])

    def test_prior_low_rating_is_excluded_even_when_item_is_in_positive_catalog(self):
        history=pd.DataFrame([
            {'user_id':u,'item_id':i,'rating':float(r),'source_line':j+1,
             'timestamp':pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=day)}
            for j,(day,u,i,r) in enumerate([(0,'seed','a',5),(0,'seed','b',5),(1,'u','a',1),(2,'u','c',5)])
        ])
        context=build_history_context(history,self.context.cutoff,self.config)
        data=build_fm_examples(history,context,self.config,negatives_per_positive=3)
        sampled=[context.item_ids[i] for i in data.items[data.sampled & (data.anchor_source_lines==4)]]
        self.assertEqual(sampled,['b'])

    def test_current_and_same_time_ratings_do_not_change_raw_prefix_features(self):
        original=self.build(q=0)
        changed=self.history.copy()
        changed.loc[changed.timestamp.ge(pd.Timestamp('2020-01-03',tz='UTC')),'rating']=1.
        context=build_history_context(changed,self.context.cutoff,self.config)
        updated=build_fm_examples(changed,context,self.config,negatives_per_positive=0)
        mask=original.timestamps_ns <= pd.Timestamp('2020-01-03',tz='UTC').value
        np.testing.assert_array_equal(original.raw_features[mask],updated.raw_features[mask])

    def test_shortage_counts_and_first_timestamp_empty_catalog_are_reported(self):
        data=self.build(q=3)
        self.assertEqual(data.report['requested_sampled_negatives'],8*3)
        self.assertEqual(data.report['anchors_without_prior_catalog'],2)
        self.assertGreater(data.report['anchors_with_negative_shortage'],0)
        self.assertLess(data.report['n_sampled_negatives'],data.report['requested_sampled_negatives'])
        self.assertFalse(data.report['future_pairs_consulted'])
        self.assertFalse(data.sampled[data.timestamps_ns==self.history.timestamp.min().value].any())

    def test_observed_labels_include_low_ratings_and_samples_are_zero(self):
        data=self.build()
        np.testing.assert_array_equal(data.labels[~data.sampled],self.history.rating.ge(4).astype(np.float32))
        self.assertEqual(data.report['n_observed_negative'],3)
        self.assertTrue((data.labels[data.sampled]==0).all())

    def test_seed_and_input_permutation_preserve_examples_exactly(self):
        first=self.build()
        second=self.build(history=self.history.sample(frac=1,random_state=9))
        for field in ('users','items','labels','raw_features','timestamps_ns','anchor_source_lines','sampled'):
            np.testing.assert_array_equal(getattr(first,field),getattr(second,field))
        self.assertEqual(first.report,second.report)

    def test_snapshot_matches_history_only_oracle_at_fitted_cutoff(self):
        data=self.build()
        for u,user in enumerate(self.context.user_ids):
            for i,item in enumerate(self.context.item_ids):
                expected=prefix_features(self.history,user,item,self.context.cutoff)
                np.testing.assert_allclose(np.r_[data.snapshot_users[u],data.snapshot_items[i]],expected,atol=2e-7)

    def test_scaler_fits_only_observed_events_and_keeps_indicators_binary(self):
        data=self.build()
        scaler=FeatureScaler().fit(data.raw_features[~data.sampled])
        other=self.build(q=0)
        reference=FeatureScaler().fit(other.raw_features)
        np.testing.assert_array_equal(scaler.mean_,reference.mean_)
        np.testing.assert_array_equal(scaler.scale_,reference.scale_)
        x=scaler.transform(data.raw_features[~data.sampled])
        for j,name in enumerate(FEATURE_NAMES):
            if name.endswith('.empty_history'):
                np.testing.assert_array_equal(x[:,j],data.raw_features[~data.sampled,j])
            else:
                self.assertAlmostEqual(float(x[:,j].mean()),0.,places=6)
        before=scaler.mean_.copy()
        scaler.transform(np.full((1,9),10000.))
        np.testing.assert_array_equal(scaler.mean_,before)

    def test_scaler_constant_columns_stay_finite_and_bad_inputs_are_rejected(self):
        scaler=FeatureScaler().fit(np.zeros((2,9)))
        np.testing.assert_array_equal(scaler.scale_,np.ones(9))
        self.assertTrue(np.isfinite(scaler.transform(np.ones((1,9)))).all())
        with self.assertRaises(RuntimeError): FeatureScaler().transform(np.zeros((1,9)))
        for x in (np.zeros((0,9)),np.zeros((2,8)),np.full((1,9),np.nan)):
            with self.assertRaises(ValueError): FeatureScaler().fit(x)

    def test_nonincreasing_group_updates_and_queries_are_rejected(self):
        state=HistoricalFeatureState(1,1,self.config)
        state.update([0],[0],[5.],100)
        with self.assertRaises(ValueError): state.query([0],[0],100)
        with self.assertRaises(ValueError): state.update([0],[0],[5.],100)

    def test_mismatched_history_and_invalid_sampling_counts_are_rejected(self):
        changed=self.history.copy();changed.loc[0,'rating']=1.
        with self.assertRaises(ValueError): self.build(history=changed)
        for q in (-1,1.5,True):
            with self.assertRaises(ValueError): self.build(q=q)

    def test_refit_snapshot_recomputes_from_combined_history_at_later_cutoff(self):
        data=self.build()
        additional=pd.DataFrame([{'user_id':'u','item_id':'f','rating':2.,'source_line':100,
                                  'timestamp':pd.Timestamp('2020-01-07',tz='UTC')}])
        combined=pd.concat([self.history,additional],ignore_index=True)
        context=build_history_context(combined,'2020-01-10T00:00:00+00:00',self.config)
        later=build_fm_examples(combined,context,self.config,negatives_per_positive=2)
        user=context.user_ids.index('u')
        self.assertGreater(later.snapshot_users[user,0],data.snapshot_users[user,0])
        for u,uid in enumerate(context.user_ids):
            for i,iid in enumerate(context.item_ids):
                expected=prefix_features(combined,uid,iid,context.cutoff)
                np.testing.assert_allclose(np.r_[later.snapshot_users[u],later.snapshot_items[i]],expected,atol=2e-7)
        before=later.timestamps_ns<self.context.cutoff.value
        np.testing.assert_array_equal(later.raw_features[before],data.raw_features)
        first=FeatureScaler().fit(data.raw_features[~data.sampled])
        refit=FeatureScaler().fit(later.raw_features[~later.sampled])
        self.assertFalse(np.array_equal(first.mean_,refit.mean_))


if __name__=='__main__':
    unittest.main()
