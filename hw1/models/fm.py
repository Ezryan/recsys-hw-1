"""Own second-order FM with causal metadata, BCE training and exact block ranking."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from models.base import BaseRecommender
from utils.evaluation import evaluate_scorer
from utils.features import FEATURE_NAMES, USER_FEATURES, FeatureScaler, build_fm_examples


LOGGER = logging.getLogger(__name__)


class FMModule(nn.Module):
    """Two ID fields and optional numerical fields; no dense ID one-hot matrix."""

    def __init__(self, n_users: int, n_items: int, n_numeric: int, rank: int, *, seed: int, init_std: float):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(()))
        self.user_linear = nn.Parameter(torch.zeros(n_users))
        self.item_linear = nn.Parameter(torch.zeros(n_items))
        self.user_factors = nn.Parameter(torch.randn(n_users, rank, generator=generator) * init_std)
        self.item_factors = nn.Parameter(torch.randn(n_items, rank, generator=generator) * init_std)
        self.numeric_linear = nn.Parameter(torch.zeros(n_numeric))
        self.numeric_factors = nn.Parameter(torch.randn(n_numeric, rank, generator=generator) * init_std)

    def forward(self, users: torch.Tensor, items: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        vu, vi = self.user_factors[users], self.item_factors[items]
        summed = vu + vi + numeric @ self.numeric_factors
        squared_terms = vu.square() + vi.square() + numeric.square() @ self.numeric_factors.square()
        interactions = .5 * (summed.square() - squared_terms).sum(dim=1)
        linear = self.bias + self.user_linear[users] + self.item_linear[items] + numeric @ self.numeric_linear
        return linear + interactions

    def penalty(self) -> torch.Tensor:
        """Half sum of squares of all linear/factor weights, excluding global bias."""
        return .5 * sum(value.square().sum() for name, value in self.named_parameters() if name != "bias")

    def collapsed_snapshot(self, user_features: torch.Tensor, item_features: torch.Tensor):
        """Group FM fields into user/item vectors without dropping within-group pairs.

        score(u,i) = bias + a_u + b_i + h_u @ h_i. The two scalar offsets
        include interactions among the fields belonging to the same entity.
        """
        user_width = user_features.shape[1]
        vu, vi = self.numeric_factors[:user_width], self.numeric_factors[user_width:]
        wu, wi = self.numeric_linear[:user_width], self.numeric_linear[user_width:]
        hu = self.user_factors + user_features @ vu
        hi = self.item_factors + item_features @ vi
        self_u = self.user_factors.square() + user_features.square() @ vu.square()
        self_i = self.item_factors.square() + item_features.square() @ vi.square()
        au = self.user_linear + user_features @ wu + .5 * (hu.square() - self_u).sum(dim=1)
        bi = self.item_linear + item_features @ wi + .5 * (hi.square() - self_i).sum(dim=1)
        return hu, hi, au, bi


class FM(BaseRecommender):
    """BCE FM with fixed epochs or explicit validation-only early stopping."""

    def __init__(self, rank: int = 32, learning_rate: float = .003, l2: float = .0001,
                 negatives_per_positive: int = 1, batch_size: int = 1024, epochs: int = 5,
                 use_metadata: bool = True, init_std: float = .01):
        super().__init__(**self._validated_params(rank, learning_rate, l2, negatives_per_positive,
                                                batch_size, epochs, use_metadata, init_std))

    @staticmethod
    def _validated_params(rank, learning_rate, l2, negatives_per_positive, batch_size, epochs, use_metadata, init_std):
        integers = {"rank": rank, "negatives_per_positive": negatives_per_positive, "batch_size": batch_size, "epochs": epochs}
        for name, value in integers.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if name == "negatives_per_positive" else 1):
                raise ValueError(f"{name} must be an integer in its allowed range")
        floats = {"learning_rate": learning_rate, "l2": l2, "init_std": init_std}
        for name, value in floats.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(float(value)) or float(value) < 0 or (name != "l2" and float(value) == 0):
                raise ValueError(f"{name} must be finite and nonnegative (strictly positive for learning rate/init)")
        if not isinstance(use_metadata, (bool, np.bool_)):
            raise ValueError("use_metadata must be boolean")
        return {**{name: int(value) for name, value in integers.items()},
                **{name: float(value) for name, value in floats.items()}, "use_metadata": bool(use_metadata)}

    def fit_with_validation(self, history, config, stage, *, patience=3):
        """Select an epoch using validation; ordinary fit/refit has no held-out input."""
        if stage.period != "validation" or stage.report["protocol"] != config["evaluation"]:
            raise ValueError("FM epoch selection requires the common validation protocol")
        if isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer)) or patience < 1:
            raise ValueError("patience must be a positive integer")
        self._selection_stage = stage
        self._selection_patience = int(patience)
        try:
            return super().fit(history, config, context=stage.context)
        finally:
            self._selection_stage = None

    def _fit(self, history: pd.DataFrame, config):
        self.params = self._validated_params(**self.params)
        if config["runtime"]["device"] != "cpu":
            raise ValueError("This FM implementation currently uses the declared CPU experiment")
        examples = build_fm_examples(history, self.context_, config,
                                     negatives_per_positive=self.params["negatives_per_positive"])
        self.scaler_ = FeatureScaler().fit(examples.raw_features[~examples.sampled])
        self.examples_report_ = examples.report
        self.snapshot_users_raw_ = examples.snapshot_users
        self.snapshot_items_raw_ = examples.snapshot_items
        numeric = self.scaler_.transform(examples.raw_features) if self.params["use_metadata"] else np.empty((len(examples.users), 0), dtype=np.float32)
        users = torch.from_numpy(examples.users)
        items = torch.from_numpy(examples.items)
        labels = torch.from_numpy(examples.labels)
        numeric = torch.from_numpy(numeric)
        # Independent seeded streams keep ID initialization and epoch permutations
        # identical in the controlled metadata/IDs-only comparison.
        self.module_ = FMModule(len(self.context_.user_ids), len(self.context_.item_ids), numeric.shape[1],
                                self.params["rank"], seed=config["seed"], init_std=self.params["init_std"])
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.params["learning_rate"])
        shuffle = torch.Generator(device="cpu").manual_seed(config["seed"] + 1)
        self.loss_history_ = []
        self.validation_history_ = []
        stage = getattr(self, "_selection_stage", None)
        maximum_epochs = self.params["epochs"]
        best_score, best_state, best_epoch, stale = -np.inf, None, None, 0
        self._record_loss(0, users, items, numeric, labels)
        for epoch in range(1, self.params["epochs"] + 1):
            order = torch.randperm(len(labels), generator=shuffle)
            self.module_.train()
            for start in range(0, len(labels), self.params["batch_size"]):
                batch = order[start:start + self.params["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                logits = self.module_(users[batch], items[batch], numeric[batch])
                objective = F.binary_cross_entropy_with_logits(logits, labels[batch]) + self.params["l2"] * self.module_.penalty()
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite FM training objective")
                objective.backward()
                optimizer.step()
            self._record_loss(epoch, users, items, numeric, labels)
            LOGGER.info("FM%s epoch %d/%d: BCE %.6f, objective %.6f",
                        " + metadata" if self.params["use_metadata"] else " IDs only", epoch,
                        self.params["epochs"], self.loss_history_[-1]["bce"], self.loss_history_[-1]["objective"])
            if stage is not None:
                self.module_.eval()
                self._prepare_snapshot()
                start = time.perf_counter()
                result = evaluate_scorer(
                    lambda u, i: self._score(u, i, timestamp=self.context_.cutoff), stage, config,
                )
                score = result.summary["ndcg@10"]
                improved = score > best_score
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = {name: tensor.detach().clone() for name, tensor in self.module_.state_dict().items()}
                else:
                    stale += 1
                self.validation_history_.append({
                    "epoch": epoch, "ndcg@10": score, "recall@10": result.summary["recall@10"],
                    "evaluation_seconds": time.perf_counter() - start, "improved": improved,
                    "best_epoch_so_far": best_epoch, "stage_sha256": result.summary["stage_sha256"],
                })
                LOGGER.info("FM validation epoch %d: NDCG %.6f, best epoch %d, stale %d", epoch, score, best_epoch, stale)
                if stale >= self._selection_patience:
                    break
        self.epochs_completed_ = epoch
        self.selected_epoch_ = best_epoch if stage is not None else epoch
        if stage is not None:
            self.module_.load_state_dict(best_state)
            self.params["epochs"] = self.selected_epoch_
        self.module_.eval()
        self._prepare_snapshot()
        digest = hashlib.sha256()
        for name, value in self.module_.state_dict().items():
            array = value.detach().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError("Nonfinite FM weights")
            digest.update(name.encode())
            digest.update(array.tobytes())
        self.training_summary_ = {
            **examples.report, "feature_scaler": self.scaler_.summary(),
            "use_metadata": self.params["use_metadata"], "numeric_features_used": list(FEATURE_NAMES) if self.params["use_metadata"] else [],
            "epochs_trained": self.epochs_completed_, "selected_epoch": self.selected_epoch_,
            "max_epochs_requested": maximum_epochs,
            "epoch_selection": "validation_ndcg_with_patience" if stage is not None else "fixed_before_implementation_check_no_validation_selection",
            "patience": self._selection_patience if stage is not None else None,
            "optimizer": "Adam, no optimizer weight_decay",
            "objective": "mean BCE over sampled training examples + l2/2 * sum of all non-global-bias squared weights",
            "loss_normalization": "per_example_mean_not_reweighted_to_full_catalog",
            "sampling_interpretation": "unobserved examples are weak negatives; logits are ranking scores, not calibrated population probabilities",
            "parameter_count": sum(p.numel() for p in self.module_.parameters()),
            "weights_storage_bytes": sum(p.numel() * p.element_size() for p in self.module_.parameters()),
            "weights_sha256": digest.hexdigest(), "initial_train_bce": self.loss_history_[0]["bce"],
            "final_train_bce": self.loss_history_[self.selected_epoch_]["bce"],
            "initial_train_objective": self.loss_history_[0]["objective"],
            "final_train_objective": self.loss_history_[self.selected_epoch_]["objective"],
            "ranking": "exact FM grouped into entity vectors and within-entity scalar offsets at history cutoff",
        }

    def _record_loss(self, epoch, users, items, numeric, labels):
        total = 0.
        self.module_.eval()
        with torch.no_grad():
            for start in range(0, len(labels), self.params["batch_size"]):
                stop = start + self.params["batch_size"]
                logits = self.module_(users[start:stop], items[start:stop], numeric[start:stop])
                total += F.binary_cross_entropy_with_logits(logits, labels[start:stop], reduction="sum").item()
            bce = total / len(labels)
            penalty = self.params["l2"] * self.module_.penalty().item()
        if not np.isfinite(bce + penalty):
            raise FloatingPointError("Nonfinite full training loss")
        self.loss_history_.append({"epoch": epoch, "bce": bce, "l2_penalty": penalty, "objective": bce + penalty})

    def _prepare_snapshot(self):
        if self.params["use_metadata"]:
            raw = np.concatenate((self.snapshot_users_raw_, np.zeros((len(self.snapshot_users_raw_), len(FEATURE_NAMES)-len(USER_FEATURES)))), axis=1)
            user_features = self.scaler_.transform(raw)[:, :len(USER_FEATURES)]
            raw = np.concatenate((np.zeros((len(self.snapshot_items_raw_), len(USER_FEATURES))), self.snapshot_items_raw_), axis=1)
            item_features = self.scaler_.transform(raw)[:, len(USER_FEATURES):]
        else:
            user_features = np.empty((len(self.context_.user_ids), 0), dtype=np.float32)
            item_features = np.empty((len(self.context_.item_ids), 0), dtype=np.float32)
        with torch.no_grad():
            hu, hi, au, bi = self.module_.collapsed_snapshot(torch.from_numpy(user_features), torch.from_numpy(item_features))
        # Float64 block dot products stabilize ordering across evaluator batch sizes.
        self.ranking_user_vectors_ = hu.numpy().astype(np.float64)
        self.ranking_item_vectors_ = hi.numpy().astype(np.float64)
        self.ranking_user_offsets_ = au.numpy().astype(np.float64)
        self.ranking_item_offsets_ = bi.numpy().astype(np.float64)
        self.ranking_bias_ = float(self.module_.bias.detach())

    def _score(self, user_indices, item_indices, *, timestamp, features=None):
        if features is not None:
            raise ValueError("FM uses its fitted history snapshot; external feature overrides are unsupported")
        return (self.ranking_user_vectors_[user_indices] @ self.ranking_item_vectors_[item_indices].T
                + self.ranking_user_offsets_[user_indices, None]
                + self.ranking_item_offsets_[None, item_indices] + self.ranking_bias_)

    def save_artifacts(self, directory: str | Path):
        """Only local ignored artifacts; no raw interaction tables in public reports."""
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.module_.state_dict(), directory / "weights.pt")
        np.savez(directory / "feature_transform.npz", mean=self.scaler_.mean_, scale=self.scaler_.scale_,
                 snapshot_users=self.snapshot_users_raw_, snapshot_items=self.snapshot_items_raw_)
