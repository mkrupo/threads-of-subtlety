"""Review classification from a normalized, relation-weighted mean of EDU pairs."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PairGroups:
    """Mean pair features and counts for each observed review and relation."""

    reviews: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    review_count: int
    label_count: int

    def subset(self, indices):
        lookup = np.full(self.review_count, -1, dtype=np.int64)
        lookup[indices] = np.arange(len(indices))
        selected = lookup[self.reviews] >= 0
        return PairGroups(lookup[self.reviews[selected]], self.labels[selected],
                          self.counts[selected], self.means[selected], len(indices), self.label_count)

    def weighted_mean(self, gates):
        weights = self.counts * gates[self.labels]
        totals = np.bincount(self.reviews, weights=weights, minlength=self.review_count)
        fractions = weights / totals[self.reviews]
        pooled = np.zeros((self.review_count, self.means.shape[1]))
        # Groups are contiguous by review. Their review IDs need not be sorted after subsetting.
        if len(weights):
            starts = np.r_[0, np.flatnonzero(np.diff(self.reviews)) + 1]
            pooled[self.reviews[starts]] = np.add.reduceat(fractions[:, None] * self.means, starts, axis=0)
        return pooled, totals


def objective(parameters, base, pairs, scaler, truth, regularization, use_labels):
    intercept, base_weights = parameters[0], parameters[1:1 + base.shape[1]]
    pair_end = 1 + base.shape[1] + pairs.means.shape[1]
    pair_weights = parameters[1 + base.shape[1]:pair_end] / scaler.scale_
    # Positive, smooth gates start at the same value for every label.
    log_gates = parameters[pair_end:]
    gates = np.logaddexp(0, log_gates) if use_labels else np.ones(pairs.label_count)
    pooled, totals = pairs.weighted_mean(gates)
    logits = intercept + base @ base_weights + (pooled - scaler.mean_) @ pair_weights
    errors = expit(logits) - truth
    loss = np.sum(np.logaddexp(0, logits) - truth * logits)
    gradient = [np.array([errors.sum()]), base.T @ errors, scaler.transform(pooled).T @ errors]
    if use_labels:
        pair_signal = pairs.means @ pair_weights
        review_signal = pooled @ pair_weights
        sensitivity = (errors[pairs.reviews] * pairs.counts / totals[pairs.reviews]
                       * (pair_signal - review_signal[pairs.reviews]))
        gradient.append(expit(log_gates) * np.bincount(
            pairs.labels, weights=sensitivity, minlength=pairs.label_count,
        ))
    gradient = np.concatenate(gradient)
    # Use the same L2 penalty for weights and gates, with an unpenalized intercept.
    loss += 0.5 * regularization * (parameters[1:] @ parameters[1:])
    gradient[1:] += regularization * parameters[1:]
    return loss, gradient


def fit_predict(train_base, train_pairs, truth, test_base, test_pairs, *, use_labels, c):
    base_scaler = StandardScaler().fit(train_base)
    base_train = base_scaler.transform(train_base)
    pair_dim = train_pairs.means.shape[1]
    pair_scaler = StandardScaler().fit(train_pairs.weighted_mean(np.ones(train_pairs.label_count))[0])
    gate_count = train_pairs.label_count if use_labels else 0
    start = np.zeros(1 + train_base.shape[1] + pair_dim + gate_count)
    fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                   args=(base_train, train_pairs, pair_scaler, truth, 1 / c, use_labels),
                   options={"maxiter": 300, "ftol": 1e-10})
    if not fit.success:
        raise RuntimeError(f"Relation-weighted fit failed: {fit.message}")

    base_end = 1 + train_base.shape[1]
    gates = np.logaddexp(0, fit.x[base_end + pair_dim:]) if use_labels else np.ones(test_pairs.label_count)
    grouped, _ = test_pairs.weighted_mean(gates)
    logits = (fit.x[0] + base_scaler.transform(test_base) @ fit.x[1:base_end]
              + pair_scaler.transform(grouped) @ fit.x[base_end:base_end + pair_dim])
    return expit(logits), gates, fit.nit
