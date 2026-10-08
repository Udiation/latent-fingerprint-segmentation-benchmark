"""Classical block classifiers: SVM, Random Forest, Weighted ELM.

All three get the same feature vector (segbench.features), so the table compares
classifiers on equal footing. These are re-implementations in the spirit of the
cited papers, not exact reproductions - say so in the manuscript.

SVM: an exact RBF-SVC does not scale to millions of blocks, so the RBF kernel is
approximated with Nystroem features followed by a linear SVM (state this in the paper).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class WeightedELM(BaseEstimator, ClassifierMixin):
    """Weighted Extreme Learning Machine (Zong, Huang & Chen 2013), the classifier
    used by Ezeobiejesi & Bhanu (2016). Weighting scheme W1: each sample weighted
    1 / (number of samples in its class), which handles fg/bg imbalance.
    beta = (I/C + H^T W H)^-1 H^T W T, accumulated in chunks to bound memory."""

    def __init__(self, n_hidden=1000, C=1.0, random_state=0, chunk=50000):
        self.n_hidden = n_hidden
        self.C = C
        self.random_state = random_state
        self.chunk = chunk

    def _hidden(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w_ + self.b_)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(bool)
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.uniform(-1.0, 1.0, size=(X.shape[1], self.n_hidden))
        self.b_ = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        n_pos, n_neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
        weights = np.where(y, 1.0 / n_pos, 1.0 / n_neg)
        targets = np.where(y, 1.0, -1.0)
        hth = np.eye(self.n_hidden) / self.C
        htt = np.zeros(self.n_hidden)
        for i in range(0, len(X), self.chunk):
            h = self._hidden(X[i:i + self.chunk])
            w = weights[i:i + self.chunk]
            hth += h.T @ (h * w[:, None])
            htt += h.T @ (w * targets[i:i + self.chunk])
        self.beta_ = np.linalg.solve(hth, htt)
        self.classes_ = np.array([False, True])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate([self._hidden(X[i:i + self.chunk]) @ self.beta_
                               for i in range(0, len(X), self.chunk)]) if len(X) else np.zeros(0)

    def predict(self, X):
        return self.decision_function(X) > 0


def build(method, params, seed, n_jobs):
    if method == "svm":
        return make_pipeline(
            StandardScaler(),
            Nystroem(kernel="rbf", gamma=params["gamma"], n_components=params.get("n_components", 1000),
                     random_state=seed),
            LinearSVC(C=params["C"], class_weight="balanced", random_state=seed, max_iter=5000))
    if method == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200), max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], class_weight="balanced",
            n_jobs=n_jobs, random_state=seed)
    if method == "welm":
        return make_pipeline(StandardScaler(),
                             WeightedELM(n_hidden=params["n_hidden"], C=params["C"], random_state=seed))
    raise ValueError(f"unknown method {method}")


PARAM_GRIDS = {
    "svm": [{"gamma": g, "C": c} for g in (0.05, 0.2) for c in (0.1, 1.0, 10.0)],
    "rf": [{"max_depth": d, "min_samples_leaf": m} for d in (None, 20) for m in (1, 5)],
    "welm": [{"n_hidden": h, "C": c} for h in (500, 1000) for c in (1.0, 1e2, 1e4, 1e6)],
}

DISPLAY = {
    "svm": "Support Vector Machine",
    "rf": "Random decision forest",
    "welm": "Weighted extreme learning machine",
}


def set_single_thread(model):
    """Use one thread at inference so ms/img is comparable across methods."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_jobs"):
        est.set_params(n_jobs=1)
    return model


def fitted_scalar_count(model):
    """Count scalar values in the fitted inference state.

    Neural-network parameter counts do not have a perfect analogue for tree
    ensembles or random-feature classifiers.  For the paper's required
    ``Params (M)`` column we therefore use a reproducible storage-oriented
    definition for the classical rows: numeric scalar values required by the
    fitted estimator at inference.  For pipelines this includes scaler state,
    the Nyström transform, and the final classifier; for a forest it includes
    every scalar field of each stored node record and its node values.  Python
    object metadata and duplicated derived arrays are excluded.
    """

    def count_estimator(estimator):
        if isinstance(estimator, RandomForestClassifier):
            total = 0
            for tree_estimator in estimator.estimators_:
                tree = tree_estimator.tree_
                state = tree.__getstate__()
                nodes = state["nodes"]
                # Count every distinct scalar field in the stored node record
                # once, plus the separate class-value tensor.
                total += sum(int(nodes[name].size) for name in nodes.dtype.names)
                total += int(state["values"].size)
            return total
        if isinstance(estimator, StandardScaler):
            return sum(
                int(getattr(estimator, name).size)
                for name in ("mean_", "scale_")
                if hasattr(estimator, name)
            )
        if isinstance(estimator, Nystroem):
            return sum(
                int(getattr(estimator, name).size)
                for name in ("components_", "normalization_")
                if hasattr(estimator, name)
            )
        if isinstance(estimator, LinearSVC):
            return sum(
                int(getattr(estimator, name).size)
                for name in ("coef_", "intercept_")
                if hasattr(estimator, name)
            )
        if isinstance(estimator, WeightedELM):
            return sum(
                int(getattr(estimator, name).size)
                for name in ("w_", "b_", "beta_")
                if hasattr(estimator, name)
            )
        raise TypeError(
            f"no fitted-scalar counting rule for {type(estimator).__name__}"
        )

    if hasattr(model, "steps"):
        return sum(count_estimator(estimator) for _, estimator in model.steps)
    return count_estimator(model)
