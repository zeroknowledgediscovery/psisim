"""Non-LSM comparison detectors (and one non-LSM generator).

Every detector is fitted on training rows only and returns an anomaly score
per row (larger = less typical), so ROC AUC is computed exactly as for
U_LSM. All take integer-coded matrices (0 = missing) as in veritas_core.

  PairwiseBest    -sum_j ln p(x_j | x_k*), k* = most informative observed partner
  LogRegPL        multinomial logistic-regression pseudo-likelihood (one model per item)
  LatentClass     mixture of independent categoricals (EM); -ln p(x_obs), a joint likelihood;
                  also exposes dists()/dist_item() so it can *generate* conditional replacements
  PCARecon        reconstruction error of one-hot rows from a rank-r PCA
  IForest         isolation forest on one-hot rows
  KNNDistance     mean disagreement with the k most similar training respondents
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

FLOOR = 1e-12


def onehot0(X, K, off):
    """Sparse one-hot (missing = all-zero block)."""
    r, c = np.nonzero(X > 0)
    cols = off[c] + X[r, c] - 1
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, cols)),
                             shape=(X.shape[0], off[-1]))


class _Base:
    def __init__(self, K):
        self.K = list(K)
        self.off = np.concatenate([[0], np.cumsum(self.K)]).astype(int)


# --------------------------------------------------------------------------
class PairwiseBest(_Base):
    name = "pairwise"

    def __init__(self, K, n_partners=5, alpha=1.0):
        super().__init__(K)
        self.n_partners, self.alpha = n_partners, alpha

    def fit(self, X):
        F = onehot0(X, self.K, self.off)
        G = (F.T @ F).toarray()
        d = len(self.K)
        self.prior = []
        for j, k in enumerate(self.K):
            c = np.bincount(X[:, j], minlength=k + 1)[1:] + 0.5
            self.prior.append(c / c.sum())
        mi = np.zeros((d, d))
        for j in range(d):
            sj = slice(self.off[j], self.off[j + 1])
            for k in range(d):
                if k == j:
                    continue
                C = G[sj, self.off[k]:self.off[k + 1]]
                n = C.sum()
                if n < 30:
                    continue
                P = C / n
                pr, pc = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
                nz = P > 0
                # bias-corrected (Miller-Madow) MI so many-level partners don't win by default
                mi[j, k] = (P[nz] * np.log(P[nz] / (pr @ pc)[nz])).sum() \
                    - (nz.sum() - 1) / (2 * n)
        self.partners = np.argsort(-mi, 1)[:, : self.n_partners]
        self.G = G
        return self

    def score(self, X):
        U = np.zeros(X.shape[0])
        for j, k in enumerate(self.K):
            sj = slice(self.off[j], self.off[j + 1])
            rows = np.flatnonzero(X[:, j] > 0)
            if len(rows) == 0:
                continue
            P = np.tile(self.prior[j], (len(rows), 1))
            done = np.zeros(len(rows), bool)
            for k2 in self.partners[j]:
                use = ~done & (X[rows, k2] > 0)
                if use.any():
                    cols = self.off[k2] + X[rows[use], k2] - 1
                    C = self.G[sj][:, cols].T  # (m, K_j)
                    P[use] = (C + self.alpha * self.prior[j]) / (C.sum(1, keepdims=True) + self.alpha)
                    done |= use
            U[rows] -= np.log(np.maximum(P[np.arange(len(rows)), X[rows, j] - 1], FLOOR))
        return U


# --------------------------------------------------------------------------
def _drop_block(F, block):
    """Zero one item's one-hot block (its own answer is not an input)."""
    m = np.ones(F.shape[1], np.float32)
    m[block[0]:block[1]] = 0
    return sparse.csr_matrix(F.multiply(m[None, :]))


def _fit_lr(F, y, block, k, prior, C):
    o = y > 0
    Fj = _drop_block(F[o], block)
    yj = y[o]
    if len(np.unique(yj)) < 2:
        return None, prior
    m = LogisticRegression(C=C, max_iter=300)
    m.fit(Fj, yj)
    return m, prior


class LogRegPL(_Base):
    name = "logreg"

    def __init__(self, K, C=0.05, n_jobs=4):
        super().__init__(K)
        self.C, self.n_jobs = C, n_jobs

    def fit(self, X):
        F = onehot0(X, self.K, self.off)
        priors = [(np.bincount(X[:, j], minlength=k + 1)[1:] + 0.5) /
                  (np.bincount(X[:, j], minlength=k + 1)[1:] + 0.5).sum()
                  for j, k in enumerate(self.K)]
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_lr)(F, X[:, j], (self.off[j], self.off[j + 1]), k, priors[j], self.C)
            for j, k in enumerate(self.K))
        self.models = [r[0] for r in res]
        self.priors = [r[1] for r in res]
        return self

    def _proba(self, j, F):
        k = self.K[j]
        m = self.models[j]
        if m is None:
            return np.tile(self.priors[j], (F.shape[0], 1))
        Fj = _drop_block(F, (self.off[j], self.off[j + 1]))
        P = np.full((F.shape[0], k), FLOOR)
        P[:, m.classes_.astype(int) - 1] = m.predict_proba(Fj)
        return P

    # generator API (veritas_core)
    def dists(self, X):
        F = onehot0(X, self.K, self.off)
        return [self._proba(j, F) for j in range(len(self.K))]

    def dist_item(self, X, j):
        return self._proba(j, onehot0(X, self.K, self.off))

    def score(self, X):
        F = onehot0(X, self.K, self.off)
        U = np.zeros(X.shape[0])
        for j in range(len(self.K)):
            o = np.flatnonzero(X[:, j] > 0)
            if len(o):
                P = self._proba(j, F[o])
                U[o] -= np.log(np.maximum(P[np.arange(len(o)), X[o, j] - 1], FLOOR))
        return U


# --------------------------------------------------------------------------
class LatentClass(_Base):
    """Mixture of independent categoricals, missing coordinates marginalised."""
    name = "latentclass"

    def __init__(self, K, n_classes=20, n_iter=150, alpha=1.0, seed=0):
        super().__init__(K)
        self.C, self.n_iter, self.alpha, self.seed = n_classes, n_iter, alpha, seed

    def _loglik_terms(self, F):
        # F sparse one-hot (n, sumK); logtheta (sumK, C)
        return np.asarray(F @ self.logtheta) + self.logpi[None, :]

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        F = onehot0(X, self.K, self.off)
        n = X.shape[0]
        R = rng.dirichlet(np.ones(self.C), size=n)
        for _ in range(self.n_iter):
            # M step
            self.logpi = np.log(R.mean(0) + 1e-12)
            cnt = np.asarray(F.T @ R)  # (sumK, C)
            th = np.empty_like(cnt)
            for j in range(len(self.K)):
                s = slice(self.off[j], self.off[j + 1])
                c = cnt[s] + self.alpha / self.K[j]
                th[s] = c / c.sum(0, keepdims=True)
            self.logtheta = np.log(th)
            # E step
            L = self._loglik_terms(F)
            R = np.exp(L - logsumexp(L, 1, keepdims=True))
        return self

    def score(self, X):
        L = self._loglik_terms(onehot0(X, self.K, self.off))
        return -logsumexp(L, 1)

    # generator API (veritas_core): phi_j(x_-j) = sum_c p(c | x_-j) theta_jc
    def dist_item(self, X, j):
        Xm = X.copy()
        Xm[:, j] = 0
        L = self._loglik_terms(onehot0(Xm, self.K, self.off))
        post = np.exp(L - logsumexp(L, 1, keepdims=True))
        return post @ np.exp(self.logtheta[self.off[j]:self.off[j + 1]]).T

    def dists(self, X):
        # Selected cells are already masked by the caller (x_-S); for observed
        # j the target's own term must be removed from the posterior.
        F = onehot0(X, self.K, self.off)
        L = self._loglik_terms(F)
        out = []
        for j in range(len(self.K)):
            s = slice(self.off[j], self.off[j + 1])
            own = np.asarray(F[:, s] @ self.logtheta[s])
            Lj = L - own
            post = np.exp(Lj - logsumexp(Lj, 1, keepdims=True))
            out.append(post @ np.exp(self.logtheta[s]).T)
        return out


# --------------------------------------------------------------------------
class PCARecon(_Base):
    name = "pca"

    def __init__(self, K, rank=50):
        super().__init__(K)
        self.rank = rank

    def fit(self, X):
        self.p = PCA(self.rank, random_state=0).fit(onehot0(X, self.K, self.off).toarray())
        return self

    def score(self, X):
        F = onehot0(X, self.K, self.off).toarray()
        R = self.p.inverse_transform(self.p.transform(F))
        # only observed blocks count, so masks don't add error
        return ((F - R) ** 2 * (np.repeat(X > 0, self.K, axis=1))).sum(1)


class IForest(_Base):
    name = "iforest"

    def fit(self, X):
        self.m = IsolationForest(n_estimators=300, random_state=0, n_jobs=4)
        self.m.fit(onehot0(X, self.K, self.off))
        return self

    def score(self, X):
        return -self.m.score_samples(onehot0(X, self.K, self.off))


class KNNDistance(_Base):
    name = "knn"

    def __init__(self, K, k=10):
        super().__init__(K)
        self.k = k

    def fit(self, X):
        self.T = X
        self.Fo = onehot0(X, self.K, self.off)
        self.Oo = sparse.csr_matrix((X > 0).astype(np.float32))
        return self

    def score(self, X):
        F = onehot0(X, self.K, self.off)
        agree = np.asarray((F @ self.Fo.T).todense())
        co = np.asarray((sparse.csr_matrix((X > 0).astype(np.float32)) @ self.Oo.T).todense())
        dist = 1.0 - agree / np.maximum(co, 1)
        return np.sort(dist, 1)[:, : self.k].mean(1)
