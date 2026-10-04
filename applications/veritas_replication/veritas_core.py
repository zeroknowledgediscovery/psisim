"""Model-agnostic Veritas experiment core.

Rows are integer-coded matrices ``X`` (n x d, int32): 0 = missing, k >= 1 is
the k-th category (1-based) of that item. A model adapter supplies

    model.K                      list[int], categories per item
    model.score(X) -> U          (n,) Veritas potential in nats, -sum ln phi
    model.dists(X) -> list[P_j]  P_j (n, K_j): phi_j(x_-j) for every row
    model.dist_item(X, j) -> P   (n, K_j): phi_j for one item

Corruption mechanisms keep every row's observed/missing mask unchanged.
"""
from __future__ import annotations

import numpy as np

PROB_FLOOR = 1e-12
MECH_IDS = {"permutation": 1, "conditional": 2, "sequential": 3,
            "uniform": 4, "hotdeck": 5}


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
def marginal_table(Xtrain: np.ndarray, K: list[int], alpha: float = 0.5):
    """Training-only empirical marginals m_i(a) with light additive smoothing."""
    tabs = []
    for j, k in enumerate(K):
        c = np.bincount(Xtrain[:, j], minlength=k + 1)[1:].astype(float) + alpha
        tabs.append(c / c.sum())
    return tabs


def marginal_score(X: np.ndarray, tabs) -> np.ndarray:
    U = np.zeros(X.shape[0])
    for j, t in enumerate(tabs):
        x = X[:, j]
        o = x > 0
        U[o] -= np.log(np.maximum(t[x[o] - 1], PROB_FLOOR))
    return U


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """ROC AUC with positives = corrupted rows (Mann-Whitney, ties = 1/2)."""
    s = np.concatenate([neg, pos])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sample_rows(P: np.ndarray, rng) -> np.ndarray:
    """One categorical draw per row from P (n, K); returns 1-based codes."""
    c = P.cumsum(1)
    c /= c[:, -1:]
    u = rng.random((P.shape[0], 1))
    return (u > c).sum(1) + 1


# --------------------------------------------------------------------------
# corruption mechanisms
# --------------------------------------------------------------------------
def marginal_permutation(X, rho, rng, tries: int = 5):
    """Per item: permute observed values among ~rho of the observed rows.

    Every item's held-out histogram is preserved exactly. Of `tries` random
    permutations the one with the most actual token changes is kept.
    """
    Y = X.copy()
    for j in range(X.shape[1]):
        obs = np.flatnonzero(X[:, j] > 0)
        m = int(round(rho * len(obs)))
        if m < 2:
            continue
        sel = rng.choice(obs, m, replace=False)
        vals = X[sel, j]
        best, bc = None, -1
        for _ in range(tries):
            p = rng.permutation(m)
            ch = int((vals[p] != vals).sum())
            if ch > bc:
                best, bc = p, ch
        Y[sel, j] = vals[best]
    return Y


def select_cells(X, rho, rng, items=None):
    """Boolean mask of ~rho of each row's observed (learned) cells."""
    O = X > 0
    if items is not None:
        keep = np.zeros(X.shape[1], bool)
        keep[items] = True
        O &= keep[None, :]
    S = np.zeros_like(O)
    for r in range(X.shape[0]):
        obs = np.flatnonzero(O[r])
        m = int(round(rho * len(obs)))
        if m > 0:
            S[r, rng.choice(obs, m, replace=False)] = True
    return S


def conditional_independent(model, X, S, rng):
    """x'_j ~ phi_j(x_-S) independently for every selected j (README 4.2)."""
    Xm = np.where(S, 0, X)
    P = model.dists(Xm)
    Y = X.copy()
    for j in np.flatnonzero(S.any(0)):
        r = np.flatnonzero(S[:, j])
        Y[r, j] = sample_rows(P[j][r], rng)
    return Y


def conditional_sequential(model, X, S, rng):
    """Sequential completion (README 4.3) in a random item order per repeat.

    Selected cells are masked, then filled one item at a time; later draws
    condition on earlier generated values. All rows share one random item
    order per call, which lets each step be evaluated as a batch.
    """
    Y = np.where(S, 0, X)
    for j in rng.permutation(X.shape[1]):
        r = np.flatnonzero(S[:, j])
        if len(r) == 0:
            continue
        P = model.dist_item(Y[r], j)
        Y[r, j] = sample_rows(P, rng)
    return Y


def uniform_replacement(X, S, K, rng):
    """Positive control: selected cells replaced uniformly over the alphabet."""
    Y = X.copy()
    for j in np.flatnonzero(S.any(0)):
        r = np.flatnonzero(S[:, j])
        Y[r, j] = rng.integers(1, K[j] + 1, size=len(r))
    return Y


def hotdeck_replacement(X, S, donors, rng, k: int = 10):
    """Model-free structural control: selected cells take a real donor's answers.

    For each row, a donor is drawn from its k nearest training respondents
    (Hamming agreement on the row's unselected observed cells); every selected
    cell takes that one donor's value where the donor answered it, otherwise a
    value from another of the k neighbours, otherwise it is left unchanged.
    No LSM is used, so values follow genuine human cross-item structure.
    """
    Y = X.copy()
    for r in range(X.shape[0]):
        cols = np.flatnonzero(S[r])
        if len(cols) == 0:
            continue
        ctx = (X[r] > 0) & ~S[r]
        agree = (donors[:, ctx] == X[r, ctx]).sum(1)
        nn = np.argsort(-agree, kind="stable")[: k]
        nn = nn[rng.permutation(len(nn))]
        for c in cols:
            for dn in nn:
                if donors[dn, c] > 0:
                    Y[r, c] = donors[dn, c]
                    break
    return Y


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------
def run_levels(model, X, mtab, levels, repeats, seed, mechanisms,
               donors=None, items=None, log=print):
    """Return list of per-repeat result dicts for each mechanism and level."""
    U0 = model.score(X)
    M0 = marginal_score(X, mtab)
    n_obs = int((X > 0).sum()) if items is None else int((X[:, items] > 0).sum())
    out = []
    for mech in mechanisms:
        for rho in levels:
            for rep in range(repeats):
                rng = np.random.default_rng([seed, rep, int(rho * 1e5), MECH_IDS[mech]])
                if mech == "permutation":
                    Y = marginal_permutation(X, rho, rng)
                else:
                    S = select_cells(X, rho, rng, items)
                    if mech == "conditional":
                        Y = conditional_independent(model, X, S, rng)
                    elif mech == "sequential":
                        if hasattr(model, "sequential"):
                            Y = model.sequential(X, S, rng)
                        else:
                            Y = conditional_sequential(model, X, S, rng)
                    elif mech == "uniform":
                        Y = uniform_replacement(X, S, model.K, rng)
                    elif mech == "hotdeck":
                        Y = hotdeck_replacement(X, S, donors, rng)
                    else:
                        raise ValueError(mech)
                assert ((Y > 0) == (X > 0)).all(), "mask must be preserved"
                U1 = model.score(Y)
                M1 = marginal_score(Y, mtab)
                dU = U1 - U0
                changed = int((Y != X).sum())
                res = dict(mechanism=mech, rho_requested=rho, repeat=rep,
                           rho_realized=changed / n_obs,
                           auc_lsm=auc(U0, U1), auc_marg=auc(M0, M1),
                           mean_dU=float(dU.mean()),
                           frac_dU_pos=float((dU > 0).mean()),
                           mean_dU_per_changed=float(dU.sum() / max(changed, 1)))
                out.append(res)
                log(f"{mech:12s} rho={rho:<6} rep={rep} realized={res['rho_realized']:.4f} "
                    f"AUC_lsm={res['auc_lsm']:.4f} AUC_marg={res['auc_marg']:.4f} "
                    f"dU={res['mean_dU']:.2f}")
    return out
