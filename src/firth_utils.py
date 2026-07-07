# ============================================================================
# firth_utils.py
#
# Standalone implementation of Firth's (1993) bias-reduced logistic
# regression, used because the third-party `firthlogist` package (v0.5.0)
# is incompatible with scikit-learn >= 1.6 due to an internal API change
# (BaseEstimator._validate_data was removed/relocated). This implementation
# has no scikit-learn dependency beyond numpy/scipy and will not break with
# future sklearn upgrades.
#
# Reference: Firth, D. (1993). Bias reduction of maximum likelihood
# estimates. Biometrika, 80(1), 27-38.
# ============================================================================

import numpy as np
from scipy import stats


def _sigmoid(z):
    """Numerically stable logistic sigmoid."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def fit_firth_logistic(X, y, max_iter=100, tol=1e-6, add_intercept=True):
    """
    Fit a Firth (1993) bias-reduced logistic regression via modified IRLS.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix (should NOT already include an intercept column
        unless add_intercept=False).
    y : ndarray, shape (n_samples,)
        Binary outcome (0/1).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on the max absolute change in coefficients.
    add_intercept : bool
        If True, prepends a column of ones to X.

    Returns
    -------
    dict with keys:
        'coef'       : ndarray, coefficients (intercept first if added)
        'se'         : ndarray, standard errors (from observed information)
        'pvals'      : ndarray, two-sided Wald p-values
        'ci_low'     : ndarray, 95% CI lower bound (on coefficient scale)
        'ci_high'    : ndarray, 95% CI upper bound (on coefficient scale)
        'converged'  : bool
        'n_iter'     : int
        'has_intercept': bool
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    n, p = X.shape
    beta = np.zeros(p)

    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        eta = X @ beta
        pi = _sigmoid(eta)
        W_diag = pi * (1 - pi)
        W_diag = np.clip(W_diag, 1e-10, None)  # avoid exact zeros

        # Weighted design matrix pieces for the information matrix
        XW = X * W_diag[:, None]
        info_matrix = X.T @ XW   # Fisher information matrix, shape (p, p)

        try:
            info_inv = np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            info_inv = np.linalg.pinv(info_matrix)

        # Hat matrix diagonal (leverage), needed for Firth's penalty term
        # h_i = W_diag_i * x_i^T (X^T W X)^-1 x_i
        XW_half = X * np.sqrt(W_diag)[:, None]
        hat_diag = np.einsum("ij,jk,ik->i", XW_half, info_inv, XW_half)
        hat_diag = np.clip(hat_diag, 0, 1)  # numerical safety

        # Firth-modified score: standard score + penalty term
        # U*(beta) = X^T (y - pi + h_i * (0.5 - pi_i))
        score = X.T @ (y - pi + hat_diag * (0.5 - pi))

        # Newton-Raphson-style update using the (unmodified) information matrix
        delta = info_inv @ score
        beta_new = beta + delta

        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            converged = True
            break

        beta = beta_new

    # Final quantities at convergence (or max_iter)
    eta = X @ beta
    pi = _sigmoid(eta)
    W_diag = np.clip(pi * (1 - pi), 1e-10, None)
    XW = X * W_diag[:, None]
    info_matrix = X.T @ XW
    try:
        info_inv = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(info_matrix)

    se = np.sqrt(np.clip(np.diag(info_inv), 0, None))
    z_scores = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    z_crit = stats.norm.ppf(0.975)
    ci_low = beta - z_crit * se
    ci_high = beta + z_crit * se

    return {
        "coef": beta,
        "se": se,
        "pvals": pvals,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "converged": converged,
        "n_iter": n_iter,
        "has_intercept": add_intercept,
    }


def fit_firth_univariate(x, y, standardize=True):
    """
    Convenience wrapper for a single-variable Firth logistic regression,
    matching the univariate design used in NB03 (Study 2, Section 3.3
    Stage 2: each significant variable entered individually).

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The single predictor (raw scale; standardisation applied inside
        if standardize=True).
    y : array-like, shape (n_samples,)
        Binary outcome.
    standardize : bool
        If True, x is standardised (mean 0, sd 1) before fitting, so the
        resulting odds ratio is interpretable "per 1 SD" — matching the
        paper's reported OR scale (paper Table 3: "OR = Firth odds ratio
        (standardised)").

    Returns
    -------
    dict with keys: 'coef', 'se', 'p_value', 'OR', 'OR_ci_low', 'OR_ci_high',
                    'converged', 'n_iter'
        Note: 'coef' / 'se' / p_value here refer ONLY to the predictor
        (the intercept term is fit internally but not returned), since
        downstream code only needs the single-variable effect.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    if standardize:
        mu, sd = x.mean(), x.std(ddof=0)
        if sd == 0:
            raise ValueError("Predictor has zero variance; cannot standardize.")
        x = (x - mu) / sd

    result = fit_firth_logistic(x, y, add_intercept=True)

    # coef[0] is the intercept, coef[1] is the predictor's coefficient
    predictor_coef = result["coef"][1]
    predictor_se = result["se"][1]
    predictor_pval = result["pvals"][1]
    predictor_ci_low = result["ci_low"][1]
    predictor_ci_high = result["ci_high"][1]

    return {
        "coef": predictor_coef,
        "se": predictor_se,
        "p_value": predictor_pval,
        "OR": np.exp(predictor_coef),
        "OR_ci_low": np.exp(predictor_ci_low),
        "OR_ci_high": np.exp(predictor_ci_high),
        "converged": result["converged"],
        "n_iter": result["n_iter"],
    }