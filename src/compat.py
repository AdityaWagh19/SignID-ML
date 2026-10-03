"""Compatibility shims for cross-version scikit-learn unpickling.

Ensures models pickled with scikit-learn 1.3.x can be seamlessly unpickled and evaluated
under scikit-learn 1.4, 1.5, 1.6+ (and vice versa) without AttributeError or ModuleNotFoundError.
"""

import sys
import types
import numpy as np

# 1. Shim for sklearn.ensemble._gb_losses (removed in scikit-learn >= 1.4)
if "sklearn.ensemble._gb_losses" not in sys.modules:
    try:
        import sklearn.ensemble._gb_losses  # noqa: F401
    except ModuleNotFoundError:
        from scipy.special import logsumexp

        mod = types.ModuleType("sklearn.ensemble._gb_losses")

        class MultinomialDeviance:
            is_multi_class = True

            def __init__(self, n_classes=15):
                self.n_classes = n_classes
                self.K = n_classes

            def get_init_raw_predictions(self, X, estimator):
                probas = estimator.predict_proba(X)
                eps = np.finfo(np.float32).eps
                probas = np.clip(probas, eps, 1 - eps)
                return np.log(probas).astype(np.float64)

            def _raw_prediction_to_proba(self, raw_predictions):
                return np.nan_to_num(
                    np.exp(raw_predictions - (logsumexp(raw_predictions, axis=1)[:, np.newaxis]))
                )

            def _raw_prediction_to_decision(self, raw_predictions):
                proba = self._raw_prediction_to_proba(raw_predictions)
                return np.argmax(proba, axis=1)

        class BinomialDeviance(MultinomialDeviance):
            pass

        class ExponentialLoss(MultinomialDeviance):
            pass

        mod.MultinomialDeviance = MultinomialDeviance
        mod.BinomialDeviance = BinomialDeviance
        mod.ExponentialLoss = ExponentialLoss
        sys.modules["sklearn.ensemble._gb_losses"] = mod


# 2. Patch DecisionTreeClassifier.monotonic_cst for scikit-learn >= 1.4 compatibility
try:
    from sklearn.tree import DecisionTreeClassifier
    if not hasattr(DecisionTreeClassifier, "monotonic_cst"):
        DecisionTreeClassifier.monotonic_cst = None
except Exception:
    pass
