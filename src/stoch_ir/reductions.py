"""Canonical reduction objects for named plate reductions.

Use these immutable singletons with
:meth:`stoch_ir.RandomVariable.reduce_plates`. Caller-defined reductions are
not a supported v0.1 extension point.
"""

from .expr.ops.reduction import LOGSUMEXP, MAX, MEAN, MIN, PROD, SUM, Reduction

__all__ = ["LOGSUMEXP", "MAX", "MEAN", "MIN", "PROD", "Reduction", "SUM"]
