"""Canonical reduction implementation objects for named plate reductions."""

from .expr.ops.reduction import LOGSUMEXP, MAX, MEAN, MIN, PROD, SUM

__all__ = ["LOGSUMEXP", "MAX", "MEAN", "MIN", "PROD", "SUM"]
