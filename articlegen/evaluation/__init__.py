"""Article quality assessment and quartile classification."""

from .evaluator import ArticleEvaluation, ArticleEvaluator, EvaluationError, evaluate_article
from .rubric import DIMENSIONS, QUARTILE_POLICY, WEIGHTS

__all__ = [
    "ArticleEvaluation", "ArticleEvaluator", "DIMENSIONS", "EvaluationError",
    "QUARTILE_POLICY", "WEIGHTS", "evaluate_article",
]
