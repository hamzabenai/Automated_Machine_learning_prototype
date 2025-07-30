import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.EvaluateClass import RegressorEvaluateStrategy, ClassifierEvaluateStrategy
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union
import logging
import pandas as pd


def model_evaluation(model: Union[ClassifierMixin, RegressorMixin], x_test: pd.DataFrame, y_test: pd.Series, model_type: bool) -> dict:
  try:
    if model_type == True:
      strategy = ClassifierEvaluateStrategy().evaluate_model(model, x_test, y_test)
      return strategy
    elif model_type == False:
      strategy = RegressorEvaluateStrategy().evaluate_model(model, x_test, y_test)
      return strategy
    else:
      raise ValueError("Invalid task type. Use 'classification' or 'regression'.")
  except Exception as e:
    logging.error(f"Error evaluating model: {e}")
    raise
