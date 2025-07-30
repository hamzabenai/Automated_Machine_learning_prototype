import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.ValidateClass import ClassifierValidateStrategy, RegressorValidateStrategy
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union
import logging
import pandas as pd


def model_validation(model: Union[ClassifierMixin,RegressorMixin] ,x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, model_type: bool) -> str:
  try:
    if model_type == True:
      strategy = ClassifierValidateStrategy()
    elif model_type == False:
      strategy = RegressorValidateStrategy()
    else:
      raise ValueError("Invalid model type. Use ClassifierMixin or RegressorMixin.")

    result = strategy.validate_model(model, x_train, y_train, x_test, y_test, model_type)
    logging.info(f"validation result: {result}")
    return result
  except Exception as e:
    logging.error(f"Error validating model: {e}")
    raise