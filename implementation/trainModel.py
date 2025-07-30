import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.TrainClass import ClassifierStrategy, RegressorStrategy
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union
import logging



def train_model(x_train, y_train, model_name: str, model_type: bool) -> Union[ClassifierMixin, RegressorMixin]:
  try:
    if model_type == True:
      strategy = ClassifierStrategy()
    elif model_type == False:
      strategy = RegressorStrategy()
    else:
      raise ValueError("Invalid task type. Use 'classification' or 'regression'.")

    model = strategy.train_model(x_train, y_train, model_name)
    logging.info(f"Model {model_name} trained successfully.")
    return model
  except Exception as e:
    logging.error(f"Error training model {model_name}: {e}")
    raise