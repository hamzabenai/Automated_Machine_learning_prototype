import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from implementation.cleanData import remove_identifiers, fill_missing_values, remove_outliers, encode_data, scale_data, reduce_dimensions
from typing import Union
from sklearn.base import ClassifierMixin, RegressorMixin
from abc import ABC, abstractmethod
import pandas as pd
import logging

class PredictionStrategy(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame, model: Union[ClassifierMixin, RegressorMixin]) -> pd.DataFrame:
      pass
    
class PredictNow(PredictionStrategy):
  def predict(self, data: pd.DataFrame, model: Union[ClassifierMixin, RegressorMixin], features: Union[list, pd.Index], proc_models: dict = None) -> pd.DataFrame:
    try:
      identifier = None
      for col in data.columns:
        if data[col].nunique() == len(data):
          identifier = data[col]
          break
      data = remove_identifiers(data, target=None, Prediction=True)
      data = fill_missing_values(data)
      data, _ = encode_data(data, Prediction=True, encoder=proc_models.get('encoder'))
      data, _ = scale_data(data, Prediction=True, scaler=proc_models.get('scaler'))
      data = data[features]
      # data, _ = reduce_dimensions(data, Prediction=True, pca=proc_models.get('pca'))

      predictions = model.predict(data)
      data['predictions'] = predictions
      data['Id'] = identifier if identifier is not None else data
      logging.info("Predictions made successfully.")
      return data
    except Exception as e:
      logging.error(f"Error in making predictions: {e}")
      raise RuntimeError(f"Error in making predictions: {e}")