import pickle
import pandas as pd
from typing import Union
from sklearn.base import ClassifierMixin, RegressorMixin
import logging
from abc import ABC, abstractmethod

class ExportStrategy(ABC):
  @abstractmethod
  def export_model(self, model: Union[ClassifierMixin, RegressorMixin], file_path: str) -> None:
    pass
  
class ClassifierExportStrategy(ExportStrategy):
  def export_model(self, model: ClassifierMixin, file_path: str) -> None:
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(model, file)
      logging.info(f"Classifier model exported successfully to {file_path}")
    except Exception as e:
      logging.error(f"Error exporting classifier model: {e}")
      raise
    
class RegressorExportStrategy(ExportStrategy):
  def export_model(self, model: RegressorMixin, file_path: str) -> None:
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(model, file)
      logging.info(f"Regressor model exported successfully to {file_path}")
    except Exception as e:
      logging.error(f"Error exporting regressor model: {e}")
      raise