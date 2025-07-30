import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.ExportClass import ClassifierExportStrategy, RegressorExportStrategy
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union  
import logging

def exporting_model(model: Union[ClassifierMixin, RegressorMixin], file_path: str = '../models/model.pkl') -> None:
  try:
    if isinstance(model, ClassifierMixin):
      strategy = ClassifierExportStrategy()
    elif isinstance(model, RegressorMixin):
      strategy = RegressorExportStrategy()
    else:
      raise ValueError("Invalid model type. Use ClassifierMixin or RegressorMixin.")

    strategy.export_model(model, file_path)
    logging.info(f"Model exported successfully to {file_path}")
  except Exception as e:
      logging.error(f"Error exporting model: {e}")
      raise