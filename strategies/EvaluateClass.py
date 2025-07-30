from typing import Union, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, confusion_matrix, roc_auc_score
import logging
from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin, RegressorMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import robust_scale

class EvaluateStrategy(ABC):
  @abstractmethod
  def evaluate_model(self, model: ClassifierMixin,  x_test: pd.DataFrame , y_test: pd.Series) -> dict:
    pass
  
class ClassifierEvaluateStrategy(EvaluateStrategy):
  def evaluate_model(self, model, x_test, y_test) -> dict:
    try:
      y_pred = model.predict(x_test)
      
      # Determine average method automatically
      average_method = 'weighted' 
      return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average=average_method),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'mean_cross_val_score': cross_val_score(model, x_test, y_test, 
                                              cv=5, scoring='accuracy').mean(),
        'cross_val_scores': cross_val_score(model, x_test, y_test, 
                                          cv=5, scoring='accuracy')
      }
    except Exception as e:
      logging.error(f"Error evaluating classifier model: {e}")
      raise
    
class RegressorEvaluateStrategy(EvaluateStrategy):
  def evaluate_model(self, model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
      y_pred = model.predict(x_test)
      scaled_y_test = robust_scale(y_test)
      scaled_y_pred = robust_scale(y_pred)
      mse = mean_squared_error(scaled_y_test, scaled_y_pred)
      r2 = r2_score(y_test, y_pred)
      cv_score = cross_val_score(model, x_test, y_test, cv=5, scoring='r2')
      mean_cv_score = cv_score.mean()
      logging.info(f"Model evaluation completed with MSE: {mse}, R2: {r2}")
      return {
        'mean_squared_error': mse,
        'r2_score': r2,
        'mean_cross_val_score': mean_cv_score,
        'cross_val_scores': cv_score
        }
    except Exception as e:
      logging.error(f"Error evaluating regressor model: {e}")
      raise