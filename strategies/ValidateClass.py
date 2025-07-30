import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from implementation.evaluateModel import model_evaluation
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union
import logging
import pandas as pd
from abc import ABC, abstractmethod


class ValidateStrategy(ABC):
  @abstractmethod
  def validate_model(self, model: Union[ClassifierMixin, RegressorMixin],x_train, y_train, x_test: pd.DataFrame, y_test: pd.Series, model_type: bool) -> str:
    pass
  
  
class ClassifierValidateStrategy(ValidateStrategy):
  def validate_model(self, model: ClassifierMixin, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, model_type: bool) -> str:
    try:
      scoring = {
        'accuracy': 0,
        'f1_score': 0,
        'cv_score': 0,
        'mean_cv_score': 0
      }
      if model_type == True:
        logging.info("Validating classifier model...")
        test_results = model_evaluation(model, x_test, y_test, model_type)
        train_results = model_evaluation(model, x_train, y_train, model_type)
        logging.info(f"Accuracy Validation ...")
        if abs(test_results['accuracy'] - train_results['accuracy']) < 0.2:
          logging.info("Training and testing accuracy are quite similar")
          if test_results['accuracy'] > 0.7 or train_results['accuracy'] > 0.7:
            logging.info("Good Accuracy Score")
            scoring['accuracy'] = 3
        logging.info(f"F1-score Validation ...")
        if abs(test_results['f1_score'] - train_results['f1_score']) < 0.2:
          logging.info("Training and testing F1 scores are quite similar")
          if test_results['f1_score'] > 0.7 or train_results['f1_score'] > 0.7:
            logging.info("Good F1 Score")
            scoring['f1_score'] = 2
        logging.info(f"Mean Cross-val Validation ...")
        if abs(test_results['mean_cross_val_score'] - train_results['mean_cross_val_score']) < 0.1:
          logging.info("Training and testing mean cross-validation scores are quite similar")
          if test_results['mean_cross_val_score'] > 0.7 or train_results['mean_cross_val_score'] > 0.7:
            logging.info("Good Mean Cross-val Score")
            scoring['mean_cv_score'] = 5
        logging.info(f"Cross-val Validation ...")
        if abs(test_results['cross_val_scores'].std() - train_results['cross_val_scores'].std()) < 0.09:
          logging.info("Training and testing cross-validation scores are quite similar")
          if test_results['cross_val_scores'].std() <= 0.11 or train_results['cross_val_scores'].std() <= 0.11:
            logging.info("Good Cross-val Score")
            scoring['cv_score'] = 4
        logging.info(f"Validation results: {sum(scoring.values())}")
        if sum(scoring.values()) >= 10:
          result = "Model is validated successfully"
        else:
          result = "Model validation failed"
        return result
    except Exception as e:
      logging.error(f"Error validating classifier model: {e}")
      raise
    
class RegressorValidateStrategy(ValidateStrategy):
  def validate_model(self, model: RegressorMixin, x_train, y_train, x_test: pd.DataFrame, y_test: pd.Series, model_type: bool) -> str:
    try:
      scoring = {
        'mean_squared_error': 0,
        'r2_score': 0,
        'cv_score': 0,
        'mean_cv_score': 0
      }
      if model_type == False:
        logging.info("Validating regressor model...")
        test_results = model_evaluation(model, x_test, y_test, model_type)
        train_results = model_evaluation(model, x_train, y_train, model_type)
        logging.info(f"MSE Validation ...")
        if abs(test_results['mean_squared_error'] - train_results['mean_squared_error']) < 0.2:
          logging.info("Training and testing MSE are quite similar")
          if test_results['mean_squared_error'] < 0.2 or train_results['mean_squared_error'] < 0.1:
            logging.info("Good MSE Score")
            scoring['mean_squared_error'] = 3
        logging.info(f"R2-score Validation ...")
        if abs(test_results['r2_score'] - train_results['r2_score']) < 0.2:
          logging.info("Training and testing R2 scores are quite similar")
          if test_results['r2_score'] > 0.7 or train_results['r2_score'] > 0.7:
            logging.info("Good R2 Score")
            scoring['r2_score'] = 3
        logging.info(f"Mean Cross-val Validation ...")
        if abs(test_results['mean_cross_val_score'] - train_results['mean_cross_val_score']) < 0.1:
          logging.info("Training and testing mean cross-validation scores are quite similar")
          if test_results['mean_cross_val_score'] > 0.7 or train_results['mean_cross_val_score'] > 0.7:
            logging.info("Good Mean Cross-val Score")
            scoring['mean_cv_score'] = 5
        logging.info(f"Cross-val Validation ...")
        if abs(test_results['cross_val_scores'].std() - train_results['cross_val_scores'].std()) < 0.09:
          logging.info("Training and testing cross-validation scores are quite similar")
          if test_results['cross_val_scores'].std() <= 0.11 or train_results['cross_val_scores'].std() <= 0.11:
            logging.info("Good Cross-val Score")
            scoring['cv_score'] = 4
        logging.info(f"Validation results: {sum(scoring.values())}")
        if sum(scoring.values()) >= 10:
          result = "Model is validated successfully"
        else:
          result = "Model validation failed"
        return result
    except Exception as e:
      logging.error(f"Error validating regressor model: {e}")
      raise
    
    