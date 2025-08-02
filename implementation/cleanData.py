import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.CleanClass import MissingValueStrategy, OutlierStrategy, EncodeDataStrategy, FeatureSelectionStrategy, DimensionalityReductionStrategy, SplitDataStrategy, ScaleDataStrategy, RemoveIdentifierStrategy, ImbalancedDataStrategy
import pandas as pd
import logging
from typing import Tuple, Dict, Optional, Union
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

def remove_identifiers(data: pd.DataFrame, target: str = None) -> pd.DataFrame:
  try:
    data = RemoveIdentifierStrategy().handle_data(data, target)
    logging.info("Identifiers removed successfully.")
    return data
  except Exception as e:
    logging.error(f"Error removing identifiers: {e}")
    raise RuntimeError(f"Error in removing identifiers: {e}") 
  
def fill_missing_values(data: pd.DataFrame, target: str = None) -> pd.DataFrame:
  try:
    data = MissingValueStrategy().handle_data(data, target)
    return data
  except Exception as e:
    raise RuntimeError(f"Error in filling missing values: {e}")
  
def remove_outliers(data: pd.DataFrame, target: str = None, Prediction: bool = False, detector: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
  try:
    data, detecor = OutlierStrategy().handle_data(data, target, Prediction, detector)
    
    return data, detecor
  except Exception as e:
    raise RuntimeError(f"Error in removing outliers: {e}")
  
def encode_data(data: pd.DataFrame, target: str = None, Prediction: bool = False, encoder: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
  try:
    data, encoder = EncodeDataStrategy().handle_data(data, target, Prediction, encoder)
    
    return data, encoder
  except Exception as e:
    raise RuntimeError(f"Error in encoding data: {e}")
  
def scale_data(data: pd.DataFrame, target: str = None, Prediction: bool = False, scaler: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
  try:
    data, scaler = ScaleDataStrategy().handle_data(data, target, Prediction, scaler)
    
    return data, scaler
  except Exception as e:
    raise RuntimeError(f"Error in scaling data: {e}")
  
def select_features(data: pd.DataFrame, target: str = None) -> Tuple[pd.DataFrame, Union[list, pd.Index]]:
  try:
    data, selected_feature = FeatureSelectionStrategy().handle_data(data, target)
    
    return data, selected_feature
  except Exception as e:
    raise RuntimeError(f"Error in feature selection: {e}")
  
def reduce_dimensions(data: pd.DataFrame, target: str = None, Prediction: bool = False, pca: PCA = None) -> Tuple[pd.DataFrame, PCA]:
  try:
    data, reducer = DimensionalityReductionStrategy().handle_data(data, target, Prediction, pca)
    
    return data, reducer
  except Exception as e:
    raise RuntimeError(f"Error in dimensionality reduction: {e}")
  
def split_data(data: pd.DataFrame, target: str = None, test_size: float = 0.2, random_state: int = 42) -> tuple: 
  try:
    x_train, x_test, y_train, y_test = SplitDataStrategy().handle_data(data, target)
    logging.info('data shape after splitting: x_train: {}, y_train: {}, x_test: {}, y_test: {}'.format(
      x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return x_train, x_test, y_train, y_test
  except Exception as e:
    raise RuntimeError(f"Error in splitting data: {e}")

def handle_imbalanced_data(data: pd.DataFrame, target: str = None, model_type: bool = False) -> pd.DataFrame:
  try:
    data = ImbalancedDataStrategy().handle_data(data, target, model_type)
    return data
  except Exception as e:
    logging.error(f"Error handling imbalanced data: {e}")
    raise RuntimeError(f"Error in handling imbalanced data: {e}")

