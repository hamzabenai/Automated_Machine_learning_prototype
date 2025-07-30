import pandas as pd 
import numpy as np 
import logging
from typing import Tuple, Union
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss, TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data : pd.DataFrame, target: str) -> Union[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
      pass

class RemoveIdentifierStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      if data is not None and target in data.columns:
        for column in data.columns:
          if data[column].nunique() == 1 or data[column].nunique() == len(data):
            data = data.drop(columns=[column])
        return data
      else:
        raise ValueError("Remove Identifier Strategy Error : Data is None or target column is missing.")
    except Exception as e:
      logging.error(f"Error removing identifier columns: {e}")
      raise
    
class MissingValueStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      if data is not None and target in data.columns:
        for column in data.columns:
          count_nan = data[column].isnull().sum()
          if len(data) <= 2000:
            if data[column].dtype in ['object', 'category']:
              data[column].fillna(data[column].mode()[0], inplace=True)
            else:
              data[column].fillna(data[column].mean(), inplace=True)
          elif len(data) > 2000 and count_nan < 0.2 * len(data):
            data = data.dropna(subset=[column])
          else:
            data = data.drop(columns=[column])
        logging.info("Missing values handled successfully.")
        return data
      elif data is None or target not in data.columns:
        raise ValueError("Missing values Strategy Error : Data is None or target column is missing.")
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise

class OutlierStrategy(DataStrategy): 
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      if data is not None and target in data.columns:
        original_rows = len(data)
        logging.info(f"Original number of rows: {original_rows}")
        if len(data) <= 2000:
          numeric_cols = data.select_dtypes(include=[np.number]).columns
          mask = pd.Series(True, index=data.index)
          
          for column in numeric_cols:
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            col_mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
            mask &= col_mask
          data = data[mask]
          if data is None or data.empty:
            raise ValueError("Outlier Strategy Error : IQR method, No data left after outlier removal.")
        elif 2000 < len(data) <= 10000:
          lof = LocalOutlierFactor(n_neighbors=20)
          outliers = lof.fit_predict(data.select_dtypes(include=[np.number]))
          data = data[outliers == 1]
          if data is None or data.empty:
            raise ValueError("Outlier Strategy Error : Local Factor method, No data left after outlier removal.")
        else:
          numeric_data = data.select_dtypes(include=[np.number])
          iso_forest = IsolationForest(contamination='auto',
                                    random_state=42,
                                    n_estimators=100)
          outliers = iso_forest.fit_predict(numeric_data)
          data = data[outliers == 1]
          if data is None or data.empty:
            raise ValueError("Outlier Strategy Error : Isolation Forest method, No data left after outlier removal.")
        
        logging.info("Outliers handled successfully.")
        logging.info(f"Removed {original_rows - len(data)} rows ({((original_rows - len(data))/original_rows):.1%})")
        return data
      else:
        raise ValueError("Outlier Strategy Error : Data is None or target column is missing.")
    except Exception as e:
      logging.error(f"Error handling outliers: {e}")
      raise
    
class ImbalancedDataStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      data_size = None
      if len(data) <= 2000:
        data_size = "small"
      elif 2000 < len(data) <= 20000:
        data_size = "medium"
      else: 
        data_size = "large"
      class_counts = Counter(data[target])
      perform_test = False
      for class_value, count in class_counts.items():
        if count >= 0.75 * len(data):
          perform_test = True
          break
      if perform_test:
        X = data.drop(columns=[target])
        y = data[target]
        logging.warning(f"Class {class_value} is a majority class with {count/len(data)*100}% of the total records.")
        if data_size == "small":
          ada = ADASYN(sampling_strategy='minority', n_neighbors=3)
          X_res, y_res = ada.fit_resample(X, y)
          data = pd.concat([X_res, y_res], axis=1)
        elif data_size == "medium":
          smote_tomek = SMOTETomek(sampling_strategy='auto', tomek=TomekLinks(sampling_strategy='majority'))
          X_res, y_res = smote_tomek.fit_resample(X, y)
          data = pd.concat([X_res, y_res], axis=1)
        else:
          nm = NearMiss(version=3, n_jobs=-1)  
          X_res, y_res = nm.fit_resample(X, y)
          data = pd.concat([X_res, y_res], axis=1)
        logging.info("Imbalanced data handled successfully.")
      return data
    except Exception as e:
      logging.error(f"Error handling imbalanced data: {e}")
      raise

class SplitDataStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    try:
      x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=[target]), 
                                                        data[target], 
                                                        test_size=0.2, 
                                                        random_state=42,stratify=data[target])
      logging.info("Data split into training and testing sets successfully.")
      return x_train, x_test, y_train, y_test
    except Exception as e:
      logging.error(f"Error splitting data: {e}")
      raise

class ScaleDataStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      X = data.drop(columns=[target])
      y = data[target]
      scaler = RobustScaler()
      X_scaled = scaler.fit_transform(X)
      scaled_data = pd.DataFrame(X_scaled, columns=X.columns)
      scaled_data[target] = y.reset_index(drop=True)
      logging.info("Data scaled successfully.")
      return scaled_data
    except Exception as e:
      logging.error(f"Error scaling data: {e}")
      raise
    
class EncodeDataStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      X = data.drop(columns=[target])
      y = data[target]
      categorical_cols = X.select_dtypes(include=['object', 'category']).columns
      for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
      if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_encoded = pd.Series(y_encoded, name=target, index=y.index)
      else:
        y_encoded = y
      X[target] = y_encoded
      logging.info("Data encoded successfully.")
      return X
    except Exception as e:
      logging.error(f"Error encoding data: {e}")
      raise
  
class FeatureSelectionStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      X = data.drop(columns=[target])
      y = data[target]
      if pd.api.types.is_categorical_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        mi = mutual_info_classif(X, y, random_state=42)
      else:
          mi = mutual_info_regression(X, y, random_state=42)
      threshold = max(0.1, np.percentile(mi, 50))
      selected = mi >= threshold
      if not any(selected): 
          selected = mi >= mi.max()
      X_filtered = X.loc[:, selected]
      logging.info(f"Selected {X_filtered.shape[1]} features with threshold {threshold:.3f}")
      return pd.concat([X_filtered, y], axis=1)
    except Exception as e:
      logging.error(f"Feature selection failed, returning original data. Error: {e}")
      return data 

# class FeatureEngineeringStrategy(DataStrategy):
#   def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
#     pass

class DimensionalityReductionStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    try:
      X = data.drop(columns=[target])
      y = data[target]
      pca = PCA(n_components=0.95, random_state=42)
      X_reduced = pca.fit_transform(X)
      reduced_data = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])])
      reduced_data[target] = y.reset_index(drop=True)
      logging.info("Dimensionality reduction applied successfully.")
      return reduced_data
    except Exception as e:
      logging.error(f"Error applying dimensionality reduction: {e}")
      raise