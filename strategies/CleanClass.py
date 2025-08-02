import pandas as pd 
import numpy as np 
import logging
from typing import Tuple, Union, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import Augmentor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.base import TransformerMixin
class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data : pd.DataFrame, target: str, Prediction: bool = False) -> Union[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
      pass

class RemoveIdentifierStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str, Prediction: bool = False) -> pd.DataFrame:
    try:
      if [data is not None and target in data.columns] or Prediction == True:
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
  def handle_data(self, data: pd.DataFrame, target: str, Prediction: bool = False) -> pd.DataFrame:
    try:
      if [data is not None and target in data.columns] or Prediction:
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
      elif [data is None or target not in data.columns] and not Prediction:
        raise ValueError("Missing values Strategy Error : Data is None or target column is missing.")
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise

class OutlierStrategy(DataStrategy): 
  def handle_data(self, data: pd.DataFrame, target: str = None, Prediction: bool = False, detector: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
    try:
      if [data is not None and target in data.columns] or Prediction:
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
          if not Prediction:
            detector = LocalOutlierFactor(n_neighbors=20)
            detector.fit(data.select_dtypes(include=[np.number]))
          outliers = detector.predict(data.select_dtypes(include=[np.number]))
          data = data[outliers == 1]
          if data is None or data.empty:
            raise ValueError("Outlier Strategy Error : Local Factor method, No data left after outlier removal.")
        else:
          numeric_data = data.select_dtypes(include=[np.number])
          if not Prediction:
            detector = IsolationForest(contamination='auto',
                                    random_state=42,
                                    n_estimators=100)
            detector.fit(numeric_data)
          outliers = detector.predict(numeric_data)
          data = data[outliers == 1]
          if data is None or data.empty:
            raise ValueError("Outlier Strategy Error : Isolation Forest method, No data left after outlier removal.")
        
        logging.info("Outliers handled successfully.")
        logging.info(f"Removed {original_rows - len(data)} rows ({((original_rows - len(data))/original_rows):.1%})")
        return data, detector
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
        if data_size == "small" or data_size == "medium":
          class_counts = y.value_counts()
          minority_class = class_counts.idxmin()
          majority_class = class_counts.idxmax()
          minority_data = data[data[target] == minority_class]
          majority_data = data[data[target] == majority_class]
          p = Augmentor.DataPipeline(minority_data)
          p.sample(1000, method="SMOTE")  # Generates synthetic samples
          augmented_minority = p.sample(1000)
          data = pd.concat([augmented_minority, majority_data])
        else:
          majority_class = y.value_counts().idxmax()
          minority_class = y.value_counts().idxmin()
          majority_data = X[y == majority_class]
          minority_data = X[y == minority_class]
          majority_downsampled = resample(
              majority_data,
              replace=False,
              n_samples=len(minority_data),
              random_state=42
          )
          X_balanced = pd.concat([majority_downsampled, minority_data])
          y_balanced = pd.Series([majority_class] * len(majority_downsampled) + 
                                [minority_class] * len(minority_data))
          data = pd.concat([X_balanced, y_balanced], axis=1)
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
  def handle_data(self, data: pd.DataFrame, target: str, prediction: bool = False, scaler: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
    try:
      if not prediction:
        X = data.drop(columns=[target])
        y = data[target]
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(X)
        scaled_data = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        scaled_data[target] = y.reset_index(drop=True)
        logging.info("Data scaled successfully (training mode).")
      else:
        scaled_data = scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logging.info("Data scaled successfully (prediction mode).")
      return scaled_data, scaler
    except Exception as e:
      logging.error(f"Error scaling data: {e}")
      raise
  
class EncodeDataStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str, prediction: bool = False, encoder: TransformerMixin = None) -> Tuple[pd.DataFrame, TransformerMixin]:
    try:
      if not prediction:
        X = data.drop(columns=[target])
        y = data[target]
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
          encoder = LabelEncoder()
          X[col] = encoder.fit_transform(X[col])
        if pd.api.types.is_categorical_dtype(y) or not pd.api.types.is_numeric_dtype(y):
          target_encoder = LabelEncoder()
          y_encoded = target_encoder.fit_transform(y)
          y_encoded = pd.Series(y_encoded, name=target, index=y.index)
        else:
          y_encoded = y
        X[target] = y_encoded
        logging.info(f"Encoded {len(categorical_cols)} categorical columns (training mode)")
        return X, encoder
      else:
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
          data[col] = encoder.transform(data[col])
        return data, encoder
    except Exception as e:
        logging.error(f"Error encoding data: {e}")
        raise ValueError(f"Data encoding failed: {str(e)}") from e
class FeatureSelectionStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Union[list, pd.Index]]:
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
      return pd.concat([X_filtered, y], axis=1), X_filtered.columns.tolist()
    except Exception as e:
      logging.error(f"Feature selection failed, returning original data. Error: {e}")
      return data, data.columns

class DimensionalityReductionStrategy(DataStrategy):
  def handle_data(self, data: pd.DataFrame, target: str, prediction: bool = False, pca: PCA = None) -> Tuple[pd.DataFrame, PCA]:
    try:
      if not prediction:
        X = data.drop(columns=[target])
        y = data[target]
        pca = PCA(n_components=0.95, random_state=42)
        X_reduced = pca.fit_transform(X)
        reduced_data = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])])
        reduced_data[target] = y.reset_index(drop=True)
        logging.info(f"Dimensionality reduction applied (training). Kept {X_reduced.shape[1]} components.")
        return reduced_data, self.pca
      else:
        X_reduced = pca.transform(data)
        reduced_data = pd.DataFrame(X_reduced,columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])])
        logging.info(f"Dimensionality reduction applied (prediction). Kept {X_reduced.shape[1]} components.")
        return reduced_data, pca
    except Exception as e:
        logging.error(f"Error in dimensionality reduction: {e}")
        raise ValueError(f"Dimensionality reduction failed: {str(e)}") from e