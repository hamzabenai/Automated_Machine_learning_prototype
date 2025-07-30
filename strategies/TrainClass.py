from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union, Tuple
import pandas as pd
from abc import ABC, abstractmethod
import logging

class TrainStrategy(ABC):
  @abstractmethod
  def train_model(self, x_train, y_train, model_name: str) -> Union[ClassifierMixin, RegressorMixin]:
    pass


class ClassifierStrategy(TrainStrategy):
  def train_model(self, x_train, y_train, model_name) -> ClassifierMixin:
    try:
      models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'XGBClassifier': XGBClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(),
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB()
      }
      
      for name, model in models.items():
        if name == model_name:
          model.fit(x_train, y_train)
          logging.info(f"{name} trained successfully.")
          return model
      model = models.get(model_name, RandomForestClassifier())
      model.fit(x_train, y_train)
      logging.info(f"we used the randomForest as a default model, model trained successfully.")
      return model
    except Exception as e:
      logging.error(f"Error training Random Forest Classifier: {e}")
      raise
    
class RegressorStrategy(TrainStrategy):
  def train_model(self, x_train, y_train, model_name) -> RegressorMixin:
    try:
      models = {
        'RandomForestRegressor': RandomForestRegressor(),
        'XGBRegressor': XGBRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR': SVR(),
        'LinearRegression': LinearRegression(),
        'DecisionTreeRegressor': DecisionTreeRegressor()
      }
      
      for name, model in models.items():
        if name == model_name:
          model.fit(x_train, y_train)
          logging.info(f"{name} trained successfully.")
          return model
      model = models.get(model_name, LinearRegression())
      model.fit(x_train, y_train)
      logging.info(f"we used the LinearRegression as a default model, model trained successfully.")
      return model
    except Exception as e:
      logging.error(f"Error training Random Forest Regressor: {e}")
      raise