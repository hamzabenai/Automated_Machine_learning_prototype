import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from implementation.cleanData import fill_missing_values, remove_outliers, encode_data, scale_data, select_features, reduce_dimensions, split_data, remove_identifiers
from implementation.trainModel import train_model
from implementation.validateModel import model_validation
from implementation.evaluateModel import model_evaluation
from implementation.exportModel import exporting_model
from strategies.IngestClass import IngestDataClass
from typing import Union
import pandas as pd
import logging


def main_pipeline(data_path: str, target: str, model_name: str, model_path: str, model_type: bool) -> Union[str, dict]:
  try:
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting main pipeline...")
    
    # import the data / IngestClass.py
    data = IngestDataClass().load_data(target, data_path)
    if model_type == True and data[target].dtype in ['int64','object', 'category']:
      model_type = True 
    elif data[target].dtype in ['int64', 'float64'] and model_type == False:
      model_type = False
    
    # handling data issues / cleanData.py
    data = remove_identifiers(data, target)
    data = fill_missing_values(data, target)
    data = remove_outliers(data, target)
    data = encode_data(data, target)
    data = scale_data(data, target)
    data = select_features(data, target)
    # data = reduce_dimensions(data, target)
    x_train, x_test, y_train, y_test = split_data(data, target)
    
    # handling model training / trainModel.py
    model = train_model(x_train, y_train, model_name, model_type)

    # handling model evaluation / evaluateModel.py
    evaluation_results = model_evaluation(model, x_test, y_test, model_type)
    logging.info(f"Model evaluation results: {evaluation_results}")
    
    # handling model validation / validateModel.py
    validation_result = model_validation(model, x_train, y_train, x_test, y_test, model_type)
    if "successfully"  in validation_result:
      # handling model exporting / exportModel.py
      exporting_model(model, model_path)
      logging.info("Main pipeline completed successfully.")
      return evaluation_results
    else:
      return 'Model validation failed.'
    
  except Exception as e:
    logging.error(f"Error in main pipeline: {e}")
    raise RuntimeError(f"Pipeline execution failed: {e}")