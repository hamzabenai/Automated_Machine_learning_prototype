import pandas as pd
from typing import Tuple
import logging
class IngestDataClass:
    def load_data(self, target: str, data_path: str = None) -> pd.DataFrame:
        try:
            if '.csv' in data_path:
                data = pd.read_csv(data_path)
            elif '.xlsx' in data_path:
                data = pd.read_excel(data_path)
            else:
                raise ValueError("Data source must be a DataFrame or a valid file path.")
            logging.info(f"Data loaded successfully from {data_path}")
            logging.info(f"Data shape: {data.shape}")
            if target in data.columns:
                return data
            else:
                raise ValueError(f"Target column '{target}' not found in the data.")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None