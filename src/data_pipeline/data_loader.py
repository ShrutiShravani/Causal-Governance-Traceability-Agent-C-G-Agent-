from logger import logging
from exception import CGAgentException
import pandas as pd
import os,sys
import yaml
from pathlib import Path


class Data_Reader:  
    @staticmethod
    def load_yaml(path:str):
        """load yaml file
        """
        with open(path,'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_data(paths_config=r"src\config\paths.yaml",features_config=r"src\config\features.yaml")->pd.DataFrame:
        """
        Loads the raw dataset, applies basic column validation,
        and returns raw DataFrame + X, y split for downstream processing.
        """
        try:
            logging.info("loading config files")

            if not os.path.exists(paths_config):
                raise CGAgentException("File not found")

            if not os.path.exists(features_config):
                raise CGAgentException("features file doesn't exists")

            #load configs
            paths=Data_Reader.load_yaml(paths_config)
            features=Data_Reader.load_yaml(features_config)

            #paths
            raw_path= Path(paths["source_path"])
            raw_data_path= Path(paths["raw_path"])

            if not raw_path.exists():
                raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
            
            #load dataset
            print(f"load dataset from :{raw_path}")
            df=pd.read_csv(raw_path)
                       
            print(f"Loaded raw dataset: {paths['dataset_name']} with shape {df.shape}")
            logging.info(f"loaded dataset :raw_path: {paths['dataset_name']} with shape {df.shape}")

            #ensure parent directory exist# ✅ ensure parent directory exists
            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_data_path, index=False)

            logging.info(f"✅ Raw dataset saved to: {raw_data_path}")

            return df, paths, features
        
        except Exception as e:
            raise CGAgentException(e,sys)