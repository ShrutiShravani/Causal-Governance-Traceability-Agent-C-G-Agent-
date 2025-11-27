import os,sys
import shutil
from pathlib import Path
from typing import Optional
from logger import logging
from exception  import CGAgentException
from src.traceability.s3_cloud_storage import S3StorageService


class CloudStorageService:
    """
    Mock cloud storage that ssimulates S3/
    """
    def __init__(self,use_real_s3:bool=False):
        self.use_real_s3=use_real_s3
        if use_real_s3:
            try:
                self.s3_service = S3StorageService()
                logging.info("Real S3 service initialized")
            except Exception as e:
                logging.info(f"Real S3 failed,using mock:{e}")


    def upload_artifact(self,local_path:str,artifact_type:str,client_transaction_id:Optional[str]=None)->str:
        try:
            if self.use_real_s3:
                presigned_url= self.s3_service.upload_artifact(local_path,artifact_type,client_transaction_id)
                return presigned_url
            else:
                raise Exception("Upload to s3 failed") 
        except Exception as e:
            logging.error(f"Cloud upload failed for {local_path}: {e}")
            raise CGAgentException(e,sys)
