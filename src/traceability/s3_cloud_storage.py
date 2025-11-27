from zoneinfo import TZPATH
import boto3
from botocore.exceptions import ClientError
import os
from pathlib import Path
from exception import CGAgentException
from logger import logging
from trail import S3StorageService


class S3StorageService():
    def __init__(self,bucket_name:str="causal-governance-artifact"):
        self.bucket_name= bucket_name
        self.s3_client= boto3.client('s3')

    def upload_artifact(self,local_path:str,artifact_type:str,client_transaction_id:str=None)->str:
        """Upload to REAL AWS S3 and return downloadable URL"""

        try:
            #create s3 key
            file_name= Path(local_path).name
            if client_transaction_id:
                s3_key= f"{artifact_type}/{client_transaction_id}/{file_name}"

            else:
                s3_key= f"{artifact_type}/global/{file_name}"
            
            #upload file to s3

            self.s3_client.upload_file(local_path,self.bucket_name,s3_key)
            
            logging.info(f"Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            
            # Generate presigned URL (downloadable for 7 days)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket':self.bucket_name,'Key': s3_key},
                ExpiresIn=604800  # 7 days
            )
            
            return presigned_url
            
        except ClientError as e:
            logging.error(f"S3 upload failed: {e}")
            raise

