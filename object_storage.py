import os
from abc import ABC, abstractmethod
from pathlib import Path

import boto3


class ObjectStorage(ABC):
    """
    Abstract base class for object storage services.
    """

    @abstractmethod
    def upload(self, bucket: str, key: str, data: bytes) -> None:
        pass

    @abstractmethod
    def download(self, bucket: str, key: str) -> bytes:
        pass

    @abstractmethod
    def delete(self, bucket: str, key: str) -> None:
        pass

    @abstractmethod
    def list_objects(self, bucket: str) -> list:
        pass

    @abstractmethod
    def check_bucket(self, bucket: str) -> bool:
        pass

    @abstractmethod
    def create_bucket(self, bucket: str) -> None:
        pass


class LocalFileStorage(ObjectStorage):

    def upload(self, bucket: str, key: str, data: bytes) -> None:
        if not os.path.isdir(bucket):
            raise ValueError("Provide dir path for bucket")

        file_save_path = Path(bucket) / Path(key)
        os.makedirs(str(file_save_path.parent), exist_ok=True)
        file_save_path.write_bytes(data)

    def download(self, bucket, key):

        file_save_path = Path(bucket) / Path(key)
        return file_save_path.read_bytes()

    def delete(self, bucket, key):

        file_save_path = Path(bucket) / Path(key)
        if file_save_path.exists():
            os.remove(file_save_path)

    def list_objects(self, bucket):
        return os.listdir(bucket)

    def check_bucket(self, bucket):
        return os.path.exists(bucket)

    def create_bucket(self, bucket):
        os.makedirs(bucket, exist_ok=True)


class S3Storage(ObjectStorage):
    """
    Implementation of ObjectStorage for AWS S3.
    """

    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )

    def upload(self, bucket: str, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=bucket, Key=key, Body=data)

    def download(self, bucket: str, key: str) -> bytes:
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def delete(self, bucket: str, key: str) -> None:
        self.s3.delete_object(Bucket=bucket, Key=key)

    def list_objects(self, bucket: str) -> list:
        response = self.s3.list_objects_v2(Bucket=bucket)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def check_bucket(self, bucket: str) -> bool:
        try:
            self.s3.head_bucket(Bucket=bucket)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def create_bucket(self, bucket: str) -> None:
        self.s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": self.s3.meta.region_name},
        )