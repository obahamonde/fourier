import logging
from functools import cached_property
from typing import Any

from boto3 import client # type: ignore
from openai._utils._proxy import LazyProxy

logger = logging.getLogger(__name__)

BUCKET_NAME = "terabytes"


class ObjectStorage(LazyProxy[Any]):
    def __load__(self):
        return client(service_name="s3")

    @cached_property
    def api(self):
        return self.__load__()

 
    def put_object(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        data = self.api.put_object( # type: ignore
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="audio/wav",
            ContentDisposition="inline",
        )


    def generate_presigned_url(
        self, *, key: str, bucket: str = BUCKET_NAME, ttl: int = 3600
    ):
        return self.api.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    def put(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        try:
            self.put_object(key=key, data=data, bucket=bucket)
            return self.generate_presigned_url(key=key, bucket=bucket)
        except Exception as e:
            logger.error(f"Failed to put object: {e.__class__.__name__}: {e}")
            raise e

    def get(self, *, key: str, bucket: str = BUCKET_NAME):
        return self.generate_presigned_url(key=key, bucket=bucket)
