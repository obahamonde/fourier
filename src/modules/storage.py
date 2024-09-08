import logging

from boto3 import client  # type: ignore
from botocore.client import Config
from openai._utils._proxy import LazyProxy

from ..utils import asyncify

logger = logging.getLogger(__name__)

BUCKET_NAME = "terabytes"


class ObjectStorage(LazyProxy[object]):
    def __load__(self):
        return client(
            service_name="s3",
            endpoint_url="https://storage.indiecloud.co",
            config=Config(signature_version="s3v4"),
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadminpassword",
        )

    @asyncify
    def put_object(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        data = self.__load__().put_object(  # type: ignore
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="audio/wav",
            ContentDisposition="inline",
        )

    @asyncify
    def get(self, *, key: str, bucket: str = BUCKET_NAME, ttl: int = 3600):
        return self.__load__().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    async def put(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        try:
            await self.put_object(key=key, data=data, bucket=bucket)
            return await self.get(key=key, bucket=bucket)
        except Exception as e:
            logger.error(f"Failed to put object: {e.__class__.__name__}: {e}")
            raise e
