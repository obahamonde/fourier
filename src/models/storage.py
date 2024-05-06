import os
from functools import cached_property
from typing import Any

from agent_proto import async_io, robust
from agent_proto.proxy import LazyProxy
from agent_proto.utils import setup_logging
from boto3 import client

logger = setup_logging(__name__)

BUCKET_NAME = "terabytes"


class ObjectStorage(LazyProxy[Any]):

    def __load__(self):
        return client(service_name="s3")

    @cached_property
    def api(self):
        return self.__load__()

    @async_io
    def put_object(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        data = self.api.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="audio/wav",
            ContentDisposition="inline",
        )

    @async_io
    def generate_presigned_url(
        self, *, key: str, bucket: str = BUCKET_NAME, ttl: int = 3600
    ):
        return self.api.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    async def put(self, *, key: str, data: bytes, bucket: str = BUCKET_NAME):
        try:
            await self.put_object(key=key, data=data, bucket=bucket)
            return await self.generate_presigned_url(key=key, bucket=bucket)
        except Exception as e:
            logger.error(f"Failed to put object: {e.__class__.__name__}: {e}")
            raise e

    async def get(self, *, key: str, bucket: str = BUCKET_NAME):
        return await self.generate_presigned_url(key=key, bucket=bucket)
