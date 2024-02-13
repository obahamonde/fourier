import os
from functools import cached_property
from typing import Any

from agent_proto import async_io, robust
from agent_proto.proxy import LazyProxy
from agent_proto.utils import setup_logging
from boto3 import client

logger = setup_logging(__name__)

class ObjectStorage(LazyProxy[Any]):
    def __load__(self):
        return client(
            service_name="s3",
            endpoint_url=os.environ.get("MINIO_ENDPOINT"),
            aws_access_key_id=os.environ.get("MINIO_ROOT_USER"),
            aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD"),
            region_name="us-east-1",
        )

    @cached_property
    def minio(self):
        return self.__load__()

    @async_io
    def put_object(
        self, *, key: str, data: bytes, bucket: str = "tera"
    ):
        self.minio.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="audio/wav",
            ACL="public-read",
            ContentDisposition="inline",
        )

    @async_io
    def generate_presigned_url(
        self, *, key: str, bucket: str = "tera", ttl: int = 3600
    ):
        return self.minio.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    @robust
    async def put(self, *, key: str, data:bytes, bucket: str = "tera"):
        await self.put_object(
            key=key, data=data, bucket=bucket
        )
        return await self.generate_presigned_url(key=key, bucket=bucket)

    @robust
    async def get(self, *, key: str, bucket: str = "tera"):
        return await self.generate_presigned_url(key=key, bucket=bucket)
