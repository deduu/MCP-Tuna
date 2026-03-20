from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from shared.persistence import get_persistence_service

logger = logging.getLogger(__name__)


class ObjectStorageService:
    """Optional S3-compatible object storage for durable blobs."""

    def __init__(self) -> None:
        self._settings = settings.object_storage
        self._client: Any = None
        self._client_failed = False

    @property
    def enabled(self) -> bool:
        return self._settings.enabled and not self._client_failed

    def build_key(self, category: str, relative_path: str) -> str:
        cleaned = relative_path.replace("\\", "/").strip().lstrip("/")
        prefix_parts = [part for part in (self._settings.prefix, category, cleaned) if part]
        return "/".join(prefix_parts)

    async def upload_file(
        self,
        local_path: str,
        *,
        category: str,
        relative_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"success": False, "disabled": True}

        path = Path(local_path)
        if not path.exists() or not path.is_file():
            return {"success": False, "error": f"File not found: {local_path}"}

        key = self.build_key(category, relative_path or path.name)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        try:
            await asyncio.to_thread(self._upload_file_sync, path, key, content_type)
        except Exception as exc:
            logger.warning("Object storage upload failed for %s: %s", local_path, exc)
            return {"success": False, "error": str(exc)}

        object_url = self._object_url(key)
        artifact_key = f"{category}:{key}"
        await get_persistence_service().upsert_artifact(
            {
                "artifact_key": artifact_key,
                "kind": category,
                "local_path": str(path.resolve()),
                "bucket": self._settings.bucket,
                "object_key": key,
                "object_url": object_url,
                "metadata": {"content_type": content_type},
            }
        )
        return {
            "success": True,
            "bucket": self._settings.bucket,
            "object_key": key,
            "object_url": object_url,
        }

    async def delete_object(self, object_key: Optional[str]) -> Dict[str, Any]:
        if not object_key:
            return {"success": False, "error": "object_key is required"}
        if not self.enabled:
            return {"success": False, "disabled": True}
        try:
            await asyncio.to_thread(self._delete_object_sync, object_key)
            return {"success": True, "object_key": object_key}
        except Exception as exc:
            logger.warning("Object storage delete failed for %s: %s", object_key, exc)
            return {"success": False, "error": str(exc)}

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_failed:
            raise RuntimeError("Object storage client initialization previously failed")

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            self._client_failed = True
            raise RuntimeError(
                "Object storage requires boto3/botocore. Install the backend extras."
            ) from exc

        self._client = boto3.client(
            "s3",
            endpoint_url=self._settings.endpoint,
            aws_access_key_id=self._settings.access_key,
            aws_secret_access_key=self._settings.secret_key,
            region_name=self._settings.region,
            use_ssl=self._settings.secure,
            config=Config(s3={"addressing_style": "path"}),
        )
        if self._settings.auto_create_bucket:
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        client = self._client
        assert client is not None
        try:
            client.head_bucket(Bucket=self._settings.bucket)
        except Exception:
            params: Dict[str, Any] = {"Bucket": self._settings.bucket}
            if self._settings.region and self._settings.region != "us-east-1":
                params["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._settings.region,
                }
            client.create_bucket(**params)

    def _upload_file_sync(self, path: Path, key: str, content_type: str) -> None:
        client = self._get_client()
        extra_args = {"ContentType": content_type}
        client.upload_file(str(path), self._settings.bucket, key, ExtraArgs=extra_args)

    def _delete_object_sync(self, object_key: str) -> None:
        client = self._get_client()
        client.delete_object(Bucket=self._settings.bucket, Key=object_key)

    def _object_url(self, key: str) -> str:
        if self._settings.public_url:
            base = self._settings.public_url.rstrip("/")
            return f"{base}/{self._settings.bucket}/{key}"
        base = self._settings.endpoint.rstrip("/")
        return f"{base}/{self._settings.bucket}/{key}"


_object_storage_service = ObjectStorageService()


def get_object_storage_service() -> ObjectStorageService:
    return _object_storage_service
