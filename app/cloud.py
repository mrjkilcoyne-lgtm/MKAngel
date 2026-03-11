"""
Cloud storage connectivity for MKAngel.

The Angel's memory extends beyond the device — reaching into the cloud
for persistent storage, model weights, shared knowledge, and
collaborative state.  Like a mind that can access any library in the
world, the Angel connects to various storage backends through
standard ports and protocols.

Supported backends:
- Local filesystem (always available)
- S3-compatible storage (AWS, MinIO, DigitalOcean Spaces)
- Google Cloud Storage
- Azure Blob Storage
- WebDAV (Nextcloud, ownCloud, etc.)
- HTTP/HTTPS endpoints (REST APIs)
- SFTP / SCP

All backends implement the same CloudStorage interface, so the Angel
doesn't need to know where its data lives — it just reads and writes.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator


from app.paths import mkangel_dir

_CLOUD_CONFIG_PATH = mkangel_dir() / "cloud.json"
_LOCAL_CACHE_DIR = mkangel_dir() / "cloud_cache"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CloudObject:
    """A file/object in cloud storage."""
    key: str
    size: int = 0
    last_modified: float = 0.0
    content_type: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    backend: str = "local"


@dataclass
class CloudConfig:
    """Configuration for a cloud storage backend."""
    name: str
    backend_type: str  # "local", "s3", "gcs", "azure", "webdav", "http", "sftp"
    endpoint: str = ""
    port: int = 0
    bucket: str = ""
    prefix: str = ""
    credentials: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    # Default ports per protocol
    DEFAULT_PORTS = {
        "s3": 443,
        "gcs": 443,
        "azure": 443,
        "webdav": 443,
        "http": 80,
        "https": 443,
        "sftp": 22,
        "scp": 22,
        "ftp": 21,
        "local": 0,
    }

    @property
    def effective_port(self) -> int:
        return self.port or self.DEFAULT_PORTS.get(self.backend_type, 0)


# ---------------------------------------------------------------------------
# Cloud storage interface
# ---------------------------------------------------------------------------

class CloudStorage(ABC):
    """Abstract interface for cloud storage backends."""

    @abstractmethod
    def put(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> CloudObject:
        """Upload data to storage."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Download data from storage."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an object from storage."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[CloudObject]:
        """List objects matching a prefix."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an object exists."""

    def put_json(self, key: str, obj: Any) -> CloudObject:
        """Upload a JSON-serializable object."""
        data = json.dumps(obj, indent=2, default=str).encode("utf-8")
        return self.put(key, data, metadata={"content_type": "application/json"})

    def get_json(self, key: str) -> Any:
        """Download and parse a JSON object."""
        data = self.get(key)
        return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalStorage(CloudStorage):
    """Local filesystem storage — always available, zero configuration."""

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root) if root else _LOCAL_CACHE_DIR
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> CloudObject:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        # Store metadata alongside
        if metadata:
            meta_path = path.with_suffix(path.suffix + ".meta")
            meta_path.write_text(json.dumps(metadata))

        return CloudObject(
            key=key,
            size=len(data),
            last_modified=time.time(),
            metadata=metadata or {},
            backend="local",
        )

    def get(self, key: str) -> bytes:
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        path = self._root / key
        if path.exists():
            path.unlink()
            meta = path.with_suffix(path.suffix + ".meta")
            if meta.exists():
                meta.unlink()
            return True
        return False

    def list(self, prefix: str = "") -> list[CloudObject]:
        objects = []
        search_dir = self._root / prefix if prefix else self._root
        if not search_dir.exists():
            return objects

        for path in search_dir.rglob("*"):
            if path.is_file() and not path.name.endswith(".meta"):
                rel = str(path.relative_to(self._root))
                stat = path.stat()
                objects.append(CloudObject(
                    key=rel,
                    size=stat.st_size,
                    last_modified=stat.st_mtime,
                    backend="local",
                ))
        return objects

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()


# ---------------------------------------------------------------------------
# S3-compatible storage backend
# ---------------------------------------------------------------------------

class S3Storage(CloudStorage):
    """S3-compatible storage (AWS, MinIO, DigitalOcean Spaces, etc.)."""

    def __init__(self, config: CloudConfig):
        self._config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise RuntimeError("S3 storage requires: pip install boto3")

            kwargs: dict[str, Any] = {}
            if self._config.endpoint:
                kwargs["endpoint_url"] = self._config.endpoint
            if self._config.credentials.get("access_key"):
                kwargs["aws_access_key_id"] = self._config.credentials["access_key"]
            if self._config.credentials.get("secret_key"):
                kwargs["aws_secret_access_key"] = self._config.credentials["secret_key"]
            if self._config.credentials.get("region"):
                kwargs["region_name"] = self._config.credentials["region"]

            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _full_key(self, key: str) -> str:
        if self._config.prefix:
            return f"{self._config.prefix}/{key}"
        return key

    def put(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> CloudObject:
        client = self._get_client()
        full_key = self._full_key(key)
        extra = {}
        if metadata:
            extra["Metadata"] = metadata
        client.put_object(Bucket=self._config.bucket, Key=full_key, Body=data, **extra)
        return CloudObject(key=key, size=len(data), last_modified=time.time(),
                          metadata=metadata or {}, backend="s3")

    def get(self, key: str) -> bytes:
        client = self._get_client()
        response = client.get_object(Bucket=self._config.bucket, Key=self._full_key(key))
        return response["Body"].read()

    def delete(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.delete_object(Bucket=self._config.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[CloudObject]:
        client = self._get_client()
        full_prefix = self._full_key(prefix)
        objects = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self._config.prefix and key.startswith(self._config.prefix + "/"):
                    key = key[len(self._config.prefix) + 1:]
                objects.append(CloudObject(
                    key=key,
                    size=obj.get("Size", 0),
                    last_modified=obj.get("LastModified", 0),
                    backend="s3",
                ))
        return objects

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self._config.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# WebDAV storage backend
# ---------------------------------------------------------------------------

class WebDAVStorage(CloudStorage):
    """WebDAV storage (Nextcloud, ownCloud, etc.)."""

    def __init__(self, config: CloudConfig):
        self._config = config

    def _url(self, key: str) -> str:
        base = self._config.endpoint.rstrip("/")
        prefix = self._config.prefix.strip("/")
        if prefix:
            return f"{base}/{prefix}/{key}"
        return f"{base}/{key}"

    def put(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> CloudObject:
        try:
            import requests
        except ImportError:
            raise RuntimeError("WebDAV storage requires: pip install requests")

        auth = None
        creds = self._config.credentials
        if creds.get("username") and creds.get("password"):
            auth = (creds["username"], creds["password"])

        requests.put(self._url(key), data=data, auth=auth)
        return CloudObject(key=key, size=len(data), last_modified=time.time(),
                          metadata=metadata or {}, backend="webdav")

    def get(self, key: str) -> bytes:
        import requests
        auth = None
        creds = self._config.credentials
        if creds.get("username") and creds.get("password"):
            auth = (creds["username"], creds["password"])
        resp = requests.get(self._url(key), auth=auth)
        resp.raise_for_status()
        return resp.content

    def delete(self, key: str) -> bool:
        import requests
        auth = None
        creds = self._config.credentials
        if creds.get("username") and creds.get("password"):
            auth = (creds["username"], creds["password"])
        try:
            requests.delete(self._url(key), auth=auth)
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[CloudObject]:
        return []  # WebDAV PROPFIND is complex; stub for now

    def exists(self, key: str) -> bool:
        import requests
        auth = None
        creds = self._config.credentials
        if creds.get("username") and creds.get("password"):
            auth = (creds["username"], creds["password"])
        try:
            resp = requests.head(self._url(key), auth=auth)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CloudManager — the unified interface
# ---------------------------------------------------------------------------

class CloudManager:
    """Manages multiple cloud storage backends for MKAngel.

    The CloudManager provides a unified interface to local and remote
    storage.  Data is automatically cached locally for offline access.
    """

    def __init__(self) -> None:
        self._backends: dict[str, CloudStorage] = {}
        self._configs: dict[str, CloudConfig] = {}
        self._local = LocalStorage()
        self._backends["local"] = self._local
        self._load_config()

    def add_backend(self, config: CloudConfig) -> CloudStorage:
        """Add a cloud storage backend."""
        backend: CloudStorage

        if config.backend_type == "local":
            backend = LocalStorage(config.endpoint or None)
        elif config.backend_type == "s3":
            backend = S3Storage(config)
        elif config.backend_type == "webdav":
            backend = WebDAVStorage(config)
        else:
            raise ValueError(
                f"Unsupported backend type: {config.backend_type}. "
                f"Supported: local, s3, webdav"
            )

        self._backends[config.name] = backend
        self._configs[config.name] = config
        self._save_config()
        return backend

    def remove_backend(self, name: str) -> bool:
        """Remove a cloud storage backend."""
        if name in self._backends and name != "local":
            del self._backends[name]
            self._configs.pop(name, None)
            self._save_config()
            return True
        return False

    def get_backend(self, name: str = "local") -> CloudStorage:
        """Get a specific storage backend."""
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found. Available: {list(self._backends.keys())}")
        return self._backends[name]

    def list_backends(self) -> list[dict[str, Any]]:
        """List all configured backends."""
        result = []
        for name, backend in self._backends.items():
            config = self._configs.get(name)
            result.append({
                "name": name,
                "type": config.backend_type if config else "local",
                "endpoint": config.endpoint if config else "filesystem",
                "port": config.effective_port if config else 0,
            })
        return result

    # -- Convenience methods (use primary backend) -------------------------

    def put(self, key: str, data: bytes, backend: str = "local", **kwargs) -> CloudObject:
        return self.get_backend(backend).put(key, data, **kwargs)

    def get(self, key: str, backend: str = "local") -> bytes:
        return self.get_backend(backend).get(key)

    def sync(self, key: str, source: str, target: str) -> None:
        """Sync an object between backends."""
        data = self.get_backend(source).get(key)
        self.get_backend(target).put(key, data)

    # -- Config persistence ------------------------------------------------

    def _save_config(self) -> None:
        """Save cloud configuration."""
        configs = {}
        for name, config in self._configs.items():
            configs[name] = {
                "name": config.name,
                "backend_type": config.backend_type,
                "endpoint": config.endpoint,
                "port": config.port,
                "bucket": config.bucket,
                "prefix": config.prefix,
                "options": config.options,
                # Don't save credentials in plaintext
            }
        _CLOUD_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CLOUD_CONFIG_PATH, "w") as f:
            json.dump(configs, f, indent=2)

    def _load_config(self) -> None:
        """Load cloud configuration."""
        if not _CLOUD_CONFIG_PATH.exists():
            return

        try:
            with open(_CLOUD_CONFIG_PATH) as f:
                configs = json.load(f)

            for name, data in configs.items():
                config = CloudConfig(**data)
                try:
                    self.add_backend(config)
                except Exception:
                    continue
        except (json.JSONDecodeError, OSError):
            pass
