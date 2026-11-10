"""Cloudflare R2 (S3-compatible) settings for call recordings, shared by the agent and the API."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> "R2Config | None":
        """The R2 settings from the environment, or None if any of them is missing.

        The endpoint comes from R2_ENDPOINT, or is built from R2_ACCOUNT_ID when that's unset.
        """
        endpoint = os.getenv("R2_ENDPOINT", "").strip().rstrip("/")
        if not endpoint and (account_id := os.getenv("R2_ACCOUNT_ID", "").strip()):
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        values = {
            "endpoint": endpoint,
            "access_key_id": os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            "bucket": os.getenv("R2_BUCKET", "").strip(),
        }
        return cls(**values) if all(values.values()) else None


def recording_key(room_name: str) -> str:
    return f"recordings/{room_name}.ogg"
