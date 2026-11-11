"""Call recording with LiveKit Egress: one mixed audio-only OGG per room, uploaded to R2."""

import logging
import os
from dataclasses import dataclass

from livekit import api

logger = logging.getLogger("talkbalt-agent")


@dataclass(frozen=True)
class R2Config:
    """Cloudflare R2 (S3-compatible) bucket the egress uploads to. The platform API has its own
    copy of these settings to hand out playback links."""

    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> "R2Config | None":
        """From R2_ENDPOINT (or R2_ACCOUNT_ID), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET; None if any is missing."""
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
    """Object key of a room's recording; the platform API presigns this key for playback."""
    return f"recordings/{room_name}.ogg"


@dataclass
class RecordingResult:
    status: str  # recording | failed
    key: str | None = None
    egress_id: str | None = None


async def start_recording(lkapi: api.LiveKitAPI, room_name: str) -> RecordingResult:
    """Starts an audio-only room composite egress. Never raises: failures return status=failed.

    Uses StartEgress with an audio-only TemplateSource (the current replacement for the
    deprecated RoomComposite egress). No layout or custom_base_url is set, which keeps it on the
    audio-only billing rate. The egress follows the room's lifecycle and stops when the room ends.
    """
    r2 = R2Config.from_env()
    if r2 is None:
        logger.warning("recording not started: R2_ENDPOINT (or R2_ACCOUNT_ID), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY or R2_BUCKET not set")
        return RecordingResult(status="failed")

    key = recording_key(room_name)
    request = api.StartEgressRequest(
        room_name=room_name,
        template=api.TemplateSource(audio_only=True),
        outputs=[api.Output(file=api.FileOutput(file_type=api.EncodedFileType.OGG, filepath=key))],
        storage=api.StorageConfig(
            s3=api.S3Upload(
                access_key=r2.access_key_id,
                secret=r2.secret_access_key,
                bucket=r2.bucket,
                region="auto",
                endpoint=r2.endpoint,
                force_path_style=True,  # required for non-AWS S3 providers
            )
        ),
    )
    try:
        info = await lkapi.egress.start_egress(request)
    except Exception as e:
        # e.g. the Build plan's monthly egress allowance is used up
        logger.warning("recording not started: egress failed to start (%s)", e)
        return RecordingResult(status="failed", key=key)

    logger.info("recording started: egress_id=%s key=%s", info.egress_id, key)
    return RecordingResult(status="recording", key=key, egress_id=info.egress_id)


async def final_recording_status(lkapi: api.LiveKitAPI, egress_id: str) -> str:
    """Recording status at call end. The upload often finishes after the call, so an egress that
    is still ending reports `processing`; the API marks it completed once the file exists."""
    try:
        res = await lkapi.egress.list_egress(api.ListEgressRequest(egress_id=egress_id))
    except Exception as e:
        logger.warning("could not read egress status: %s", e)
        return "processing"
    if not res.items:
        return "processing"
    status = res.items[0].status
    if status == api.EgressStatus.EGRESS_COMPLETE:
        return "completed"
    if status in (
        api.EgressStatus.EGRESS_FAILED,
        api.EgressStatus.EGRESS_ABORTED,
        api.EgressStatus.EGRESS_LIMIT_REACHED,
    ):
        return "failed"
    return "processing"
