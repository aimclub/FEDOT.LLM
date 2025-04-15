import logging
from pathlib import Path
CONVERSATION_BASE_DIR = "sessions"

logger = logging.getLogger(__name__)


def get_conversation_dir(sid: str, user_id: str | None = None) -> str:
    if user_id:
        return f"users/{user_id}/conversations/{sid}/"
    else:
        return f"{CONVERSATION_BASE_DIR}/{sid}/"


def get_conversation_events_dir(sid: str, user_id: str | None = None) -> str:
    return f"{get_conversation_dir(sid, user_id)}events/"

def get_id_from_filename(filename: Path) -> int:
    try:
        return int(filename.name.split(".")[0])
    except ValueError:
        logger.warning(f"get id from filename ({filename}) failed.")
        return -1
    
def get_filename_for_id(
    sid: str, id: int, user_id: str | None = None
) -> str:
    return f'{get_conversation_events_dir(sid, user_id)}{id}.json'
