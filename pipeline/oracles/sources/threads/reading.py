"""The thread lane's cached session and image inputs."""

from pipeline.image_cache import load_image_cache
from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms import sessions
from opyt_core.paths import opyt_home
from pipeline.platforms.x import conversation as x_conversation


def page(ctx, focal: str) -> list[dict]:
    if "session" not in ctx.cache:
        ctx.cache["session"] = sessions.x_session("https://x.com/home")
    session = ctx.cache["session"]
    try:
        return x_conversation.fetch_thread_page(focal, session)
    except SyncAuthError:
        ctx.cache.pop("session", None)
        raise


def images(ctx) -> dict:
    if "images" not in ctx.cache:
        ctx.cache["images"] = load_image_cache(opyt_home())
    else:
        ctx.cache["images"].update(load_image_cache(opyt_home()))
    return ctx.cache["images"]
