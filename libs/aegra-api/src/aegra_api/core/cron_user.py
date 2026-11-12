"""Load the resolver that picks the user a scheduled cron firing runs as.

A firing has no request to authenticate, so ``auth.cron_user`` in aegra.json can
name an async function that maps the cron owner's identity to the user the
firing runs as. It returns the same shape ``@auth.authenticate`` returns, or
None to disable the cron.
"""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import structlog

from aegra_api.config import load_auth_config
from aegra_api.core.auth_middleware import import_auth_target, split_import_path
from aegra_api.models.auth import User

logger = structlog.getLogger(__name__)

CronUserResolver = Callable[[str], Awaitable[User | Mapping[str, Any] | None]]


def load_cron_user_resolver() -> CronUserResolver | None:
    """Return the async function ``auth.cron_user`` names, or None when it names none.

    Raises instead of falling back, so a server whose resolver does not load stops
    at startup rather than firing crons as the bare scheduler user.

    Raises:
        ValueError: If the path has no ``:`` before the function name
        FileNotFoundError: If the named file does not exist
        ImportError: If the named module cannot be imported
        AttributeError: If the module defines no such name
        TypeError: If the name is not an async function
    """
    auth_config = load_auth_config()
    path = auth_config.get("cron_user") if auth_config else None
    if not path:
        return None

    target, name = split_import_path(path)
    module = import_auth_target(target)

    resolver = getattr(module, name, None)
    if resolver is None:
        raise AttributeError(f"auth.cron_user {path!r}: {name!r} not found")
    if not inspect.iscoroutinefunction(resolver):
        raise TypeError(f"auth.cron_user {path!r} must name an async function")

    logger.info("Loaded cron user resolver", path=path)
    return resolver
