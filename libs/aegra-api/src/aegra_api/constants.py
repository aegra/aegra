import re
from uuid import UUID

# Standard namespace UUID for deriving deterministic assistant IDs from graph IDs.
# IMPORTANT: Do not change after initial deploy unless you plan a data migration.
ASSISTANT_NAMESPACE_UUID = UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")

# Regex to decompose a PostgreSQL URL into its components.
# Used by DatabaseSettings._to_sqlalchemy_multihost() to detect and rewrite
# comma-separated host lists into SQLAlchemy query-param format.
MULTIHOST_URL_RE = re.compile(
    r"^(?P<scheme>[^:]+://)"
    r"(?:(?P<userinfo>.+)@)?"
    r"(?P<hostlist>[^/?]+)"
    r"(?:/(?P<path>[^?]*))?"
    r"(?:\?(?P<query>.+))?$"
)
