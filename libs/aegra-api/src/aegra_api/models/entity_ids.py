"""Shared bounds for client-provided entity ids."""

# Stay well under PostgreSQL btree's ~2704-byte index tuple cap even uncompressed.
# LangGraph SDK has no cap; PostgresSaver docs recommend 255 characters.
MAX_ENTITY_ID_LENGTH = 255

# Rejects all-whitespace ids while keeping inner spaces; lands in the OpenAPI schema,
# which a bare validator would not.
ENTITY_ID_PATTERN = r"\S"
