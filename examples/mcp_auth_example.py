"""Example MCP auth provider configurations.

Export one of these as ``mcp.auth.path`` in aegra.json:

    {
      "mcp": {
        "auth": {
          "path": "./examples/mcp_auth_example.py:mcp_auth"
        }
      }
    }

The auth provider can be any object that implements the FastMCP auth
provider interface.  Below are three common configurations using
FastMCP's built-in OIDCProxy.  Uncomment the one that matches your
setup and fill in the values.
"""

import os

from fastmcp.server.auth.oidc_proxy import OIDCProxy

# ---------------------------------------------------------------------------
# Option 1: Confidential client (most common)
# ---------------------------------------------------------------------------
mcp_auth = OIDCProxy(
    config_url="https://your-idp.com/.well-known/openid-configuration",
    client_id=os.environ.get("MCP_OIDC_CLIENT_ID", "your-client-id"),
    client_secret=os.environ.get("MCP_OIDC_CLIENT_SECRET", "your-client-secret"),
    audience="https://api.example.com",
    required_scopes=["openid", "profile", "email"],
    base_url=os.environ.get("SERVER_URL", "http://localhost:2026") + "/mcp",
)

# ---------------------------------------------------------------------------
# Option 2: Public PKCE client (no client_secret, e.g. SPA or CLI)
# ---------------------------------------------------------------------------
# mcp_auth = OIDCProxy(
#     config_url="https://your-idp.com/.well-known/openid-configuration",
#     client_id=os.environ.get("MCP_OIDC_CLIENT_ID", "your-public-client-id"),
#     jwt_signing_key=os.environ["MCP_OIDC_JWT_SIGNING_KEY"],
#     token_endpoint_auth_method="none",
#     base_url=os.environ.get("SERVER_URL", "http://localhost:2026") + "/mcp",
# )

# ---------------------------------------------------------------------------
# Option 3: Advanced — custom token verifier, extra OAuth params
# ---------------------------------------------------------------------------
# from fastmcp.server.auth.providers.introspection import IntrospectionTokenVerifier
#
# verifier = IntrospectionTokenVerifier(
#     introspection_endpoint="https://your-idp.com/oauth2/introspect",
#     client_id="your-client-id",
#     client_secret=os.environ["MCP_OIDC_CLIENT_SECRET"],
# )
#
# mcp_auth = OIDCProxy(
#     config_url="https://your-idp.com/.well-known/openid-configuration",
#     client_id="your-client-id",
#     client_secret=os.environ["MCP_OIDC_CLIENT_SECRET"],
#     base_url=os.environ.get("SERVER_URL", "http://localhost:2026") + "/mcp",
#     token_verifier=verifier,
#     extra_authorize_params={"prompt": "consent", "access_type": "offline"},
#     allowed_client_redirect_uris=["http://localhost:*", "https://*.example.com/*"],
# )
