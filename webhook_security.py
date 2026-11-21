"""
webhook_security.py
Safety rules for the outbound webhook (approved result -> the CRM's endpoint):

1. validate_webhook_url(): a tenant-supplied URL must not let the server be pointed at its own
   internal network (SSRF): https only, no credentials in the URL, and every address the host
   resolves to must be a public one. Checked when the organization is registered AND again right
   before each delivery (DNS can change).
2. Signing: every delivery carries `X-Webhook-Timestamp` and `X-Webhook-Signature: sha256=<hex>`,
   an HMAC-SHA256 over f"{timestamp}.{raw_body}" with a per-organization secret derived from one
   server-side master value (config.WEBHOOK_SIGNING_SECRET, falling back to INTERNAL_API_KEY).
   The CRM verifies the signature (and rejects old timestamps) so a forged or replayed request is
   recognised. Deriving the secret means no extra database column is needed.

Receiver-side verification, in Python:

    expected = "sha256=" + hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, "sha256").hexdigest()
    ok = hmac.compare_digest(expected, request.headers["X-Webhook-Signature"]) and abs(time.time() - int(ts)) < 300

Residual risk (documented, not solved here): between the DNS check and the connection a hostile DNS
server could rebind the name to a private address. Combine this with an egress firewall rule that
blocks the worker host from reaching internal ranges if that matters in your deployment.
"""

import hashlib
import hmac
import ipaddress
import socket
import time
from typing import Optional
from urllib.parse import urlparse

import config

SIGNATURE_HEADER = "X-Webhook-Signature"
TIMESTAMP_HEADER = "X-Webhook-Timestamp"


class WebhookUrlError(ValueError):
    """The webhook URL is not acceptable (message is safe to show to the API caller)."""


def _is_public(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:127.0.0.1 and friends
        ip = mapped
    return ip.is_global and not ip.is_multicast


def validate_webhook_url(url: str, resolve: bool = True) -> str:
    """Returns the cleaned URL, or raises WebhookUrlError. resolve=False skips the DNS lookup
    (syntax-only check)."""
    if not isinstance(url, str) or not url.strip():
        raise WebhookUrlError("webhook_url must be a non-empty string.")
    url = url.strip()
    if len(url) > 1024:
        raise WebhookUrlError("webhook_url is too long (max 1024 characters).")
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        raise WebhookUrlError("webhook_url is not a valid URL.")
    allow_private = config.WEBHOOK_ALLOW_PRIVATE_URLS
    allowed_scheme = ("http", "https") if allow_private else ("https",)
    if parsed.scheme not in allowed_scheme:
        raise WebhookUrlError("webhook_url must use https." if not allow_private else "webhook_url must use http or https.")
    if not host:
        raise WebhookUrlError("webhook_url has no host.")
    if parsed.username or parsed.password:
        raise WebhookUrlError("webhook_url must not contain credentials.")
    if allow_private or host in config.WEBHOOK_ALLOWED_HOSTS or not resolve:
        return url
    try:
        infos = socket.getaddrinfo(host, port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise WebhookUrlError("webhook_url host could not be resolved.")
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise WebhookUrlError("webhook_url host could not be resolved.")
    for address in addresses:
        if not _is_public(address.split("%")[0]):
            raise WebhookUrlError("webhook_url must point to a public address (private, loopback and link-local addresses are refused).")
    return url


def _master_secret() -> str:
    if config.IS_PRODUCTION:
        return config.WEBHOOK_SIGNING_SECRET   # never derived from the admin key in production
    return config.WEBHOOK_SIGNING_SECRET or config.INTERNAL_API_KEY


def derive_org_secret(organization_id: str) -> Optional[str]:
    """The signing secret of one organization, or None if no master secret is configured."""
    master = _master_secret()
    if not master:
        return None
    return hmac.new(master.encode("utf-8"), f"webhook:{organization_id}".encode("utf-8"), hashlib.sha256).hexdigest()


def sign(secret: str, timestamp: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def signature_headers(organization_id: str, body: bytes, now: Optional[float] = None) -> dict:
    """{} if signing is not configured, else the two headers to add to the request."""
    secret = derive_org_secret(organization_id)
    if not secret:
        return {}
    timestamp = str(int(now if now is not None else time.time()))
    return {TIMESTAMP_HEADER: timestamp, SIGNATURE_HEADER: sign(secret, timestamp, body)}
