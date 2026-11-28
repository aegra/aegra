# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Aegra, please report it privately via
[GitHub Security Advisories](https://github.com/aegra/aegra/security/advisories/new).

Do **not** open a public issue for security bugs. Public issues let attackers
weaponize a bug before a fix is available.

We will:

- Acknowledge receipt within 48 hours
- Confirm or dispute the vulnerability within 7 days
- Ship a patch release for confirmed high-severity bugs as soon as the fix is reviewed
- Credit you in the published advisory unless you prefer anonymity

## Supported Versions

We patch security bugs in the latest minor release. Older minors do not receive
security backports, upgrade to the latest patch on each minor.

| Version | Supported |
| ------- | --------- |
| 0.13.x  | Yes       |
| < 0.13  | No        |

## Scope

**In scope:**

- `aegra-api` (core server)
- `aegra-cli`
- Official Docker images published from this repo
- Official examples in `examples/`

**Out of scope:**

- Third-party graphs and user-written `@auth.on` handlers
- Deployment infrastructure you operate (Postgres, Redis, K8s configs, reverse proxies)
- Vulnerabilities in upstream dependencies (report those to the upstream project)

## Past advisories

Published advisories are listed at:
https://github.com/aegra/aegra/security/advisories
