# Security

## Reporting a vulnerability

Report vulnerabilities through [GitHub's private vulnerability reporting](https://github.com/trevinbrooks/pearlarr/security/advisories/new) (the Security tab, "Report a vulnerability").
Do not open a public issue for a security problem.

Only the latest release receives fixes.

## Threat model

Pearlarr is outbound-only automation: it listens on no ports and serves nothing.
Its config file holds credentials for your own services - Sonarr/Radarr API keys, the qBittorrent WebUI login, webhook URLs - so the file is created owner-only (`0600`), and a warning fires at load when an existing config is readable by group or other.
Outbound connections go to your arr and qBittorrent instances plus a small set of public endpoints: SeaDex, AniList, the mapping sources on GitHub, tracker pages for grabbed releases, and any webhook you configure - the complete list is the external-hosts table in [docs/architecture.md](docs/architecture.md#external-hosts).

## The redaction guarantee

Log files, `pearlarr config show` output, and error text never contain API keys, passwords, webhook URLs, or logins embedded in URLs - at any log level.
This is a tested promise, so pasting logs or `config show` output into a bug report is safe by design.

## No telemetry

Pearlarr sends nothing anywhere except the services you configure.
There is no telemetry, no update check, and no crash reporting.
