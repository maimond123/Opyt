"""service/ — the process that serves one person's knowledge base to another over HTTPS.

DELIBERATELY NOT PART OF THE CLIENT WHEEL. `pyproject.toml`'s packages-find list does not
include `service*`, and FastAPI/uvicorn live under the `server` optional-dependency group, so
installing OPYT to read a knowledge base never pulls in a web framework to serve one.

Design record: docs/plans/2026-08-26-foreign-kb-service-phase3.md.
"""
