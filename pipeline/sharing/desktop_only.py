"""
pipeline/sharing/desktop_only.py — what Opyt does only where it is installed, in one sentence.

Said to a user wherever they meet Opyt in a browser: when they connect their claude.ai or
ChatGPT (`link.py`), and by the connector itself (`service/connector.py`). One sentence, written
for someone who knows nothing about how Opyt works. The website says the same
(`website/connector.html`, `website/llms.txt`); change them together.
"""
from __future__ import annotations

SENTENCE = ("Some things only work in the desktop app where Opyt is installed, not in "
            "claude.ai, ChatGPT or their phone apps: setting up Opyt, connecting X or Substack, "
            "picking people from who you follow, searching X and Substack live, reviewing new "
            "finds on your topics, and reading a paper's full text.")
