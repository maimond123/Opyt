"""
pipeline/sharing/agent_invite.py — what an agent invite gives and how it is handed over, in the
sentences both `share`s say.

An agent invite is an invite whose code buys an agent key: the user's own agent then acts as them
over plain HTTPS, where a reader only reads. The Mac's `share` and the connector's word it here,
so the consent a person hears is the same from either. Plain values in, plain values out.
"""
from __future__ import annotations

# Read to the user before an agent invite is minted: what their yes covers.
CONSENT = (
    "The agent that redeems this link acts as the user: it can search the whole library, save "
    "links to it, remove items from it, add people to follow, and invite or cut off readers.",
    "It is standing access, not a one-time pass: the agent keeps it until it is cut off with "
    "`unshare`.",
    "The link works once, and whoever redeems it gets this access, so it goes to the agent and "
    "to nobody else.",
)


def handover(service_url: str) -> str:
    """What to do with a minted agent invite: who gets the link, and how the agent redeems it at
    the service that minted it."""
    return ("Give the user this link for their agent, and nobody else. The agent redeems it "
            f"once, as https://useopyt.com/llms.txt describes: POST {service_url}/v1/redeem "
            'with the JSON body {"code": "<the part of the link after #>"}, and keeps the '
            "`owner` and `token` it gets back.")
