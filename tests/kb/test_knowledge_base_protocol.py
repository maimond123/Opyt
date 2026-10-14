"""Both knowledge bases meet the read contract (A3). The project has no type checker, so this is
the check that `reads.KnowledgeBase` is what `LocalKB` and `ServedKB` actually implement: every
method the Protocol names exists on both, and takes the Protocol's positional parameters.
"""
from __future__ import annotations

import inspect

import pytest

from pipeline.retrieval.local import LocalKB
from pipeline.retrieval.reads import KnowledgeBase
from pipeline.retrieval.served import ServedKB

METHODS = [name for name, member in vars(KnowledgeBase).items()
           if callable(member) and not name.startswith("_")]


def _positional(fn) -> list[str]:
    return [p.name for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def test_the_protocol_names_the_four_methods():
    assert sorted(METHODS) == ["aggregate", "close", "open_atom", "search"]


@pytest.mark.parametrize("impl", [LocalKB, ServedKB], ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", METHODS)
def test_each_knowledge_base_has_the_method_with_its_positional_parameters(impl, method):
    assert callable(getattr(impl, method, None)), f"{impl.__name__} has no {method}()"
    want = _positional(getattr(KnowledgeBase, method))
    assert _positional(getattr(impl, method))[:len(want)] == want
