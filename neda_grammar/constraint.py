"""E004 core: token-level trie constraint for ALFWorld-style action decoding.

Design: see DESIGN.md. v1 = dynamic trie over admissible commands +
left-to-right commit inside the action span. Pure python, no torch needed
for the core (logits masking returns index sets; the engine hook applies
them to tensors).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


class TokenTrie:
    """Prefix tree over tokenized legal commands.

    Built per environment step from `admissible_commands` (each command
    tokenized to a list of token ids). Queries:
      - allowed_next(prefix): set of token ids that keep the prefix legal
      - is_complete(prefix): prefix exactly matches a full legal command
    """

    __slots__ = ("_root",)

    def __init__(self, tokenized_commands: Iterable[Sequence[int]]):
        # node = dict[token_id -> node]; END marker key = -1
        self._root: Dict[int, dict] = {}
        n = 0
        for seq in tokenized_commands:
            node = self._root
            for tok in seq:
                node = node.setdefault(tok, {})
            node[-1] = {}  # end-of-command marker
            n += 1
        if n == 0:
            raise ValueError("TokenTrie built from empty command set")

    def _walk(self, prefix: Sequence[int]) -> Optional[dict]:
        node = self._root
        for tok in prefix:
            node = node.get(tok)
            if node is None:
                return None
        return node

    def allowed_next(self, prefix: Sequence[int]) -> Set[int]:
        """Token ids legal after `prefix`. Empty set => prefix is dead/complete."""
        node = self._walk(prefix)
        if node is None:
            return set()
        return {t for t in node.keys() if t != -1}

    def is_complete(self, prefix: Sequence[int]) -> bool:
        node = self._walk(prefix)
        return node is not None and -1 in node

    def accepts(self, seq: Sequence[int]) -> bool:
        return self.is_complete(seq)


@dataclass
class ActionConstraint:
    """Per-response decoding constraint for one environment step.

    The engine consults this object inside the unmasking loop:

      c = ActionConstraint(trie, action_start)        # span located by engine
      mask = c.allowed_at(pos)                        # None => unconstrained
      ...engine sets logits[pos, ~mask] = -inf...
      c.commit(pos, token_id)                         # after a token commits

    Enforcement rules (DESIGN.md decision 3):
      - positions < action_start: unconstrained (thought span)
      - inside the action span only the LEFTMOST uncommitted position may
        commit (engine asks `may_commit(pos)`), so the trie state is always
        a contiguous prefix.
      - once the trie prefix is complete, the next allowed token is only
        `eos_or_newline` (terminator set), closing the action span.
    """

    trie: TokenTrie
    action_start: int
    terminators: Set[int] = field(default_factory=set)  # e.g. {newline_id, eos_id}
    _prefix: List[int] = field(default_factory=list)
    _closed: bool = False

    def _cursor(self) -> int:
        """Leftmost uncommitted position inside the action span."""
        return self.action_start + len(self._prefix)

    def may_commit(self, pos: int) -> bool:
        if pos < self.action_start or self._closed:
            return pos < self.action_start
        return pos == self._cursor()

    def allowed_at(self, pos: int) -> Optional[Set[int]]:
        """Allowed token ids at `pos`; None = position unconstrained."""
        if pos < self.action_start or self._closed:
            return None
        if pos != self._cursor():
            return set()  # not the cursor: nothing may commit here yet
        allowed = self.trie.allowed_next(self._prefix)
        if self.trie.is_complete(self._prefix):
            allowed = allowed | self.terminators
        return allowed

    def commit(self, pos: int, token_id: int) -> None:
        if pos < self.action_start or self._closed:
            return
        assert pos == self._cursor(), (
            f"out-of-order commit at {pos}, cursor={self._cursor()}"
        )
        if token_id in self.terminators and self.trie.is_complete(self._prefix):
            self._closed = True
            return
        allowed = self.trie.allowed_next(self._prefix)
        assert token_id in allowed, (
            f"illegal token {token_id} at {pos}; prefix={self._prefix}"
        )
        self._prefix.append(token_id)

    @property
    def action_tokens(self) -> List[int]:
        return list(self._prefix)

    @property
    def done(self) -> bool:
        return self._closed


class LegalityMeter:
    """Counts schema-valid actions; feeds R003, the 7/20 gate, and Table 3."""

    def __init__(self) -> None:
        self.total = 0
        self.valid = 0

    def observe(self, action_tokens: Sequence[int], trie: TokenTrie) -> bool:
        ok = trie.accepts(action_tokens)
        self.total += 1
        self.valid += int(ok)
        return ok

    def rate(self) -> float:
        return self.valid / self.total if self.total else 0.0
