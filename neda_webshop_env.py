#!/usr/bin/env python3
"""Canonical, replayable WebShop environment contract for NeDA v4.

The original AgentBoard adapter exposes ``search[]`` as an open-text action.
That is unsuitable for a raw==executed Action likelihood/replay contract.  The
common-interface benchmark therefore maps the observed task goal to a small,
deterministic set of legitimate ``search[...]`` actions and otherwise uses the
finite click actions exposed by the page.  Every method sees the same set.

This module owns no model code.  It provides deterministic action candidates,
state fingerprints, a thin live-server session wrapper, and exact prefix
replay that can be tested with a fake environment on a login node.
"""

import json
import os
import re
import sys
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

try:
    import requests
except ImportError:  # Keep pure Action/state contract tests login-node safe.
    requests = None

from neda_repro import sha256_json


ACTION_INTERFACE = "goal-query-candidates-plus-page-click-trie-v2"
BACKEND_ACTION_MAP_CONTRACT = "neda-webshop-backend-action-map-v1"
MEMORY_CONTRACT = "executed-action-observation-only-v1"
STATE_CONTRACT_VERSION = "neda-webshop-state-v1"
SEARCH_CONTRACT_VERSION = "neda-webshop-search-candidates-v1"

_GENERIC_WORDS = {
    "a",
    "an",
    "and",
    "be",
    "below",
    "buy",
    "dollar",
    "dollars",
    "find",
    "for",
    "get",
    "i",
    "in",
    "is",
    "it",
    "less",
    "like",
    "lower",
    "me",
    "of",
    "or",
    "please",
    "price",
    "purchase",
    "should",
    "than",
    "that",
    "the",
    "to",
    "under",
    "want",
    "with",
    "would",
}


def canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.replace("\x00", " ").split())


def _query_words(goal: str) -> List[str]:
    text = canonical_text(goal).lower()
    # Search syntax is carried by the outer ``search[...]`` wrapper; brackets
    # and punctuation inside a query would make the Action grammar ambiguous.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [word for word in text.split() if word]


def search_action_candidates(goal: str, max_words: int = 20) -> Dict[str, Any]:
    """Build an ordered, finite, goal-only WebShop search action set.

    No product database, target ASIN, reward, or page result is consulted.  The
    three views are: a bounded normalized instruction, content words with
    generic shopping language removed, and a compact prefix.  Duplicate views
    are removed, so very short goals may legitimately expose fewer than three
    actions.
    """

    max_words = int(max_words)
    if max_words < 4:
        raise ValueError("WebShop query max_words must be at least four")
    words = _query_words(goal)
    if not words:
        raise ValueError("WebShop goal cannot produce an identifiable search query")
    bounded = words[:max_words]
    content = [word for word in words if word not in _GENERIC_WORDS][:max_words]
    compact_width = min(max(4, max_words // 2), len(words))
    compact = words[:compact_width]
    queries: List[str] = []
    for candidate in (content, bounded, compact):
        query = " ".join(candidate).strip()
        if query and query not in queries:
            queries.append(query)
    actions = ["search[{}]".format(query) for query in queries]
    result = {
        "contract_version": SEARCH_CONTRACT_VERSION,
        "source": "task-goal-only",
        "max_words": max_words,
        "goal_sha256": sha256_json(canonical_text(goal)),
        "queries": queries,
        "actions": actions,
    }
    result["candidate_sha256"] = sha256_json(result)
    return result


def canonical_action_space(
    raw_actions: Iterable[Any],
    goal: str,
    page_type: str = "",
    page_num: int = 1,
) -> Dict[str, Any]:
    # The model sees whitespace-canonical Action coordinates, while AgentBoard
    # compares item-option labels byte-for-byte.  Preserve both views and make
    # the mapping explicit.  This prevents a valid label such as
    # ``500cm |  16.4ft`` from becoming an invalid backend click after prompt
    # normalization, without changing the policy-selected semantic Action.
    raw = [str(value or "") for value in raw_actions]
    raw = [value for value in raw if canonical_text(value)]
    canonical_raw = [canonical_text(value) for value in raw]
    if canonical_raw == ["search[]"]:
        search = search_action_candidates(goal)
        actions = list(search["actions"])
        backend_action_map = {value: value for value in actions}
        source = "goal-query-candidates"
    else:
        search = None
        actions = list(canonical_raw)
        backend_action_map: Dict[str, str] = {}
        for exact, canonical in zip(raw, canonical_raw):
            previous = backend_action_map.get(canonical)
            if previous is not None and previous != exact:
                raise ValueError(
                    "WebShop backend labels collide after canonicalization"
                )
            backend_action_map[canonical] = exact
        source = "page-click-actions"
        # AgentBoard advertises navigation buttons that its own transition code
        # rejects at page boundaries.  Removing them makes "admissible" exact.
        if str(page_type) == "search" and int(page_num) <= 1:
            actions = [value for value in actions if value != "click[< Prev]"]
        if str(page_type) == "search" and int(page_num) >= 5:
            actions = [value for value in actions if value != "click[Next >]"]
    unique: List[str] = []
    for action in actions:
        if action not in unique:
            unique.append(action)
    if not unique:
        raise ValueError("WebShop canonical Action set is empty")
    if any(re.match(r"^(?:search|click)\[[^\[\]]+\]$", value) is None for value in unique):
        raise ValueError("WebShop canonical Action set contains malformed actions")
    backend_action_map = {value: backend_action_map[value] for value in unique}
    if any(canonical_text(exact) != canonical for canonical, exact in backend_action_map.items()):
        raise ValueError("WebShop backend Action map is not normalization-equivalent")
    result = {
        "contract_version": ACTION_INTERFACE,
        "backend_action_map_contract": BACKEND_ACTION_MAP_CONTRACT,
        "source": source,
        "raw_actions": raw,
        "actions": unique,
        "backend_action_map": backend_action_map,
        "page_type": str(page_type),
        "page_num": int(page_num),
        "search_candidates": search,
    }
    result["action_space_sha256"] = sha256_json(result)
    return result


def make_state_record(
    observation: Any,
    actions: Iterable[Any],
    progress: float,
    terminal_reward: float,
    done: bool,
    goal: str,
) -> Dict[str, Any]:
    payload = {
        "observation": canonical_text(observation),
        "admissible_actions": sorted(set(canonical_text(value) for value in actions)),
        "progress": float(progress),
        "terminal_reward": float(terminal_reward),
        "done": bool(done),
        "goal_sha256": sha256_json(canonical_text(goal)),
    }
    result = {"contract_version": STATE_CONTRACT_VERSION, **payload}
    result["state_sha256"] = sha256_json(payload)
    return result


def validate_state_record(state: Mapping[str, Any]) -> None:
    if state.get("contract_version") != STATE_CONTRACT_VERSION:
        raise ValueError("unsupported WebShop state contract")
    rebuilt = make_state_record(
        state.get("observation", ""),
        state.get("admissible_actions", []),
        state.get("progress", 0.0),
        state.get("terminal_reward", 0.0),
        state.get("done", False),
        "",  # compared below using the already frozen goal hash
    )
    body = {
        "observation": canonical_text(state.get("observation", "")),
        "admissible_actions": sorted(
            set(canonical_text(value) for value in state.get("admissible_actions", []))
        ),
        "progress": float(state.get("progress", 0.0)),
        "terminal_reward": float(state.get("terminal_reward", 0.0)),
        "done": bool(state.get("done", False)),
        "goal_sha256": str(state.get("goal_sha256", "")),
    }
    # ``rebuilt`` normalizes all non-goal fields; retaining the recorded hash is
    # intentional because the raw goal is stored once at episode level.
    del rebuilt
    if len(body["goal_sha256"]) != 64 or state.get("state_sha256") != sha256_json(body):
        raise ValueError("WebShop state SHA does not match its payload")


def assert_state_match(expected: Mapping[str, Any], actual: Mapping[str, Any], context: str) -> None:
    validate_state_record(expected)
    validate_state_record(actual)
    if expected["state_sha256"] != actual["state_sha256"]:
        changed = [
            key
            for key in (
                "observation",
                "admissible_actions",
                "progress",
                "terminal_reward",
                "done",
                "goal_sha256",
            )
            if expected.get(key) != actual.get(key)
        ]
        raise ValueError(
            "WebShop exact replay mismatch at {} (fields={})".format(
                context, ",".join(changed) or "unknown"
            )
        )


class WebShopSession(object):
    """One unique fixed-task session against a local AgentBoard server."""

    def __init__(
        self,
        agentboard_root: str,
        task_id: int,
        session_namespace: str,
        web_url: str = "http://127.0.0.1:3000",
        request_timeout: int = 30,
    ):
        self.agentboard_root = os.path.realpath(agentboard_root)
        self.task_id = int(task_id)
        self.session_namespace = re.sub(r"[^A-Za-z0-9-]+", "-", str(session_namespace))
        self.web_url = str(web_url).rstrip("/")
        self.request_timeout = int(request_timeout)
        self.env = None
        self.session = None
        self.goal = ""
        self.state = None
        self.action_contract = None

    def _commands(self) -> Dict[str, Any]:
        raw = self.env.get_action_space(self.session)
        live = self.env.sessions[self.session]
        return canonical_action_space(
            raw,
            self.goal,
            page_type=str(live.get("page_type", "")),
            page_num=int(live.get("page_num", 1)),
        )

    def reset(self) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("WebShop live sessions require the requests package")
        if self.agentboard_root not in sys.path:
            sys.path.insert(0, self.agentboard_root)
        from environment.webshop_env import Webshop

        requests.head(
            self.web_url + "/failed", timeout=self.request_timeout
        ).raise_for_status()
        self.env = Webshop(web_url=self.web_url)
        self.session = "{}_fixed_{}".format(self.session_namespace, self.task_id)
        observation, reward, done, progress, grounding = self.env.step(
            self.session, "reset[]"
        )
        if grounding is not True or done:
            raise ValueError("WebShop fixed-task reset failed its grounding contract")
        self.goal = canonical_text(self.env.goal)
        self.action_contract = self._commands()
        self.state = make_state_record(
            observation,
            self.action_contract["actions"],
            progress,
            reward,
            done,
            self.goal,
        )
        return {
            "observation": canonical_text(observation),
            "reward": float(reward),
            "progress": float(progress),
            "done": bool(done),
            "grounding": bool(grounding),
            "goal": self.goal,
            "actions": list(self.action_contract["actions"]),
            "action_contract": self.action_contract,
            "state": self.state,
        }

    def step(self, action: str) -> Dict[str, Any]:
        action = canonical_text(action)
        if self.state is None:
            raise ValueError("WebShop session must be reset before step")
        if self.state["done"]:
            raise ValueError("WebShop session cannot step after terminal state")
        if action not in self.action_contract["actions"]:
            raise ValueError("WebShop Action is outside the canonical action set")
        if self.action_contract.get("backend_action_map_contract") != BACKEND_ACTION_MAP_CONTRACT:
            raise ValueError("WebShop backend Action map contract drift")
        backend_action = self.action_contract.get("backend_action_map", {}).get(action)
        if not isinstance(backend_action, str) or canonical_text(backend_action) != action:
            raise ValueError("WebShop canonical/backend Action mapping is invalid")
        observation, reward, done, progress, grounding = self.env.step(
            self.session, backend_action
        )
        if grounding is not True:
            raise ValueError("canonical WebShop Action was rejected by the environment")
        if done:
            next_contract = {
                "contract_version": ACTION_INTERFACE,
                "backend_action_map_contract": BACKEND_ACTION_MAP_CONTRACT,
                "source": "terminal",
                "raw_actions": [],
                "actions": [],
                "backend_action_map": {},
                "page_type": "end",
                "page_num": 0,
                "search_candidates": None,
            }
            next_contract["action_space_sha256"] = sha256_json(next_contract)
        else:
            next_contract = self._commands()
        state = make_state_record(
            observation,
            next_contract["actions"],
            progress,
            reward,
            done,
            self.goal,
        )
        self.action_contract = next_contract
        self.state = state
        return {
            "observation": canonical_text(observation),
            "reward": float(reward),
            "progress": float(progress),
            "done": bool(done),
            "grounding": bool(grounding),
            "canonical_action": action,
            "backend_action": backend_action,
            "goal": self.goal,
            "actions": list(next_contract["actions"]),
            "action_contract": next_contract,
            "state": state,
        }


def replay_prefix(
    session_factory: Callable[[str], Any],
    episode: Mapping[str, Any],
    anchor_turn: int,
    replay_namespace: str,
    max_history: int,
    env_reset_seed: Any = None,
) -> Dict[str, Any]:
    """Restore an anchor by replaying the recorded environment Actions."""

    turns = list(episode.get("turns", []))
    anchor_turn = int(anchor_turn)
    if not (0 <= anchor_turn < len(turns)):
        raise ValueError("WebShop replay anchor is outside the episode")
    session = session_factory(str(replay_namespace))
    current = session.reset()
    if canonical_text(current["goal"]) != canonical_text(episode.get("goal", "")):
        raise ValueError("WebShop exact replay restored a different task goal")
    assert_state_match(episode["initial_state"], current["state"], "reset")
    history = ["Observation: {}".format(current["observation"])]
    actions: List[str] = []
    state_hashes = [current["state"]["state_sha256"]]
    for turn_id in range(anchor_turn):
        turn = turns[turn_id]
        if int(turn.get("turn_id", -1)) != turn_id:
            raise ValueError("WebShop base turn IDs are not contiguous")
        assert_state_match(turn["state_before"], current["state"], "turn-{}-before".format(turn_id))
        action = str(turn["executed_action"])
        expected_backend_action = turn.get("backend_action")
        if not isinstance(expected_backend_action, str) or canonical_text(
            expected_backend_action
        ) != canonical_text(action):
            raise ValueError("WebShop recorded canonical/backend Action mapping is invalid")
        current = session.step(action)
        if current.get("backend_action") != expected_backend_action:
            raise ValueError("WebShop exact replay restored a different backend Action label")
        assert_state_match(turn["state_after"], current["state"], "turn-{}-after".format(turn_id))
        actions.append(action)
        state_hashes.append(current["state"]["state_sha256"])
        history.extend(
            [
                "Thought/Action: {}".format(action),
                "Observation: {}".format(current["observation"]),
            ]
        )
        if len(history) > int(max_history):
            history = history[:1] + history[-(int(max_history) - 1) :]
    assert_state_match(turns[anchor_turn]["state_before"], current["state"], "anchor")
    audit = {
        "contract_version": "neda-webshop-prefix-replay-audit-v1",
        "status": "PASS",
        "env_reset_seed": None if env_reset_seed is None else int(env_reset_seed),
        "n_replayed_transitions": anchor_turn,
        "executed_actions_sha256": sha256_json(actions),
        "state_path_sha256": sha256_json(state_hashes),
        "anchor_state_sha256": current["state"]["state_sha256"],
    }
    audit["audit_sha256"] = sha256_json(audit)
    return {
        "session": session,
        "current": current,
        "history": history,
        "audit": audit,
    }
