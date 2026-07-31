"""Versioned rollout/decision trace contract for Decision-Aligned NeDA."""

import math
import re
import unicodedata

from neda_repro import sha256_json


TRACE_CONTRACT_VERSION = "neda-trace-v2"
SAMPLING_CONTRACT_VERSION = "neda-sampling-v1"
EXACT_ACTION_SCORING_LAYOUT = "full-duplicate-ar-v1"
NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION = "neda-native-thought-replay-v2"
NATIVE_THOUGHT_SCORING_LAYOUT = "fixed-width-absolute-block-duplicate-ao-v2"
ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)
STRUCTURAL_ACTION_DELIMITER = "\nAction: "
STRUCTURAL_BOUNDARY_CONTRACT_VERSION = "sampled-id-prefix-before-marker-v4"


def make_sampling_spec(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    constraint="none",
    logprob_dtype="float32",
):
    result = {
        "contract_version": SAMPLING_CONTRACT_VERSION,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "constraint": str(constraint),
        "logprob_space": "post_transform",
        "logprob_dtype": str(logprob_dtype),
    }
    validate_sampling_spec(result)
    return result


def validate_sampling_spec(spec, exact_replay=False):
    if not isinstance(spec, dict):
        raise ValueError("sampling metadata must be a dictionary")
    if spec.get("contract_version") != SAMPLING_CONTRACT_VERSION:
        raise ValueError("unsupported sampling contract: {}".format(spec.get("contract_version")))
    temperature = float(spec.get("temperature", 0.0))
    top_k = int(spec.get("top_k", -1))
    top_p = float(spec.get("top_p", 0.0))
    if temperature <= 0 or top_k < 0 or not (0.0 < top_p <= 1.0):
        raise ValueError("invalid temperature/top-k/top-p in sampling metadata")
    if spec.get("logprob_space") != "post_transform":
        raise ValueError("behavior logprobs must describe the post-transform distribution")
    if spec.get("logprob_dtype") != "float32":
        raise ValueError("behavior logprobs must be computed in float32")
    if exact_replay and str(spec.get("constraint", "none")) != "none":
        raise ValueError("exact replay currently requires an unconstrained stored Action distribution")
    return True


def make_sample_id(game_id, rollout_id, turn_id):
    payload = [str(game_id), int(rollout_id), int(turn_id)]
    return "turn-{}".format(sha256_json(payload)[:20])


def trim_generation_trace(
    response_ids,
    step_map,
    behavior_logprobs,
    mask_id,
    stop_ids,
    confidence=None,
    sampling=None,
):
    """Trim padded generation at the first EOS/mask while preserving ID alignment."""

    response_ids = [int(x) for x in response_ids]
    step_map = [int(x) for x in step_map]
    behavior_logprobs = [float(x) for x in behavior_logprobs]
    if confidence is None:
        confidence = [math.exp(x) if math.isfinite(x) else 0.0 for x in behavior_logprobs]
    confidence = [float(x) for x in confidence]
    lengths = {len(response_ids), len(step_map), len(behavior_logprobs), len(confidence)}
    if len(lengths) != 1:
        raise ValueError("token ids, step map, logprobs, and confidence must align")
    stop_set = {int(x) for x in (stop_ids or []) if x is not None}
    cut = len(response_ids)
    for index, token_id in enumerate(response_ids):
        if token_id == int(mask_id) or token_id in stop_set:
            cut = index
            break
    result = {
        "response_ids": response_ids[:cut],
        "step_map": step_map[:cut],
        "behavior_logprobs": behavior_logprobs[:cut],
        "commit_confidence": confidence[:cut],
    }
    if sampling is not None:
        validate_sampling_spec(sampling)
        result["sampling"] = dict(sampling)
    if any(step < 0 for step in result["step_map"]):
        raise ValueError("kept response contains an uncommitted token")
    if any(not math.isfinite(value) for value in result["behavior_logprobs"]):
        raise ValueError("kept response contains a non-finite behavior logprob")
    return result


def action_char_span(text):
    match = ACTION_RE.search(text)
    if not match:
        return None
    start = match.start(1)
    tail = match.group(1)
    first_line = tail.split("\n", 1)[0]
    leading = len(first_line) - len(first_line.lstrip())
    content = first_line.strip()
    if content.endswith("."):
        content = content[:-1].rstrip()
    start += leading
    return start, start + len(content)


def _span_from_offsets(offsets, start_char, end_char):
    touched = []
    for index, pair in enumerate(offsets):
        left, right = int(pair[0]), int(pair[1])
        if right > start_char and left < end_char:
            touched.append(index)
    if not touched:
        return [0, 0]
    return [touched[0], touched[-1] + 1]


def token_boundary_at_char(tokenizer, text, response_ids, char_boundary):
    """Map a character boundary to the first exact token starting after it."""

    char_boundary = max(0, min(int(char_boundary), len(text)))
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == list(response_ids):
            end = 0
            for index, pair in enumerate(encoded["offset_mapping"]):
                left, right = int(pair[0]), int(pair[1])
                if right <= char_boundary or (left < char_boundary < right):
                    end = index + 1
            return min(end, len(response_ids))
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass
    count = len(tokenizer(text[:char_boundary], add_special_tokens=False)["input_ids"])
    return max(0, min(count, len(response_ids)))


def _safe_structural_overlap(fragment):
    """Whether a boundary-overlap fragment can be discarded losslessly.

    A byte-level BPE token may contain the final punctuation of a Thought and
    the following ``Action:`` marker.  Such a token cannot be split while
    retaining sampled IDs.  We permit dropping at most a short run of Unicode
    punctuation/whitespace, but never letters, numbers, symbols, or other
    semantic Thought content.
    """

    fragment = str(fragment)
    return (
        bool(fragment.strip())
        and len(fragment) <= 8
        and all(
            character.isspace()
            or unicodedata.category(character).startswith("P")
            for character in fragment
        )
    )


def _safe_marker_prefix_overlap(fragment):
    """Whether a crossing token only contributes a partial ``Action`` marker.

    Byte-level BPE can merge a duplicated or partially emitted marker prefix
    with the complete ``Action:`` marker that follows it (for example the
    observed ``"AAction:"`` token).  Dropping that one crossing token is safe
    for the interface policy only when its retained non-whitespace content is
    a *proper prefix* of ``Action``.  Arbitrary alphanumeric Thought content
    remains fail-closed.
    """

    value = str(fragment).strip().casefold()
    return bool(value) and value != "action" and "action".startswith(value)


def exact_token_prefix_before_char(
    tokenizer,
    text,
    response_ids,
    char_boundary,
    allow_punctuation_overlap=False,
    allow_marker_prefix_overlap=False,
    return_metadata=False,
):
    """Return the exact sampled-token prefix ending before ``char_boundary``.

    Unlike :func:`token_boundary_at_char`, a token touching both sides of the
    boundary is *excluded*.  This is required when the suffix is being replaced
    by a fixed structural delimiter: keeping a BPE token such as ``" go"``
    would silently move the first Action word into the Thought decision.

    If excluding the overlapping token would drop non-whitespace content from
    the retained prefix, the split is normally not representable with sampled
    IDs and fails closed.  The structural caller may explicitly allow a short
    punctuation-only overlap (for example ``".\nAction:"`` in one BPE token).
    That entire token is excluded and the dropped punctuation is authenticated
    in the returned boundary metadata.  A structural caller may additionally
    accept a proper prefix of the immediately following ``Action`` marker
    (the observed tokenizer artifact is ``"AAction:"``).  Every other
    alphanumeric overlap still fails closed.
    """

    response_ids = [int(token) for token in response_ids]
    char_boundary = max(0, min(int(char_boundary), len(text)))

    def finish(end, source, dropped="", overlap_index=None):
        if not dropped:
            overlap_kind = "none"
        elif _safe_structural_overlap(dropped):
            overlap_kind = "punctuation"
        elif _safe_marker_prefix_overlap(dropped):
            overlap_kind = "action-marker-prefix"
        else:
            raise ValueError("unauthenticated structural boundary overlap")
        result = {
            "thought_end": int(end),
            "alignment_source": str(source),
            "dropped_boundary_overlap_text": str(dropped),
            "dropped_boundary_overlap_kind": overlap_kind,
            "dropped_boundary_overlap_token_index": (
                None if overlap_index is None else int(overlap_index)
            ),
            "dropped_boundary_overlap_token_id": (
                None if overlap_index is None else int(response_ids[overlap_index])
            ),
        }
        return result if return_metadata else int(end)

    def retained_target(dropped):
        target = text[:char_boundary]
        if dropped:
            if not target.endswith(dropped):
                raise ValueError("structural overlap is not a suffix of Thought")
            target = target[: -len(dropped)]
        return target

    def check_overlap(fragment):
        if not fragment.strip():
            return ""
        if allow_punctuation_overlap and _safe_structural_overlap(fragment):
            return fragment
        if (
            allow_marker_prefix_overlap
            and _safe_marker_prefix_overlap(fragment)
        ):
            return fragment
        raise ValueError(
            "structural boundary bisects sampled Thought content: {!r}".format(
                fragment
            )
        )

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == response_ids:
            end = 0
            dropped = ""
            overlap_index = None
            for index, pair in enumerate(encoded["offset_mapping"]):
                left, right = int(pair[0]), int(pair[1])
                if right <= char_boundary:
                    end = index + 1
                    continue
                if left < char_boundary < right:
                    dropped = check_overlap(text[left:char_boundary])
                    overlap_index = index if dropped else None
                break
            decoded = tokenizer.decode(response_ids[:end], skip_special_tokens=True)
            target = retained_target(dropped)
            if decoded.rstrip() != target.rstrip():
                raise ValueError(
                    "exact token prefix does not round-trip at structural boundary"
                )
            return finish(end, "offset-mapping", dropped, overlap_index)
    except (KeyError, TypeError, NotImplementedError):
        pass

    # SDAR uses the slow byte-level Qwen tokenizer.  It does not expose offset
    # mappings, and decoding an intermediate sampled prefix can be lossy when a
    # UTF-8 character is split across BPE tokens.  Recover exact token byte
    # intervals from the tokenizer's own byte decoder before considering the
    # more general prefix-round-trip fallback.
    try:
        tokens = tokenizer.convert_ids_to_tokens(response_ids)
        byte_decoder = tokenizer.byte_decoder
        token_bytes = [
            bytes(byte_decoder[character] for character in token)
            for token in tokens
        ]
        full_bytes = b"".join(token_bytes)
        errors = getattr(tokenizer, "errors", "replace")
        if full_bytes.decode("utf-8", errors=errors) == text:
            boundary_bytes = len(text[:char_boundary].encode("utf-8"))
            cursor = 0
            end = 0
            dropped = ""
            overlap_index = None
            for index, piece in enumerate(token_bytes):
                right = cursor + len(piece)
                if right <= boundary_bytes:
                    end = index + 1
                    cursor = right
                    continue
                if cursor < boundary_bytes < right:
                    retained_fragment = piece[: boundary_bytes - cursor].decode(
                        "utf-8", errors=errors
                    )
                    dropped = check_overlap(retained_fragment)
                    overlap_index = index if dropped else None
                break
            decoded = full_bytes[: sum(len(piece) for piece in token_bytes[:end])].decode(
                "utf-8", errors=errors
            )
            target = retained_target(dropped)
            if decoded.rstrip() != target.rstrip():
                raise ValueError(
                    "byte-level sampled prefix does not round-trip at structural boundary"
                )
            return finish(end, "byte-level-token-interval", dropped, overlap_index)
    except (AttributeError, KeyError, TypeError, UnicodeError):
        pass

    # Re-tokenizing a text prefix is not an exact fallback: BPE merges depend
    # on right context, so encode(text[:b]) need not equal the sampled-ID
    # prefix. Search prefixes of the *sampled* IDs and round-trip those.
    target = text[:char_boundary].rstrip()
    exact_matches = []
    whitespace_matches = []
    for end in range(len(response_ids) + 1):
        decoded = tokenizer.decode(response_ids[:end], skip_special_tokens=True)
        if decoded == target:
            exact_matches.append(end)
        elif decoded.rstrip() == target and not decoded[len(target):].strip():
            whitespace_matches.append(end)
    if exact_matches:
        return finish(max(exact_matches), "sampled-prefix-roundtrip")
    if whitespace_matches:
        # Exclude as much structural trailing whitespace as the sampled token
        # boundaries permit before appending the fixed delimiter.
        return finish(min(whitespace_matches), "sampled-prefix-whitespace-roundtrip")
    raise ValueError(
        "tokenizer has no offsets and no sampled-ID prefix round-trips at the structural boundary"
    )


def build_structural_action_prefix(tokenizer, prompt_ids, thought_text, thought_ids):
    """Split sampled Thought from Action and rebuild a fixed ``Action:`` prefix.

    The first pass may commit the delimiter and part of the Action in one
    diffusion block.  None of those tokens belong to the Thought decision.
    We retain only exact sampled IDs before the marker, then append a canonical
    non-learned delimiter.  The second pass consequently generates the Action
    from its first content token.
    """

    match = ACTION_RE.search(thought_text)
    marker_generated = match is not None
    removed_trailing_whitespace_tokens = 0
    if marker_generated:
        # First cut at the marker itself.  Then remove only complete sampled
        # whitespace tokens.  Calling rstrip() before token alignment can put
        # the boundary inside a token that also contains real Thought content,
        # making exact replay impossible for otherwise valid generations.
        alignment = exact_token_prefix_before_char(
            tokenizer,
            thought_text,
            thought_ids,
            match.start(),
            allow_punctuation_overlap=True,
            allow_marker_prefix_overlap=True,
            return_metadata=True,
        )
        thought_end = int(alignment["thought_end"])
        while thought_end > 0:
            tail = tokenizer.decode(
                [int(thought_ids[thought_end - 1])], skip_special_tokens=True
            )
            if not tail or tail.strip():
                break
            thought_end -= 1
            removed_trailing_whitespace_tokens += 1
    else:
        thought_end = len(thought_ids)
        alignment = {
            "thought_end": int(thought_end),
            "alignment_source": "no-generated-marker",
            "dropped_boundary_overlap_text": "",
            "dropped_boundary_overlap_kind": "none",
            "dropped_boundary_overlap_token_index": None,
            "dropped_boundary_overlap_token_id": None,
        }

    delimiter_ids = list(
        tokenizer(STRUCTURAL_ACTION_DELIMITER, add_special_tokens=False)["input_ids"]
    )
    if not delimiter_ids:
        raise ValueError("structural Action delimiter tokenized to an empty sequence")
    kept_thought_ids = [int(token) for token in thought_ids[:thought_end]]
    action_prefix_ids = (
        [int(token) for token in prompt_ids] + kept_thought_ids + delimiter_ids
    )
    return {
        "thought_end": int(thought_end),
        "delimiter_ids": [int(token) for token in delimiter_ids],
        "action_prefix_ids": action_prefix_ids,
        "marker_generated": bool(marker_generated),
        "boundary_contract": STRUCTURAL_BOUNDARY_CONTRACT_VERSION,
        "removed_trailing_whitespace_tokens": int(
            removed_trailing_whitespace_tokens
        ),
        "alignment_source": alignment["alignment_source"],
        "dropped_boundary_overlap_text": alignment[
            "dropped_boundary_overlap_text"
        ],
        "dropped_boundary_overlap_kind": alignment[
            "dropped_boundary_overlap_kind"
        ],
        "dropped_boundary_overlap_token_index": alignment[
            "dropped_boundary_overlap_token_index"
        ],
        "dropped_boundary_overlap_token_id": alignment[
            "dropped_boundary_overlap_token_id"
        ],
    }


def token_spans(tokenizer, text, response_ids):
    """Return half-open Thought/Action token spans for the exact response IDs."""

    span = action_char_span(text)
    if span is None:
        return {"thought_span": [0, len(response_ids)], "action_span": [0, 0]}
    action_start, action_end = span
    offsets = None
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        candidate_ids = list(encoded["input_ids"])
        if candidate_ids == list(response_ids):
            offsets = encoded["offset_mapping"]
    except (KeyError, TypeError, ValueError, NotImplementedError):
        offsets = None
    if offsets is not None:
        action_tokens = _span_from_offsets(offsets, action_start, action_end)
        thought_tokens = _span_from_offsets(offsets, 0, max(0, action_start))
        return {"thought_span": thought_tokens, "action_span": action_tokens}

    # Slow-tokenizer fallback. Prefix token counts can differ at a BPE boundary,
    # so clamp and let the round-trip validator catch a truly ambiguous split.
    prefix_n = len(tokenizer(text[:action_start], add_special_tokens=False)["input_ids"])
    action_n = len(tokenizer(text[:action_end], add_special_tokens=False)["input_ids"])
    prefix_n = max(0, min(prefix_n, len(response_ids)))
    action_n = max(prefix_n, min(action_n, len(response_ids)))
    return {"thought_span": [0, prefix_n], "action_span": [prefix_n, action_n]}


def make_decision_trace(prefix_ids, trace, text, tokenizer, kind="response"):
    response_ids = list(trace["response_ids"])
    if kind == "action":
        spans = {"thought_span": [0, 0], "action_span": [0, len(response_ids)]}
    elif kind == "thought":
        spans = {"thought_span": [0, len(response_ids)], "action_span": [0, 0]}
    else:
        spans = token_spans(tokenizer, text, response_ids)
    result = {
        "kind": str(kind),
        "prefix_ids": [int(x) for x in prefix_ids],
        "response_ids": response_ids,
        "step_map": list(trace["step_map"]),
        "behavior_logprobs": list(trace["behavior_logprobs"]),
        "commit_confidence": list(trace["commit_confidence"]),
        "thought_span": spans["thought_span"],
        "action_span": spans["action_span"],
    }
    if "sampling" in trace:
        validate_sampling_spec(trace["sampling"])
        result["sampling"] = dict(trace["sampling"])
    if "replay_width" in trace:
        result["replay_width"] = int(trace["replay_width"])
    if "scoring_layout" in trace:
        result["scoring_layout"] = str(trace["scoring_layout"])
    result["trace_sha256"] = sha256_json(
        {
            "prefix_ids": result["prefix_ids"],
            "response_ids": result["response_ids"],
            "step_map": result["step_map"],
            "behavior_logprobs": result["behavior_logprobs"],
            "sampling": result.get("sampling"),
            "replay_width": result.get("replay_width"),
            "scoring_layout": result.get("scoring_layout"),
        }
    )
    return result


def validate_decision_trace(trace, require_logprobs=True, require_sampling=False, exact_replay=False):
    required = ("prefix_ids", "response_ids", "step_map", "behavior_logprobs")
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError("decision trace missing {}".format(missing))
    n_tokens = len(trace["response_ids"])
    if len(trace["step_map"]) != n_tokens:
        raise ValueError("step_map length does not equal response_ids length")
    if len(trace["behavior_logprobs"]) != n_tokens:
        raise ValueError("behavior_logprobs length does not equal response_ids length")
    if require_logprobs and any(
        not math.isfinite(float(value)) for value in trace["behavior_logprobs"]
    ):
        raise ValueError("behavior logprobs must be finite")
    if require_sampling or "sampling" in trace:
        if "sampling" not in trace:
            raise ValueError("exact decision trace is missing sampling metadata")
        validate_sampling_spec(trace["sampling"], exact_replay=exact_replay)
    scoring_layout = trace.get("scoring_layout")
    replay_width = trace.get("replay_width")
    if scoring_layout is not None or replay_width is not None:
        if scoring_layout != EXACT_ACTION_SCORING_LAYOUT:
            raise ValueError("unsupported exact Action scoring layout: {}".format(scoring_layout))
        if replay_width is None or int(replay_width) < n_tokens:
            raise ValueError("replay_width must cover every recorded response token")
    for name in ("thought_span", "action_span"):
        if name not in trace:
            continue
        start, end = [int(x) for x in trace[name]]
        if not (0 <= start <= end <= n_tokens):
            raise ValueError("invalid {}: {}".format(name, trace[name]))
    native = trace.get("native_replay")
    if native is not None:
        if trace.get("kind") != "thought":
            raise ValueError("native Thought replay attached to a non-Thought trace")
        if not isinstance(native, dict) or native.get(
            "contract_version"
        ) != NATIVE_THOUGHT_REPLAY_CONTRACT_VERSION:
            raise ValueError("unsupported native Thought replay contract")
        if native.get("scoring_layout") != NATIVE_THOUGHT_SCORING_LAYOUT:
            raise ValueError("unsupported native Thought replay scoring layout")
        block_size = int(native.get("block_size", 0))
        arrays = {
            key: list(native.get(key, []))
            for key in (
                "response_ids", "step_map", "behavior_logprobs",
                "commit_confidence", "optimization_mask",
            )
        }
        lengths = {len(values) for values in arrays.values()}
        if block_size <= 0 or lengths == {0} or len(lengths) != 1:
            raise ValueError("native Thought replay arrays/block size are invalid")
        width = len(arrays["response_ids"])
        replay_width = int(native.get("replay_width", 0))
        if replay_width < width:
            raise ValueError("native Thought replay width does not cover committed tokens")
        if (len(trace["prefix_ids"]) + replay_width) % block_size != 0:
            raise ValueError("native Thought fixed replay width is not block aligned")
        if (len(trace["prefix_ids"]) + width) % block_size != 0:
            raise ValueError("native Thought replay does not end on an absolute block boundary")
        if any(int(value) < 0 for value in arrays["step_map"]):
            raise ValueError("native Thought replay contains an uncommitted token")
        if any(not math.isfinite(float(value)) for value in arrays["behavior_logprobs"]):
            raise ValueError("native Thought replay contains non-finite behavior scores")
        mask = arrays["optimization_mask"]
        if any(type(value) is not bool for value in mask):
            raise ValueError("native Thought optimization mask must be boolean")
        expected = [index < n_tokens for index in range(width)]
        if mask != expected:
            raise ValueError("native Thought optimization mask/interface prefix drift")
        selected = [index for index, enabled in enumerate(mask) if enabled]
        for native_key, trace_key in (
            ("response_ids", "response_ids"),
            ("step_map", "step_map"),
            ("behavior_logprobs", "behavior_logprobs"),
            ("commit_confidence", "commit_confidence"),
        ):
            if [arrays[native_key][index] for index in selected] != list(
                trace.get(trace_key, [])
            ):
                raise ValueError("native Thought replay/interface {} drift".format(trace_key))
        stored_replay_sha = str(native.get("replay_sha256", ""))
        replay_body = dict(native)
        replay_body.pop("replay_sha256", None)
        if stored_replay_sha != sha256_json(replay_body):
            raise ValueError("native Thought replay hash drift")
    return True


def validate_rollout_record(record, require_exact_trace=True):
    required = (
        "contract_version",
        "game_id",
        "group_id",
        "episode_id",
        "turn_id",
        "sample_id",
        "raw_action",
        "executed_action",
        "action_transform",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("rollout record missing {}".format(missing))
    if record["contract_version"] != TRACE_CONTRACT_VERSION:
        raise ValueError("unsupported trace contract: {}".format(record["contract_version"]))
    identity_sha = str(record.get("model_identity_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", identity_sha):
        raise ValueError("rollout record requires a valid model_identity_sha256")
    if require_exact_trace:
        traces = record.get("decision_traces")
        if not isinstance(traces, dict) or not traces:
            raise ValueError("exact record requires decision_traces")
        for trace in traces.values():
            validate_decision_trace(trace, require_sampling=True)
        boundary = record.get("decision_boundary")
        if boundary is not None:
            if not isinstance(boundary, dict) or boundary.get(
                "contract_version"
            ) not in (
                "sampled-id-prefix-before-marker-v2",
                "sampled-id-prefix-before-marker-v3",
                STRUCTURAL_BOUNDARY_CONTRACT_VERSION,
            ):
                raise ValueError("unsupported exact decision-boundary contract")
            thought = traces.get("thought")
            thought_end = int(boundary.get("thought_end", -1))
            full_count = int(boundary.get("thought_full_token_count", -1))
            removed = int(boundary.get("removed_trailing_whitespace_tokens", -1))
            overlap = str(boundary.get("dropped_boundary_overlap_text", ""))
            overlap_kind = str(
                boundary.get(
                    "dropped_boundary_overlap_kind",
                    "punctuation" if overlap else "none",
                )
            )
            overlap_tokens = int(bool(overlap))
            action_only = boundary.get("action_only_decision")
            if boundary.get("contract_version") == STRUCTURAL_BOUNDARY_CONTRACT_VERSION:
                overlap_is_valid = (
                    (not overlap and overlap_kind == "none")
                    or (
                        overlap_kind == "punctuation"
                        and _safe_structural_overlap(overlap)
                    )
                    or (
                        overlap_kind == "action-marker-prefix"
                        and _safe_marker_prefix_overlap(overlap)
                    )
                )
            else:
                overlap_is_valid = (
                    (not overlap and overlap_kind == "none")
                    or (
                        overlap_kind == "punctuation"
                        and _safe_structural_overlap(overlap)
                    )
                )
            if (
                not isinstance(thought, dict)
                or thought_end != len(thought.get("response_ids", []))
                or not (0 <= thought_end <= full_count)
                or not (0 <= removed <= full_count - thought_end)
                or removed + overlap_tokens > full_count - thought_end
                or not overlap_is_valid
                or (
                    action_only is not None
                    and bool(action_only) != (thought_end == 0)
                )
            ):
                raise ValueError("exact decision-boundary accounting drift")
            native = thought.get("native_replay") if isinstance(thought, dict) else None
            if native is not None and full_count != len(native.get("response_ids", [])):
                raise ValueError("decision boundary/native Thought replay count drift")
    return True


def history_action(executed_action):
    """The next state must describe the action that produced its observation."""

    return "Thought/Action: {}".format(str(executed_action))
