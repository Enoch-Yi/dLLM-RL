"""Torch-only primitives shared by exact Action rollout and learner replay."""

from contextlib import contextmanager
import os

import torch

from neda_data_contract import NATIVE_THOUGHT_SCORING_LAYOUT


def make_basic_block_attention(total_length, start_pos, block_size, device=None):
    """Build the canonical 0/1 mask for SDAR's duplicated learner layout."""

    total_length = int(total_length)
    start_pos = int(start_pos)
    block_size = int(block_size)
    response_width = (total_length - start_pos) // 2
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if start_pos + 2 * response_width != total_length:
        raise ValueError("input length must equal L0 + 2*L1")
    bias = torch.zeros(
        (1, 1, total_length, total_length),
        dtype=torch.float32,
        device=device,
    )
    duplicated_rows = torch.arange(
        start_pos + response_width, total_length, device=device
    )
    token_rows = torch.arange(
        start_pos, start_pos + response_width, device=device
    )
    for block_index in range((response_width + block_size - 1) // block_size):
        left_end = start_pos + min(block_index * block_size, response_width)
        right_start = start_pos + response_width + (left_end - start_pos)
        begin = block_index * block_size
        end = min((block_index + 1) * block_size, response_width)
        rows = duplicated_rows[begin:end]
        bias[:, :, rows.unsqueeze(-1), 0:left_end] = 1
        bias[
            :, :, rows.unsqueeze(-1), right_start : right_start + block_size
        ] = 1
        rows = token_rows[begin:end]
        left_end = start_pos + min((block_index + 1) * block_size, response_width)
        bias[:, :, rows.unsqueeze(-1), 0:left_end] = 1
    if start_pos > 0:
        for block_index in range((start_pos + block_size - 1) // block_size):
            row_end = max(start_pos - block_index * block_size, 0)
            row_start = max(start_pos - (block_index + 1) * block_size, 0)
            if row_end > row_start:
                rows = torch.arange(row_start, row_end, device=device)
                bias[:, :, rows.unsqueeze(-1), 0:row_end] = 1
    return bias


def make_absolute_block_duplicate_attention(
    total_length, start_pos, block_size, device=None
):
    """Replay absolute-position AO blocks in the duplicated learner layout.

    Rollout generation anchors diffusion blocks at sequence position zero.  A
    Thought can therefore begin part-way through a block when the prompt length
    is not divisible by ``block_size``.  The legacy learner silently restarted
    the partition at the Thought boundary.  This mask retains finalized prior
    blocks in the first copy and places the exact current block in the second.
    """

    total_length = int(total_length)
    start_pos = int(start_pos)
    block_size = int(block_size)
    response_width = (total_length - start_pos) // 2
    if (
        block_size <= 0
        or response_width <= 0
        or start_pos + 2 * response_width != total_length
    ):
        raise ValueError("invalid absolute-block duplicated layout")
    original_length = start_pos + response_width
    bias = torch.zeros(
        (1, 1, total_length, total_length),
        dtype=torch.float32,
        device=device,
    )
    # First copy: ordinary absolute block-causal sequence.  These finalized
    # states become the cached past seen by subsequent response blocks.
    for query in range(original_length):
        block_end = min(
            ((query // block_size) + 1) * block_size, original_length
        )
        bias[:, :, query, :block_end] = 1
    # Second copy: prior absolute blocks come from the first copy; the current
    # block (including a possible prompt tail) is reconstructed exactly.
    for response_position in range(response_width):
        absolute_position = start_pos + response_position
        block_start = (absolute_position // block_size) * block_size
        block_end = min(block_start + block_size, original_length)
        query = original_length + response_position
        prefix_end = min(start_pos, block_end) if block_start < start_pos else block_start
        bias[:, :, query, :prefix_end] = 1
        response_start = max(start_pos, block_start) - start_pos
        response_end = block_end - start_pos
        bias[
            :, :, query,
            original_length + response_start : original_length + response_end,
        ] = 1
    return bias


@contextmanager
def exact_replay_numerics():
    """Use deterministic operators for behavior scoring and learner replay.

    Math SDPA removes query-shape backend drift.  The scoped RMSNorm flag also
    replaces independently autotuned Triton RMSNorm with the reference PyTorch
    operator; a two-process GPU probe showed the fused operator changed an
    uncertain token's log-probability by .158876 even on an identical sampled
    path, whereas the reference operator was bit-identical across processes.
    """

    # ``@use_kernel_forward_from_hub`` can bypass the method body that checks
    # the environment flag.  This happened only after Accelerator/DeepSpeed
    # preparation and left a .163 replay drift even though the direct scorer
    # probe passed.  Patch the class method for the complete exact scope; all
    # existing module instances resolve it dynamically, including ZeRO-3
    # wrappers and gradient-checkpoint recomputation during backward.
    from models.sdar.modeling_sdar import (
        SDARRMSNorm,
        neda_reference_rmsnorm_forward,
    )

    previous_forward = SDARRMSNorm.forward
    SDARRMSNorm.forward = neda_reference_rmsnorm_forward
    if SDARRMSNorm.forward is not neda_reference_rmsnorm_forward:
        raise RuntimeError("failed to install deterministic exact RMSNorm")
    previous_rmsnorm = os.environ.get("NEDA_EXACT_PURE_RMSNORM")
    os.environ["NEDA_EXACT_PURE_RMSNORM"] = "1"
    if not torch.cuda.is_available():
        try:
            yield
        finally:
            SDARRMSNorm.forward = previous_forward
            if previous_rmsnorm is None:
                os.environ.pop("NEDA_EXACT_PURE_RMSNORM", None)
            else:
                os.environ["NEDA_EXACT_PURE_RMSNORM"] = previous_rmsnorm
        return
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        SDARRMSNorm.forward = previous_forward
        if previous_rmsnorm is None:
            os.environ.pop("NEDA_EXACT_PURE_RMSNORM", None)
        else:
            os.environ["NEDA_EXACT_PURE_RMSNORM"] = previous_rmsnorm
