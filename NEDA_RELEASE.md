# NeDA learner release

This fork contains the learner, rollout, credit, evaluation, and contract code
used by the current NeDA ALFWorld and WebShop pipelines. The orchestration
scripts and RAIDEN installation guide live in
[`Enoch-Yi/NeDA`](https://github.com/Enoch-Yi/NeDA).

The source inventory is frozen by:

- `manifests/neda_v2_alfworld_code.SHA256SUMS`
- `manifests/neda_webshop_matrix_code.SHA256SUMS`
- `manifests/neda_release_files.txt`

Large artifacts are deliberately excluded from Git:

- SDAR-8B-Chat weights;
- `sft_neda/ckpt/neda_sft_v1`;
- `data/sft_alfworld_expert.json`;
- AgentBoard/ALFWorld/WebShop data and the Lucene index;
- run directories, logs, generated rollouts, and checkpoints.

Use the RAIDEN guide in the NeDA repository to acquire and verify these assets.
Do not run the two `SHA256SUMS` files from this repository root: they are
workspace manifests whose paths are rooted at `~/yinuo_dLLM` and also cover
the external qsub and AgentBoard files installed by the NeDA repository.
