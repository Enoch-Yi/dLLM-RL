# NeDA learner release

This fork contains the learner, rollout, credit, evaluation, and contract code
used by the current NeDA ALFWorld and WebShop pipelines. The 2026-08-03
snapshot also includes the collective-safe DCoLT replay path, authenticated
checkpoint-weight identities, and the frozen training contracts. The current
paper, accepted result tables, account-portable orchestration, and RAIDEN
installation guide live in the private
[`Enoch-Yi/NeDA`](https://github.com/Enoch-Yi/NeDA).

Historical formal-audit entry points are intentionally not advertised as part
of this portable learner snapshot: several bind to earlier frozen bundle
hashes and scheduler recovery records. Current public claims must be taken from
the authenticated result tables in the private repository, not recomputed from
an arbitrary mixture of historical manifests.

The source inventory is frozen by:

- `manifests/neda_v2_alfworld_code.SHA256SUMS`
- `manifests/neda_webshop_matrix_code.SHA256SUMS`
- `manifests/neda_joint_alfworld_sft_code.SHA256SUMS`
- `manifests/neda_joint_alfworld_ablation_code.SHA256SUMS`
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
