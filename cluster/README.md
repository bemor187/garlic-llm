# Cluster profiles

Each JSON file comes from `pc/cluster_plan.py --out`. It records which console runs which layers, and the planner's memory and speed numbers for that split.

Feed one to `pc/shard_model.py` (to cut the converted model into shards) and to `pc/cluster_deploy.py` (to copy the shards to the consoles). The `host` fields are placeholders: pass the real addresses with `--hosts`.

| Profile | Consoles | Model | Status |
|---|---|---|---|
| `qwen3.5-9b-2x-test.json` | 2 | Qwen3.5-9B (abliterated), layers 0-18 / 19-31 | Test rig for milestone M1: needs no new kernels |
| `qwen3.5-27b-3x.json` | 3 | Qwen3.5-27B (abliterated), `mlp3` quant, layers 0-22 / 23-44 / 45-63 | Needs GQA 6 and the 27B converter (docs/PLAN-3X.md) |

See [docs/CLUSTER.md](../docs/CLUSTER.md).
