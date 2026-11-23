# Three PS4s (branch `darkgarlic-cluster-3x`)

The cluster code is the same as on `darkgarlic-cluster` (docs/CLUSTER.md). This branch targets **Qwen3.5-27B, abliterated** (`huihui-ai/Huihui-Qwen3.5-27B-abliterated`, Apache-2.0). It has the same architecture as the 9B (Gated DeltaNet + gated attention, 3:1), and is three times the size.

## The shape, against the 9B

| | Qwen3.5-9B | Qwen3.5-27B |
|---|---:|---:|
| Layers (DeltaNet / full attention) | 32 (24 / 8) | 64 (48 / 16) |
| Hidden / MLP | 4096 / 12288 | 5120 / 17408 |
| Attention heads (Q / KV), head dim | 16 / 4, 256 | **24 / 4**, 256 |
| DeltaNet heads (QK / V), dims | 16 / 32, 128 | 16 / **48**, 128 |
| Vocabulary, MTP | 248320, 1 layer | 248320, 1 layer |

The figures come from the official model card and `config.json`. The bold entries are what the current kernels or converter do not handle yet.

## Profile: `cluster/profiles/qwen3.5-27b-3x.json` (quant `mlp3`)

```
node 0 (head):   layers  0..22  weights 3971 MiB  cache 158  free 319 MiB  +embedding rows (disk)
node 1 (worker): layers 23..44  weights 3791 MiB  cache 172  free 485 MiB
node 2 (worker): layers 45..63  weights 3957 MiB  cache 145  free 346 MiB  +lm_head
11719 MiB of weights; ~8 tokens/s for one chat on base PS4s (~10 with PS4 Pros), before speculative decoding
```

`mlp3` stores gate, up **and down** in 3 bits; attention, DeltaNet and lm_head stay in 4. The default layout (only gate/up in 3 bits) also fits, but it leaves just 75 MiB on node 0. That is too little for the system dialogs, the API thread and the Python heap, so `mlp3` is the recommended setup. On four consoles the default layout leaves about 1.2 GiB free on each (`--nodes 4`).

With PS4 Pros (`--console pro`) the same split has 450–610 MiB free per console, and memory bandwidth goes from 176 to 218 GB/s.

## Work list

1. **M1 on two consoles first:** the 9B two-way test (`darkgarlic-cluster-2x`, CLUSTER.md). Everything in the cluster code is the same for 3 nodes. The only new part is the middle worker that returns rows instead of a token, and the 2-node test does not exercise it. To cover it, run the 9B on **three** consoles: `python pc/cluster_plan.py --model qwen3.5-9b --nodes 3 --out cluster/profiles/qwen3.5-9b-3x-test.json`.
2. **Converter for the 27B.**
   - `pc/local_qwen35_9b.py` already reads its shapes from the checkpoint's `config.json`. Give it `--name` (today it is fixed to `qwen3.5-9b-mix`) and drop the 9B-only snapshot paths.
   - `pc/convert_9b_q4s.py` is fixed to the 9B: `NAME`, `HF_REPO`, the index paths, and `--q3=0-31`. It needs `--name`, `--hf-repo`, `--q3=0-63`, and a `--q3-down` switch for `mlp3`. The q3s writer already exists for gate/up; down uses the same format.
   - Check the padding. The DeltaNet input projection is 2·2048 + 2·6144 + 2·48 = 16 480 rows, padded to 16 512 (a multiple of 128). `SWAR_SPLIT_ROWS 12416` in `engine.c` means matrices taller than that skip split-K. That is fine (slower, not wrong), but time it.
   - Keep the MTP head conversion (`--mtp`) for later. It cannot run in a cluster yet (ROADMAP).
3. **Attention with 6 query heads per KV head.** Raise `GQ_MAX` to 6 in `shaders/attn.h` and the TurboQuant `attn_tq.h`, and relax the `n_heads / n_kv > 4` check in `engine_init`. LDS for `attn_part` grows to about 34 KiB per workgroup (`acc[4][6][256]` floats = 24 KiB). Compare it with a two-pass variant (3 heads per pass) using the `speed_case` start-up timing. Check the result against the PC reference on a 4-layer slice of the real 27B: `pc/cluster_plan.py` can size it, and `shard_model.py` can cut it.
4. **Shard and deploy:**
   ```bash
   python pc/shard_model.py cluster/profiles/qwen3.5-27b-3x.json --idx models/27b/model.idx \
          --bin models/qwen3.5-27b-q4s.bin --out models/cluster-27b-3x
   python pc/cluster_deploy.py cluster/profiles/qwen3.5-27b-3x.json models/cluster-27b-3x \
          --hosts <head> <worker1> <worker2>
   ```
   Start both workers, then the head. Then run `python pc/cluster_bench.py --bench http://<head>:8080`.
5. **Speed after it works:** prompt-lookup drafts (no memory), then the MTP head on node 2, then batched API serving (ROADMAP §3).

## Expectations

- About 8 tokens/s for one chat is a planner estimate (bandwidth × the efficiency the 9B reaches, plus network hops). Real numbers will come from `--bench`.
- The prompt goes through at roughly the 9B's prefill speed divided by three, because the consoles take turns. Overlapping them (pipelined prefill) is on the roadmap.
- One whole-27B abliteration benchmark reports a lower compliance rate than the smaller abliterated Qwen3.5 models. Expect the 27B to refuse now and then where the 9B would not.
