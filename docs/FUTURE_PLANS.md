# DarkGarlic future plans

## Stage 1: shipping now (branch `darkgarlic`)

- Abliterated Qwen3.5-9B (`huihui-ai/Huihui-Qwen3.5-9B-abliterated`) on one PS4's GPU. Measured 31–39 tokens/s in chat, up to 43 on code (upstream measurements, base and Slim PS4).
- OpenAI-compatible API on the LAN: `/v1/chat/completions` with streaming, `/v1/models`, `/health`, optional API key.
- Voice in and out (Parakeet, Chatterbox), Python mode, Think mode, 16K context.
- An all-in-one package on Hugging Face: `bemor187/darkgarlic-llm-ps4`.

## Stage 2: once we get funding

Stage 2 makes DarkGarlic bigger (models that need 2–3 consoles), faster, and more useful as a home AI server.

The groundwork is already written on three branches. It has not run on a PS4 yet:

| Branch | What is there |
|---|---|
| [`darkgarlic-cluster`](https://github.com/bemor187/garlic-llm/tree/darkgarlic-cluster) | The model override folder, the cluster engine and protocol, the planner and tools. Design and bring-up plan: `docs/CLUSTER.md`. Itemised roadmap: `docs/ROADMAP.md`. |
| [`darkgarlic-cluster-2x`](https://github.com/bemor187/garlic-llm/tree/darkgarlic-cluster-2x) | The two-console setup: a 9B test rig and Qwen3-14B. `docs/PLAN-2X.md`. |
| [`darkgarlic-cluster-3x`](https://github.com/bemor187/garlic-llm/tree/darkgarlic-cluster-3x) | Abliterated Qwen3.5-27B on three consoles. `docs/PLAN-3X.md`. |

### What funding pays for

| Need | Why |
|---|---|
| 2 more PS4s on GoldHEN, ideally PS4 Pros | The cluster needs 2–3 consoles to test. Pros have ~24 % more memory bandwidth and ~128 MiB more GPU memory for the model, and users report up to 56 tokens/s on the 9B. |
| A gigabit switch and cables | Rows travel between consoles for every token. The PS4's gigabit Ethernet is enough; Wi-Fi is not reliable for it. |
| A PC with 64 GB+ RAM, or cloud conversion time | Converting the 27B (about 55 GB in BF16) and running the PC reference checks against the console output. |
| Development time | GPU kernel work (§2.2) and the testing milestones. |

### 2.1 Cluster bring-up: bigger models over several PS4s

The cluster splits a model by layers across consoles. It adds room for a bigger model, not speed: each token still passes through every console.

| Milestone | Goal | Done when |
|---|---|---|
| M1 | Abliterated 9B split over 2 consoles | `pc/cluster_bench.py --compare` gives the same replies as one console |
| M1b | The same 9B over 3 consoles | The middle-worker path is proven |
| M2 | Kernels for the bigger shapes (§2.2) | They match the PC reference on a few layers of the real model |
| M3 | Qwen3.5-27B on 3 consoles (`mlp3` quant) | Long conversations work; est. ~8 tokens/s (~10 on Pros) |
| M3b | Qwen3-14B on 2 consoles, or a "better 9B" with all-4-bit weights, voice + MTP together and 32K context | See PLAN-2X.md for which is worth it |
| M4 | Cluster speed (§2.3) | Measured with `--bench` |

Planner numbers (`pc/cluster_plan.py`), base PS4s:

| Setup | Model | Weights | Free per console | Est. tokens/s |
|---|---|---:|---:|---:|
| 2× | Qwen3.5-9B (test) | 3880 MiB | ~2.4 GiB | ~25 |
| 2× | Qwen3-14B, 8K context | 6656 MiB | ~460–500 MiB | ~14 |
| 3× | Qwen3.5-27B, `mlp3` | 11719 MiB | ~320–485 MiB | ~8 |
| 4× | Qwen3.5-27B, default quant | 12399 MiB | ~1.2 GiB | ~8 |

### 2.2 Engine and kernel work

- **More query heads per KV head.** `GQ_MAX` 4 → 6 in `shaders/attn.h` and `attn_tq.h`. The 27B has 24/4 heads and the 14B has 40/8.
- **One converter for any Qwen3.5 model.** Generalise `convert_9b_q4s.py` and `local_qwen35_9b.py` with model name, HF repo, layer count and `--q3-down`, plus a Qwen3 mode for the 14B.
- **Qwen3 path upgrades** (for the 14B): q4s/q3s weights, the batched prompt, and an int8 or TurboQuant KV cache.
- **Mixture-of-experts support** (Qwen3.5-35B-A3B and up). This is the best long-term fit for a cluster: only ~3B parameters are read per token, so a much stronger model could run near today's 9B speed over 4–5 consoles.
- **Model kernels from the override folder.** Allow a model-specific `kernels.bin` in `/data/darkgarlic`.

### 2.3 Speed

- **Native PS4 Pro mode.** Tune it and make it the default on Pros.
- **Prompt-lookup decoding.** It drafts tokens from text already in the context, with no extra memory. It helps most on code, summaries and documents, and it works in a cluster today.
- **The MTP head on the last cluster node.** That brings speculative decoding (about +50 % on the 9B) to bigger models.
- **Batched API serving.** 2–4 chats share each pass, at nearly the cost of one. This is also how every console in a cluster stays busy.
- **Pipelined prompt in the cluster.** Consoles work on different batches at the same time.
- **Coherent (Onion) transfer buffers.** These make the CPU's reads of relayed rows cheap.

### 2.4 Features

- **Tool / function calling in the API**, so agents and coding tools can use the console.
- **Anthropic-compatible `/v1/messages`** and an embeddings endpoint.
- **Documents from a USB drive (RAG).** Ask questions about your own files, offline.
- **Save and restore conversations**, with long system prompts prefilled once.
- **Repetition control** (presence and frequency penalty). The abliterated weights loop more than the original.
- **Temperature and fan readout** for long API sessions.
- **A web chat page** served by the console, for phones and PCs on the same network.

### 2.5 Cluster robustness

- Reconnect a worker that drops, then replay the conversation.
- A model fingerprint in the cluster handshake, to catch mixed conversions.
- Each worker's status and round-trip time in the head's chat header.

## Ground rules that carry into Stage 2

- Credit stays as in `PROMOTE.md`: Leonardo Alegría for the engine and kernels, huihui-ai for the abliterated weights, Alibaba Qwen for the base models.
- Speed figures in public posts are only ever measured ones, never planner estimates.
- The jailbreak (GoldHEN) is the user's own choice and risk. DarkGarlic contains no exploit.
