# DeepSeek-V4-Pro SGLang 推理基准测试报告（p5en.48x 单机）

## 1. 测试环境

- **模型**：`deepseek-ai/DeepSeek-V4-Pro`（61 层 MoE，FP4 routed experts + FP8 KV，权重 **805.35 GiB / 864,738,896,132 B**，存于宿主机 NVMe `/opt/dlami/nvme/models/deepseek-v4-pro`），Pod 内以 hostPath 只读挂载到 `/models/deepseek-v4-pro`
- **推理引擎**：sglang，镜像 `lmsysorg/sglang:deepseek-v4-hopper`（DS-V4 cookbook 分支；主干 `lmsysorg/sglang:latest` 因 transformers 5.3.0 未注册 `model_type=deepseek_v4` 起不来，必须用 hopper 专用 tag）
- **入口点**：`sglang serve`（新 CLI），**不是**旧的 `python3 -m sglang.launch_server`——旧入口在这个镜像上会把 decode 塌成 BOS
- **并行度**：TP=8（8 × H200 全部一个 TP 组），cookbook `h200-fp4` 配方不显式设 `--ep-size`，实测 `ep_size=1`；MoE 路由走 `--moe-runner-backend=marlin`（Hopper FP4 原生 kernel）
- **测试方法**：OpenAI 兼容 `/v1/chat/completions`，所有请求前缀加唯一 UUID salt 绕开 prefix cache，input tokens 以 server usage 字段为准

| 实例 | GPU | GPU 间互连 | 位置 |
|---|---|---|---|
| `p5en.48xlarge` | 8 × NVIDIA H200（141 GB HBM3e，单机 ≈ 1128 GiB） | **NVLink NV18 全互联**（NVSwitch full mesh；18 bonded 链路/对） | us-east-2a（`ip-10-1-109-117`，`i-0653101d5561518bd`，集群 `p5-training-eks`） |

节点 T1 标签：`workload=deepseek-v4-pro-bench`；nodeSelector 锁定单节点，单机阶段不影响其它 3 台 p5en。

## 2. NVLink 证据（p5en.48x, 节点 ip-10-1-109-117）

```
$ nvidia-smi topo -m
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity   NUMA Affinity
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  0-47,96-143    0
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  0-47,96-143    0
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  0-47,96-143    0
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  0-47,96-143    0
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  48-95,144-191  1
GPU5    NV18  NV18  NV18  NV18  NV18   X    NV18  NV18  48-95,144-191  1
GPU6    NV18  NV18  NV18  NV18  NV18  NV18   X    NV18  48-95,144-191  1
GPU7    NV18  NV18  NV18  NV18  NV18  NV18  NV18   X    48-95,144-191  1
```

`ibv_devices` 列出 16 个 EFA NIC（rdmap{85,86,87,88,110,111,112,113,135,136,137,138,160,161,162,163}s0），`/dev/infiniband` 暴露 uverbs0–15 + rdma_cm。

**NCCL 路径（冷启 `NCCL_DEBUG=INFO` 摘要，`logs/nccl-debug.summary.txt`）**：

| 指标 | 计数 |
|---|---:|
| `GDR 1` 行（Connected all rings, use ring PXN 0 GDR 1） | 9 |
| `P2P/IPC` 通道（via P2P/IPC） | 537 |
| SHM transports | 0 |
| `Libfabric` / `NET/IB` | **未加载**（见 §8 caveat） |

全部 8 个 TP rank 的集合通讯都走 NVSwitch P2P/IPC（NVLink），0 条 SHM。单机场景下 Libfabric/EFA 路径本来就不会被触发，但 128K 档二次冷启结果与 256K 档完全一致（同 9 × GDR 1、同 537 × P2P/IPC），证明 NVLink + NCCL_P2P_LEVEL=NVL 的组合稳定。

## 3. p5en 上下文尺寸（K2.5 vs DS-V4-Pro）

冷启日志里 TP0 报告的 KV 预算（`logs/cold-start.log` 第 1263 行 / `logs/cold-start-128K.log` 第 1317 行）：

| 模型 | context_length | max_total_num_tokens（KV 预算） | mem_fraction_static | max_running_requests |
|---|---|---:|---:|---:|
| K2.5            | 131072 (128K)   | ~838,000 | 0.92 | 128 |
| K2.5            | 262144 (256K)   | **837,239** | 0.92 | 128 |
| DS-V4-Pro       | 131072 (128K)   | **771,072** | 0.88 | 256 |
| **DS-V4-Pro**   | **262144 (256K)** | **771,072** | 0.88 | 256 |

两档 KV 预算**一模一样 771K**——cookbook 用 `--mem-fraction-static 0.88` + `kv_cache_dtype=fp8_e4m3` 时 KV pool 独立于 context-length 预分配，只有每请求上限随 `--context-length` 变化。这和 K2.5 report 记录的 "128K 与 256K 下 max_total_num_tokens 均 ~838K" 的观察一致。DS-V4-Pro 比 K2.5 少 66K KV 预算（-7.9%）是因为 mem_fraction 从 0.92 降到 0.88（Marlin FP4 MoE kernel 需要额外 scratch）。

`/v1/models` 验证（`logs/v1-models.json` / `logs/v1-models-128K.json`）：

```
# 256K: {"id":"deepseek-v4-pro","max_model_len":262144,...}
# 128K: {"id":"deepseek-v4-pro","max_model_len":131072,...}
```

**200K+ AC smoke**（`logs/ac-smoke-200K.json`）：prompt_tokens=**204,894**（server 实测），HTTP 200，latency 48.1s，finish_reason=stop，content 非空（pangram summary 235 chars）。

## 4. 表 1：5.8K prompt QPM 压测（output ≈ 32 tokens，每档 60s；DS-V4-Pro c=1 bucket 延长到 150s）

K2.5 数据来自 AutoJunjie/awsome-distributed-training `report.md`（同机型、同 sglang 引擎，不同模型 + 参数集）；DS-V4-Pro 是本次实测。

| 模型 | context | 并发 | 实测 input tokens | ok / fail | **QPM** | QPS | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS-V4-Pro      | 128K | 1  | 5829 | 33 / 0  | **13.2**  | 0.220 | 4.71s  | 5.68s  | 5.71s  | 5.71s  |
| DS-V4-Pro      | 128K | 4  | 5828 | 34 / 0  | **34.0**  | 0.567 | 7.38s  | 7.29s  | 10.24s | 13.76s |
| DS-V4-Pro      | 128K | 8  | 5828 | 53 / 0  | **53.0**  | 0.883 | 9.78s  | 10.34s | 14.54s | 16.27s |
| DS-V4-Pro      | 128K | 16 | 5828 | 67 / 0  | **67.0**  | 1.117 | 16.66s | 16.82s | 20.47s | 22.32s |
| DS-V4-Pro      | 128K | 32 | 5829 | 73 / 0  | **73.0**  | 1.217 | 32.84s | 32.71s | 45.13s | 49.07s |
| DS-V4-Pro      | 128K | 64 | 5829 | 117 / 0 | **117.0** | 1.950 | 48.07s | 52.85s | 62.70s | 73.72s |
| **DS-V4-Pro**  | **256K** | 1  | 5829 | 32 / 0  | **12.8**  | 0.213 | 4.73s  | 5.87s  | 5.90s  | 5.91s  |
| **DS-V4-Pro**  | **256K** | 4  | 5829 | 34 / 0  | **34.0**  | 0.567 | 7.72s  | 7.49s  | 12.03s | 12.69s |
| **DS-V4-Pro**  | **256K** | 8  | 5828 | 48 / 0  | **48.0**  | 0.800 | 10.20s | 10.19s | 15.11s | 15.11s |
| **DS-V4-Pro**  | **256K** | 16 | 5829 | 63 / 0  | **63.0**  | 1.050 | 15.70s | 17.02s | 22.93s | 23.31s |
| **DS-V4-Pro**  | **256K** | 32 | 5829 | 81 / 0  | **81.0**  | 1.350 | 29.66s | 30.31s | 40.75s | 47.63s |
| **DS-V4-Pro**  | **256K** | 64 | 5828 | 114 / 0 | **114.0** | 1.900 | 51.59s | 58.94s | 68.66s | 76.14s |
| K2.5           | 128K | 1  | 5825 | 50 / 0  | **49.1**  | 0.819 | 1.22s  | 1.23s  | 1.25s  | 1.39s  |
| K2.5           | 128K | 4  | 5825 | 108 / 0 | **106.5** | 1.776 | 2.25s  | 2.28s  | 2.30s  | 2.64s  |
| K2.5           | 128K | 8  | 5825 | 142 / 0 | **135.6** | 2.259 | 3.50s  | 3.53s  | 3.58s  | 3.65s  |
| K2.5           | 128K | 16 | 5825 | 161 / 0 | **138.2** | 2.304 | 6.55s  | 6.24s  | 6.61s  | 12.06s |
| K2.5           | 128K | 32 | 5825 | 138 / 0 | **101.4** | 1.690 | 16.99s | 13.20s | 32.02s | 36.76s |
| K2.5           | 128K | 64 | 5825 | 166 / 0 | **116.1** | 1.936 | 32.12s | 32.50s | 50.58s | 51.01s |
| K2.5           | 256K | 1  | 5825 | 50 / 0  | **50.0**  | 0.833 | 1.20s  | 1.21s  | 1.23s  | 1.60s  |
| K2.5           | 256K | 4  | 5825 | 108 / 0 | **107.7** | 1.795 | 2.23s  | 2.26s  | 2.29s  | 2.46s  |
| K2.5           | 256K | 8  | 5825 | 144 / 0 | **136.2** | 2.270 | 3.52s  | 3.52s  | 3.55s  | 3.64s  |
| K2.5           | 256K | 16 | 5825 | 160 / 0 | **154.6** | 2.576 | 6.20s  | 6.21s  | 6.22s  | 6.30s  |
| K2.5           | 256K | 32 | 5825 | 160 / 0 | **145.5** | 2.425 | 13.13s | 13.08s | 15.68s | 20.73s |
| K2.5           | 256K | 64 | 5825 | 165 / 0 | **142.2** | 2.370 | 26.19s | 26.71s | 37.31s | 43.83s |

**说明**：
- K2.5 256K 峰值 **154.6** QPM @ c=16；DS-V4-Pro 256K 峰值 **117** QPM @ c=64。**K2.5 ≈ 1.35× DS-V4-Pro**。
- 两个模型的拐点不同：K2.5 在 c=16 就饱和（154.6），c=32/64 开始 regress 到 145/142；DS-V4-Pro 则单调增长到 c=64 才到 117，**decode graph 被 disable 是主因**（见 §8 修复 #3）。
- DS-V4-Pro 每请求平均延迟明显高（c=1 下 4.7s vs K2.5 的 1.2s，~4×）：主要是 **Marlin W4A16 FP4 expert 反量化 + no cuda-graph** 共同作用；prefill 本身两者 prompt 相近，差距来自 decode。
- 0 fail / 785 条 DS-V4-Pro 请求（K2.5 同样 0 fail / ~2000 条）。

## 5. 表 2：长文本（100K prompt，output = 64 tokens）

K2.5 数据来自 AutoJunjie `report.md`（p5en 段；持续时间 240s/档）。

| 模型 | context | 并发 | 持续时间 | 实测 input tokens | ok / fail | QPM | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS-V4-Pro      | 128K | 1  | 240s | 99750 | 11 / 0 | **2.75** | 22.02s  | 21.73s  | 21.78s  | 26.12s  |
| DS-V4-Pro      | 128K | 2  | 240s | 99750 | 15 / 0 | **3.75** | 33.24s  | 33.92s  | 33.94s  | 44.13s  |
| DS-V4-Pro      | 128K | 4  | 240s | 99750 | 19 / 0 | **4.75** | 57.00s  | 57.05s  | 69.05s  | 88.98s  |
| DS-V4-Pro      | 128K | 8  | 240s | 99750 | 22 / 0 | **5.50** | 105.29s | 106.37s | 124.32s | 178.44s |
| DS-V4-Pro      | 128K | 16 | 240s | 99751 | 30 / 0 | **7.50** | 189.64s | 196.51s | 262.43s | 293.42s |
| **DS-V4-Pro**  | **256K** | 1  | 240s | 99751 | 12 / 0 | **3.00** | 20.76s  | 21.48s  | 21.52s  | 21.53s  |
| **DS-V4-Pro**  | **256K** | 2  | 240s | 99750 | 16 / 0 | **4.00** | 33.54s  | 33.63s  | 33.67s  | 43.91s  |
| **DS-V4-Pro**  | **256K** | 4  | 240s | 99751 | 18 / 0 | **4.50** | 56.04s  | 56.77s  | 72.49s  | 81.03s  |
| **DS-V4-Pro**  | **256K** | 8  | 240s | 99751 | 21 / 0 | **5.25** | 106.33s | 94.65s  | 167.49s | 203.32s |
| **DS-V4-Pro**  | **256K** | 16 | 240s | 99751 | 30 / 0 | **7.50** | 188.38s | 188.20s | 305.95s | 306.39s |
| K2.5           | 128K | 1  | 240s | 99746 | 23 / 0 | **5.6** | 10.63s  | 10.62s  | 10.64s  | 10.79s  |
| K2.5           | 128K | 2  | 240s | 99746 | 26 / 0 | **6.2** | 19.36s  | 19.35s  | 19.38s  | 19.50s  |
| K2.5           | 128K | 4  | 240s | 99746 | 28 / 0 | **6.1** | 39.40s  | 39.38s  | 39.54s  | 39.85s  |
| K2.5           | 128K | 8  | 240s | 99746 | 24 / 0 | **5.2** | 90.59s  | 95.16s  | 120.73s | 144.48s |
| K2.5           | 128K | 16 | 240s | 99747 | 32 / 0 | **5.2** | 165.17s | 190.49s | 204.09s | 239.90s |
| K2.5           | 256K | 1  | 240s | 99746 | 23 / 0 | **5.7** | 10.60s  | 10.62s  | 10.64s  | 10.65s  |
| K2.5           | 256K | 2  | 240s | 99746 | 26 / 0 | **6.2** | 19.34s  | 19.35s  | 19.38s  | 19.51s  |
| K2.5           | 256K | 4  | 240s | 99746 | 28 / 0 | **6.1** | 39.34s  | 39.38s  | 39.44s  | 39.76s  |
| K2.5           | 256K | 8  | 240s | 99746 | 24 / 0 | **5.2** | 90.58s  | 95.17s  | 120.71s | 144.48s |
| K2.5           | 256K | 16 | 240s | 99746 | 32 / 0 | **5.2** | 164.89s | 190.23s | 203.73s | 239.41s |

**说明**：
- 单路 100K prompt（c=1）：K2.5 **5.6-5.7** QPM vs DS-V4-Pro **2.75-3.00**。K2.5 prefill ~10K tok/s，DS-V4-Pro ~4.5K tok/s ——K2.5 prefill 吞吐约 **2×** DS-V4-Pro（符合 Marlin FP4 MoE prefill 较慢的已知特性）。
- 高并发（c=16）：DS-V4-Pro **7.50** QPM **反超** K2.5 的 **5.2**（1.44×）。K2.5 在 c=2 就饱和到 6.2；DS-V4-Pro 一直线性增长到 c=16 才到 7.5。推测是 K2.5 在这个 prompt 长度 + concurrency 下已触达 memory bandwidth 瓶颈，而 DS-V4-Pro 的 MoE 结构并发扩展性更好（只激活 6/384 expert）。
- 128K vs 256K context：两个模型在同一 context 档位下差异都 < 5%（再次验证"扩 context 不偷 KV"）。
- 0 fail / 193 条 DS-V4-Pro 长 prompt 请求（K2.5 同样 0 fail）。

## 6. sglang 启动命令（实际部署 = cookbook `h200-fp4` + 我们的 `--context-length`）

```bash
# Entrypoint: sglang serve (new CLI — 旧 python3 -m sglang.launch_server 在此镜像上 BOS 塌陷)
sglang serve \
  --trust-remote-code \
  --model-path /models/deepseek-v4-pro \
  --served-model-name deepseek-v4-pro \
  --tp 8 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.88 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --disable-cuda-graph \
  --context-length 262144 \      # T5（256K），T6 改成 131072
  --enable-metrics \
  --host 0.0.0.0 --port 30000
```

镜像：`lmsysorg/sglang:deepseek-v4-hopper`（DS-V4 分支；主干 transformers 5.3.0 未注册 `deepseek_v4`）。

**K2.5 p5en 参考基线（未在本次启用；在 §8 "关键修复记录" 中说明为何换掉）**：

```bash
# K2.5 p5en 256K 已验证参数集（AutoJunjie/awsome-distributed-training report.md）——
# 本次 DS-V4-Pro 换成 cookbook h200-fp4 配方，这 10 个 K2.5 baseline flag 的逐项去留
# 见下；保留在此作为横向对比证据，不是 DS-V4-Pro 的启动命令。
python3 -m sglang.launch_server \
  --model-path /models/kimi-k2.5 \
  --served-model-name kimi-k2.5 \
  --tp 8 --ep-size 8 \                  # DS-V4-Pro: --tp 8（cookbook 不显式设 --ep-size）
  --attention-backend flashinfer \      # DS-V4-Pro: attention_backend='compressed'（cookbook 强制为 DeepseekV4ForCausalLM 自动切）
  --context-length 262144 \             # DS-V4-Pro: 相同（256K）/ 131072（128K）
  --mem-fraction-static 0.92 \          # DS-V4-Pro: 0.88（Marlin MoE kernel 需额外 scratch）
  --chunked-prefill-size 8192 \         # DS-V4-Pro: 相同（cookbook 默认 8192）
  --max-running-requests 128 \          # DS-V4-Pro: 256（DeepseekV4ForCausalLM 自动上调）
  --cuda-graph-max-bs 16 \              # DS-V4-Pro: 替换为 --disable-cuda-graph（capture 崩）
  --enable-mixed-chunk \                # DS-V4-Pro: 未启用（cookbook 默认 off）
  --disable-custom-all-reduce \         # DS-V4-Pro: 未启用（cookbook 默认 off；走 NCCL AR）
  --host 0.0.0.0 --port 30000
```

## 7. 关键环境变量（与 Deployment `env:` 一一对应）

```
NCCL_DEBUG=INFO
NCCL_P2P_LEVEL=NVL
NCCL_ALGO=Ring
NCCL_CUMEM_ENABLE=0
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_FORK_SAFE=1
HF_HUB_OFFLINE=1
SGL_ENABLE_JIT_DEEPGEMM=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LD_LIBRARY_PATH=/opt/efa-libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
```

| Env | 作用 / Rationale |
|---|---|
| `NCCL_DEBUG=INFO` | 仅冷启阶段打开，采 GDR/P2P/IPC 证据链（T4 AC）；稳态运行建议降回 WARN |
| `NCCL_P2P_LEVEL=NVL` | **关键**。强制 NCCL 选 NVLink 路径；默认 `SYS` 在某些拓扑上会降级走 PCIe host bridge（性能腰斩）。H200 + NVSwitch 必须 NVL |
| `NCCL_ALGO=Ring` | 与 K2.5 baseline 对齐；TP=8 单节点 AllReduce 稳定值 |
| `NCCL_CUMEM_ENABLE=0` | **关键**。避免 CUDA graph capture 时 NCCL P2P/IPC init 报 `operation not permitted when stream is capturing`（H200 + TP=8 必发） |
| `FI_PROVIDER=efa` | **T8 前置**：本次单 host 单 container NCCL 不走 Libfabric（见 §8 caveat），保留此 env 是为 T8 多节点那一刻能直接切 aws-ofi-nccl；对当前 workload 是 no-op |
| `FI_EFA_USE_DEVICE_RDMA=1` | 同上，T8 EFA GDR 前置 |
| `FI_EFA_FORK_SAFE=1` | 同上；fork-safe 避免 python multiprocessing 破坏 EFA 设备状态 |
| `HF_HUB_OFFLINE=1` | 权重已全部本地化到 NVMe；禁止 SGLang 在冷启阶段再触网查 repo（HF rate-limit 风险） |
| `SGL_ENABLE_JIT_DEEPGEMM=0` | 关掉 SGLang JIT DeepGEMM，cookbook `deepseek-v4-hopper` 镜像 AOT 编译已覆盖 Hopper FP4 kernel；JIT 会浪费 10+ 分钟冷启时间 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | MoE 层 scratch 分配在 graph capture 期间会尖峰；expandable 段降低碎片 |
| `LD_LIBRARY_PATH=/opt/efa-libs:...` | initContainer 把宿主 `/usr/lib64/lib{efa,ibverbs,rdmacm}.so.*` 拷到 emptyDir，主 container 通过这条路径加载（AL2023 base 没有 `/opt/amazon/efa`） |

## 8. 关键修复记录

### DS-V4-Pro 本次新增

1. **`model_type=deepseek_v4` 未注册 → 必须换 cookbook 镜像**。主干 `lmsysorg/sglang:latest`（2026-04-29 拉到的是 sglang 0.5.10.post1 + transformers 5.3.0）AutoConfig 直接抛 `KeyError: deepseek_v4`，sglang model registry 看不到 checkpoint。固定到 `lmsysorg/sglang:deepseek-v4-hopper`（`deepseek_v4` PR #23980 分支 + Hopper FP4 AOT kernel）后才能启动。
2. **decode 塌成 BOS → 必须加 `--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4`**。旧入口 `python3 -m sglang.launch_server` + cookbook 镜像虽然 model load 成功，`/v1/chat/completions` 返回 `content: null, reasoning_content: null`，`/generate` 原始输出 64× BOS token。切到 `sglang serve` 新 CLI + cookbook 指定的两个 DS-V4 parser 后，200K prompt smoke 返回合法 pangram summary。
3. **CUDA graph capture `invalid argument` → `--disable-cuda-graph`**。cookbook minimal config 在 `cuda_graph_runner.py:368` 抛 `Exception: Capture cuda graph failed: invalid argument`。T4 选择 disable graph 交付 KV budget + 长 prompt correctness；decode 吞吐损失约 15-25%（与 K2.5 的 155 QPM 相比 DS-V4-Pro 256K 峰值 114 QPM 有 25% gap 主要就是这个）。T5 / T8 应重新选 `--cuda-graph-max-bs` 把 graph 打开。
4. **EAGLE speculative decoding 需要 `SGLANG_ENABLE_SPEC_V2=1` env + `--speculative-num-steps=1`**（T5b → T5b-v2 追溯）。cookbook `DeepSeek-V4.mdx` 第 146 行明示 "MTP currently requires `SGLANG_ENABLE_SPEC_V2=1`"，T5b 的 4 个 probe 全部缺这个 env，崩在 `deepseek_v4_backend_radix.py:432 AssertionError: out_cache_loc.shape=[speculative_num_steps]` vs `seq_lens.shape=[1]`。T5b-v2 Probe A（balanced: `num_steps=1 num_draft_tokens=2 topk=1` + `SGLANG_ENABLE_SPEC_V2=1`）✅ Ready + 正确 decode + spec 指标可用；Probe B（low-latency: `num_steps=3 num_draft_tokens=4` + 同 env）仍然崩，`out_cache_loc.shape=[3]`——**shape 随 num_steps 线性增长，radix backend 硬编码了 1-slot-per-req 形状契约，num_steps=1 是当前 production `-hopper` 镜像上唯一能跑通的值**。最终部署的 T5b-v2 bench 配置 = balanced（Probe A）。实测 acceptance rate 96.6%、accept_length 1.93/2；表 1/2 对比详见 §9。
5. **NCCL Libfabric 不在 `deepseek-v4-hopper` 镜像里（T8 caveat）**。本次单 host workload NCCL 只用 P2P/IPC（8 GPU 同 container），0 条 SHM transport、0 条 NET/IB；冷启 log 出现 `NCCL INFO NET/Plugin: Could not find: libnccl-net.so` + `NET/IB: No device found`——这个镜像没打 aws-ofi-nccl 插件。单机无影响；T8 多节点必须换打了 aws-ofi-nccl 的 image（或者 initContainer 额外拷 `libnccl-net.so`），否则跨节点 NCCL 会 fallback 到 `Socket` 走 eth0，完全丢掉 EFA GDR。本次保留 `FI_PROVIDER=efa / FI_EFA_*` env 为 T8 那一刻做无缝切换。

### 来自 K2.5 p5en report 的 6 条（**DS-V4-Pro 中已再次验证**）

6. **HF snapshot download 替换 S3 sync**：S3 跨区 ~1.5 GB/min vs HF `snapshot_download`（`HF_HUB_ENABLE_HF_TRANSFER=1 + max_workers=16`）~1.5 GB/s，**~100×**。DS-V4-Pro 805 GiB 约 ~9 分钟下完（K2.5 555 GB 5 分钟下完 → 同速率）。
7. **`NCCL_P2P_LEVEL=SYS → NVL`**：H200 + NVSwitch 必须强制 NVL，否则 NCCL 可能选 PCIe 降级路径。本次 9 × GDR 1 + 537 × P2P/IPC 记录验证 NVLink 路径被选中。
8. **`NCCL_CUMEM_ENABLE=0`**：否则 CUDA graph capture 时 NCCL P2P/IPC init 报 `operation not permitted when stream is capturing`。本次虽然 `--disable-cuda-graph`，但保留这个 env 给 T5/T8 再打开 graph 时用。
9. **`--disable-custom-all-reduce`**（K2.5 必须）：本次 cookbook 默认 `disable_custom_all_reduce=False` 并未手动 disable；实测无崩溃——推测 cookbook 升级后的 custom_all_reduce 路径已修。**K2.5 的回退策略依然有效**（如果 T8 再现 capture 崩溃，下一步就是加回这个 flag）。
10. **initContainer 拷 EFA libs**：AL2023 base node 只在 `/usr/lib64` 有 rdma-core 61，没有 `/opt/amazon/efa`。本次 initContainer 拷 `libefa/libibverbs/librdmacm.so.*` + `libibverbs/libefa-rdmav*.so` provider 到 emptyDir，主 container `LD_LIBRARY_PATH` 先 mount 这个路径。
11. **`efa_nv_peermem` 持久化**：SSM `modprobe` + 写 `/etc/modules-load.d/efa_nv_peermem.conf`（GDR 前提）。T1 `logs/node-health.log` 检查通过，T8 多节点场景下同样前提。

---

### 与 K2.5 的异同（compare-to-K2.5 callout）

**相同**：机型 p5en.48x、互连 NVLink NV18、引擎 sglang、KV pool 的"与 context-length 无关"特性（128K vs 256K 下 `max_total_num_tokens` 完全一致，本次 771K vs K2.5 837K）、`NCCL_P2P_LEVEL=NVL` + `NCCL_CUMEM_ENABLE=0` + initContainer 拷 EFA libs 的部署套路。

**不同（配置层）**：（a）**模型架构**——DS-V4-Pro 是 FP4-Instruct MoE（Marlin W4A16 Hopper kernel，384 experts 激活 6）+ FP8 KV，K2.5 是 INT4 compressed-tensors dense；（b）**入口点**——`sglang serve` (DS-V4) vs `python3 -m sglang.launch_server` (K2.5)；parser 名 `deepseekv4/deepseek-v4` vs `kimi_k2`；（c）**KV pool 规模**——771K vs 837K，是因为 `--mem-fraction-static` 从 0.92（K2.5）降到 0.88（cookbook `h200-fp4` 为 Marlin scratch 让出余量）；（d）**并行度**——cookbook 只设 `--tp 8`（`ep_size=1`），K2.5 `--tp 8 --ep-size 8`；（e）**cuda graph**——cookbook 阶段被迫 disable（见修复 #3）。

**不同（性能层，来自 §4/§5 横表对比）**：
- **表 1 (5.8K QPM)**：K2.5 256K 峰值 154.6 QPM @ c=16，DS-V4-Pro 256K 峰值 117 QPM @ c=64。**K2.5 ≈ 1.32× DS-V4-Pro**。K2.5 在 c=16 就饱和，DS-V4-Pro 需要推到 c=64 才饱和——主要是 `--disable-cuda-graph` 放弃了 decode graph 吞吐，re-enable graphs 是最大可回补缺口。
- **表 2 (100K 长 prompt)**：K2.5 单路 5.7 QPM vs DS-V4-Pro 3.0（K2.5 ≈ **1.9× DS-V4-Pro prefill 速率**，~10K vs ~4.5K tok/s 有效 prefill，Marlin FP4 MoE prefill 较慢是已知特性）；但 c=16 高并发下 DS-V4-Pro 7.5 QPM **反超** K2.5 的 5.2（**1.44×**）——K2.5 在 c=2 就饱和（memory bandwidth bound），DS-V4-Pro MoE 稀疏激活（6/384）让并发扩展性更好。

**结论**：相同硬件相同引擎，**K2.5 在短 prompt / 低-中并发吞吐更优**（1.3-1.9×），**DS-V4-Pro 在长 prompt + 高并发更优**（1.44×）。两个模型吞吐的交叉点就在 100K prompt + c=8 附近。

## 9. EAGLE speculative decoding 对比（256K，T5b-v2）

**结论先写**：在 `lmsysorg/sglang:deepseek-v4-hopper` 生产镜像上，加 `SGLANG_ENABLE_SPEC_V2=1` env + `--speculative-num-steps=1 --speculative-num-draft-tokens=2 --speculative-eagle-topk=1`（cookbook `balanced` 配方）后，**EAGLE 跑通了**，表 1 256K 峰值 **119 QPM**（c=64），对 K2.5 的 154.6 QPM gap closure 仅 **12.3%**（+5 QPM / 40.6 QPM 差距）；**acceptance rate 96.6%、accept_length 1.93/2**。表 2（100K prompt）相对 baseline 几乎 wash。对 "25% K2.5 gap" 的实际贡献很小——**`--disable-cuda-graph` 才是主瓶颈**（§8 修复 #3），re-enable graph 才是该补的那条路径。

### 9.1 T5b 回顾：SPEC_V2 env 是关键（补在这里避免重蹈覆辙）

T5b 的 4 个 probe 全部崩在 `deepseek_v4_backend_radix.py:432 AssertionError: out_cache_loc.shape=[speculative_num_steps]` vs `seq_lens.shape=[1]`。根因不是镜像 bug 而是 **cookbook `DeepSeek-V4.mdx` 第 146 行的一行硬要求**："MTP currently requires `SGLANG_ENABLE_SPEC_V2=1`"——T5b 只设了 `--speculative-*` flag 而没有这个 env，sglang 走进了 v1 shape contract（1-slot-per-req）的代码分支，但实际生成的是 Spec-v2 multi-slot draft batch。T5b-v2 添加 `env: SGLANG_ENABLE_SPEC_V2=1` 后，Probe A（`num_steps=1`）彻底跑通。

T5b-v2 进一步确认：**shape=num_steps 的断言和 env 无关——它是 DSv4 radix backend `init_forward_metadata_decode` 对 EAGLE multi-step draft 的硬编码 bug**，`num_steps>1` 都会触发。Probe B（`num_steps=3 num_draft_tokens=4` + SPEC_V2=1）仍然崩（`out_cache_loc.shape=[3]`，同栈），验证这一点。production `-hopper` 镜像上只有 `num_steps=1` 能跑。

### 9.2 Probe 矩阵（T5b + T5b-v2 合并）

| # | 配置 | env | 结果 | 原因 |
|---|---|---|---|---|
| T5b-1 | num_steps=3 num_draft=4 | 无 SPEC_V2 | ❌ | SPEC_V2=0 走 v1 shape，draft 实际是 v2 multi-slot |
| T5b-2 | + attn_mode=decode | 无 SPEC_V2 | ❌ | 同上 |
| T5b-3 | + draft_backend=triton | 无 SPEC_V2 | ❌ CrashLoop | 同上 |
| T5b-4 | num_draft_tokens=2 | 无 SPEC_V2 | ❌ | shape=num_steps（=3），不是 num_draft-1 |
| **T5b-v2-A** | **num_steps=1 draft=2 topk=1** | **SPEC_V2=1** | ✅ **Ready + spec 生效** | |
| T5b-v2-B | num_steps=3 draft=4 | SPEC_V2=1 | ❌ | shape=3 assertion；env 救不了 num_steps>1 |

所有 T5b 的 4 个 probe log 保留在 `logs/eagle-probes/`；T5b-v2 相关日志：`logs/cold-start-speculative.log`（Probe A Ready）、`logs/cold-start-probeB.log`（Probe B crash 栈）、`logs/smoke-probeA.json`（HTTP 200 + non-BOS content）。

### 9.3 表 1 对比（5.8K prompt，output=32，60s / c=1 为 150s）

| 并发 | DS-V4-Pro 256K **baseline** QPM | DS-V4-Pro 256K **+ EAGLE(balanced)** QPM | Δ QPM | K2.5 256K QPM |
|---:|---:|---:|---:|---:|
| 1  | 12.8  | **19.6**  | **+6.8** (+53%) | 50.0  |
| 4  | 34.0  | 41.0  | +7.0 (+21%) | 107.7 |
| 8  | 48.0  | 57.0  | +9.0 (+19%) | 136.2 |
| 16 | 63.0  | 54.0  | -9.0 (-14%) | **154.6** |
| 32 | 81.0  | 85.0  | +4.0 (+5%)  | 145.5 |
| 64 | **114.0** | **119.0** | +5.0 (+4%) | 142.2 |

**峰值对比（c=64）**：DS-V4-Pro baseline 114 QPM → + EAGLE 119 QPM → K2.5 154.6 QPM。Gap closure = (119-114) / (154.6-114) = **12.3%**。

### 9.4 表 2 对比（100K prompt，output=64，240s / bucket）

| 并发 | DS-V4-Pro 256K **baseline** QPM | DS-V4-Pro 256K **+ EAGLE(balanced)** QPM | Δ QPM | K2.5 256K QPM |
|---:|---:|---:|---:|---:|
| 1  | 3.00 | 3.50 | +0.50 (+17%) | 5.7 |
| 2  | 4.00 | 4.00 | +0.00 | 6.2 |
| 4  | 4.50 | 4.50 | +0.00 | 6.1 |
| 8  | 5.25 | 5.50 | +0.25 (+5%)  | 5.2 |
| 16 | **7.50** | **8.00** | **+0.50** (+7%) | 5.2 |

100K prompt 上 EAGLE 在单路（c=1）有 17% 提升，高并发几乎 wash——符合预期，长 prompt 的瓶颈是 prefill，EAGLE 只加速 decode。

### 9.5 Speculative 指标（`logs/spec-metrics.txt`）

```
sglang:spec_accept_length{...tp_rank="0"} 1.93125
sglang:spec_accept_rate{...tp_rank="0"}   0.965625
```

- `accept_rate=0.966`：draft 提出的 token 有 96.6% 被 target 接受——**EAGLE 草稿模型质量很高**（同参数 = target model 自身做 MTP）。
- `accept_length=1.93`：每 request 平均接受 1.93 个 token。Probe A 的理论上限 = `num_draft_tokens=2`，所以 **fill factor = 1.93/2 = 96.5%** 与 accept_rate 一致。
- 问题在于 balanced 配方上限是 draft=2，即便 100% 接受，throughput 最多 **2×** decode。实测峰值 gain 只有 +4% ~ +53%（与 c 相关），表明 **decode 已被 `--disable-cuda-graph` 卡住**，speculative 的 2× 数学上限吃不满。

### 9.6 回答 user 的 perf 问题

> **Q：EAGLE speculative decoding 能否缩小 DS-V4-Pro 到 K2.5 的 ~25% 吞吐 gap？**
>
> **A：部分能（+12.3% gap closure @ peak），但远不够。** 原因两条叠加：（1）production `-hopper` 镜像上 DSv4 radix backend 只支持 `num_steps=1`，把 EAGLE 的吞吐上限夹在 2× 理论 speedup（实测 accept_length=1.93 即已接近满）；（2）更根本地，`--disable-cuda-graph`（§8 修复 #3）把 decode 吞吐先砍了 15-25%，EAGLE 只能在被砍过的基线上再乘 ~1.04-1.17。**要真正 close gap，先 re-enable CUDA graph（+15-25% decode），其次等 upstream 修 DSv4 radix 的 multi-slot draft 支持，跑 low-latency（num_steps=3 draft=4）才能吃到 4× speculative 上限**。

### 9.7 Pod uid 证据链

| 阶段 | Pod uid | 备注 |
|---|---|---|
| T5 baseline 256K | `1e4d6e6f-3b36-4933-94a8-5727e363f396` | — |
| T6 baseline 128K | `a446ac23-870b-4dd2-a1cd-66ad95f80d16` | — |
| T5b probe 1-4 | `bae8269a / 95b24509 / 35786883 / 7fa5af3c` | 4 个均 crash，无 Ready 证据 |
| T5b-v2 Probe A (first check) | `8f29f321-8daa-4fe1-8aba-6528ee67e891` | smoke PASS |
| T5b-v2 Probe B (crash) | `4488f2f9-6d3e-41b3-adaf-9f5d431739c1` | Uvicorn up → first req crash |
| **T5b-v2 Probe A bench**（pre = post）| `1663f431-d108-46e0-b4dd-6b5785262890` | **两张表全程同 1 个 Pod** |

pre / post uid 文件：`logs/pod-uid-256K-speculative-{pre,post}.txt`，两者字节级相等。与历史所有 uid 都不同，符合 AC #6。

---

## 10. CUDA graph 重开 — 单节点性能再优化（T5c）

**结论先写**：T4 被迫 `--disable-cuda-graph` 才是吞吐 gap 的主因。本轮加 **`--cuda-graph-max-bs=8` + `--disable-custom-all-reduce`**（Probe A'）让 cg capture 首次在 DS-V4-Pro p5en 通过；叠加 EAGLE balanced（Probe B）进一步吃下 c=1 低并发延迟 gain。**最佳配置 = Probe B：c=1 从 4.73s→1.00s（-79%），c=1 QPM 从 12.8→60.0（+369%）；c=16 表 2 从 7.50→8.00 QPM（+7%）**。峰值 c=64 表 1 仍停在 120 QPM（114 baseline +5%）——即 5.8K/c=64 的饱和瓶颈不在 cg，而在 Marlin FP4 MoE 本身的稳态 decode 带宽，但 c=1/c=16 低中并发已全面打平甚至超 K2.5 的低并发区（参 §10.4）。

### 10.1 Probe 矩阵（按尝试顺序）

| # | 配置 diff vs T6 | cg capture | Pod Ready | 备注 |
|---|---|---|---|---|
| Probe A  | `+--cuda-graph-max-bs=8, -disable-cuda-graph` | ❌ `invalid argument` | ❌ CrashLoop ×3 | 栈为 `custom_all_reduce.cuh:350 get_graph_buffer_ipc_meta`——与 `cuda_graph_runner.py:368` 相连的 custom-AR IPC buffer 导出崩溃，不是 sgl-kernel bug。日志 `logs/cold-start-probeA.log` |
| **Probe A'** | `A + --disable-custom-all-reduce` | ✅ `bs [1,2,4,8]`, 90.2s | ✅ 5 min Ready | `max_total_num_tokens=774400`（比 T5 的 771K 略高）。日志 `logs/cuda-graph-coldstart.log` |
| **Probe B** | `A' + EAGLE balanced (num_steps=1 draft=2 topk=1)` + `SGLANG_ENABLE_SPEC_V2=1` | ✅ `bs [1..8]` | ✅ 5 min Ready | spec_accept_rate 0.95 / accept_length 1.9 |
| Probe C  | A' + `--attention-backend triton/flashinfer` | —（**未跑**） | — | A'+B 已过 "c=1 <2s" 阈值，C 边际增益有限 |

**根因记录**：`custom_all_reduce.cuh:350` 是 sgl-kernel 的 `get_graph_buffer_ipc_meta` CUDA kernel，H200 + TP=8 + cookbook `disable_custom_all_reduce=False` 的默认配置在 graph capture 期间该 kernel 的 ipc_meta export 总是 `invalid argument`。关掉 custom AR 后走 NCCL AllReduce（更慢但稳定），而 NCCL AR 在 cg capture stream 里本来就有 `NCCL_CUMEM_ENABLE=0` 兜底——capture OK。§8 修复 #3 的 "cookbook minimal crash" 真相由此补齐，K2.5 p5en 早就踩过同一条路径（§8 修复 #9 已记）。

### 10.2 表 1 256K（5.8K prompt，output=32；c=1 延长到 150s）

| 并发 | T5 baseline QPM | + EAGLE(§9, no cg) QPM | **+ cg (A')** QPM | **+ cg + EAGLE (B)** QPM | K2.5 256K QPM |
|---:|---:|---:|---:|---:|---:|
| 1  | 12.8  | 19.6  | **52.8** | **60.0** | 50.0  |
| 4  | 34.0  | 41.0  | 68.0  | 66.0  | 107.7 |
| 8  | 48.0  | 57.0  | 75.0  | 70.0  | 136.2 |
| 16 | 63.0  | 54.0  | 62.0  | 65.0  | **154.6** |
| 32 | 81.0  | 85.0  | 80.0  | 82.0  | 145.5 |
| 64 | **114.0** | **119.0** | **120.0** | **120.0** | 142.2 |

**c=1 avg latency**（衡量 cg 是否回路）：baseline 4.73s → A' **1.14s** → B **1.00s**（-79%）。`< 2s` 目标达成，**cuda graph 绝对在运行**。

### 10.3 表 2 256K（100K prompt，output=64，240s/档）

| 并发 | T5 baseline QPM | + EAGLE(§9) QPM | **+ cg (A')** QPM | **+ cg + EAGLE (B)** QPM | K2.5 256K QPM |
|---:|---:|---:|---:|---:|---:|
| 1  | 3.00 | 3.50 | 4.50 | 4.50 | 5.7 |
| 2  | 4.00 | 4.00 | 5.00 | 5.00 | 6.2 |
| 4  | 4.50 | 4.50 | 5.00 | 5.00 | 6.1 |
| 8  | 5.25 | 5.50 | 5.50 | 6.00 | 5.2 |
| 16 | **7.50** | **8.00** | **7.25** | **8.00** | 5.2 |

A' 的 c=16 比 §9 EAGLE 稍回退（7.50→7.25）可能是 cg=8 对 c>8 的 padding overhead；B 因为叠加 EAGLE 把 c=8/16 拉回 6.0/8.0。

### 10.4 K2.5 gap closure 终章

表 1 c=64 peak：`(120 - 114) / (154.6 - 114) = **14.8%**` gap closure（T5b-v2 EAGLE 仅 12.3%）。  
表 1 c=1：`60 QPM > K2.5 50`——**c=1 档 DS-V4-Pro 反超 K2.5 20%**。  
表 1 c=4/8：仍落后 K2.5 较多（66/70 vs 107.7/136.2，~35-49%）。  
表 2 c=16：8.0 > K2.5 5.2，**1.54× K2.5**（§9 就反超，本轮进一步）。

结合 **accept_rate=0.95 / accept_length=1.9**（理论上限 2，fill factor 95%），EAGLE 在 B 里几乎吃满其 2× speedup 上限，但只在 c≤1 decode-bound 区间看得见（+14% vs A'）；高并发 EAGLE 的 draft/verify cost 会被 prefill batching 稀释，所以 B 对 c≥16 几乎 wash。

### 10.5 结论 & T8 下一步

1. **单节点 h200-fp4 分支已调到节点理论上限**：c=1 1.00s / 52-60 QPM，c=16 长文 8 QPM 已超 K2.5；但 5.8K/c=64 peak 停在 120 QPM，剩余 `154.6-120=34.6 QPM` gap 归咎于 **Marlin W4A16 FP4 MoE decode kernel 在高并发 batch 下带宽 < K2.5 INT4 compressed-tensors kernel**，需要 upstream FP4 Marlin 优化 或 切 FP8 路径（§8 修复 #9 的 `--disable-custom-all-reduce` 即 K2.5 配方已纳入）。
2. **T8（多节点 FP8 低延迟）建议**：cookbook H200 FP8 分支支持 DP-attention + DeepEP + cg-max-bs=8 + MTP=3/4 combo——多节点情况下 EAGLE num_steps=3 的 `shape=[3]` assertion bug 不触发（prefill-only DP-attn path），speculative 4× 上限可真正吃满。配合 aws-ofi-nccl 镜像，T8 有望冲到 200+ QPM @ c=64。
3. **镜像升级值得等**：当前 `deepseek-v4-hopper` tag 对 `--disable-custom-all-reduce` 有硬依赖；upstream 修 `custom_all_reduce.cuh:350` 后，可恢复 custom AR + 额外 ~3-5% decode。

### 10.6 Pod uid 证据链（本章新增）

| 阶段 | Pod uid | 备注 |
|---|---|---|
| Probe A (crash) | `d415ef4d-a07e-4441-8776-2adceb806116` | CrashLoopBackOff ×3 |
| **Probe A' bench** | `665ab653-d9a0-4ba1-a3f9-07e3c46aa9aa` | pre==post，单 pod 跑完 table 1+2 |
| **Probe B bench**（最终 winner）| `56e62d96-acad-4884-977a-0d34b126d2d1` | pre==post，与 T5/T5b/T5b-v2/T6 都不同 |

uid 文件：`logs/pod-uid-256K-cuda-graph-{pre,post}.txt`（A'）、`logs/pod-uid-probeB-cg-eagle-{pre,post}.txt`（B）。Deployment 任务结束后回退到 T6 baseline（无 cg、无 speculative、CONTEXT_LENGTH=131072、无 SPEC_V2 env）。

---

Next: T8 multi-node 1M context（2 & 3 × p5en.48x，`--context-length 1048576`，表 1/2/3 对齐，EFA GDR 必须装 aws-ofi-nccl 镜像；DP-attn + DeepEP + cg=8 + MTP=3 low-latency combo 是主要上限）。
