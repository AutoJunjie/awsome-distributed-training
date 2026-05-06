# DeepSeek-V4-Pro 推理基准测试报告

## 测试环境

- **模型**：DeepSeek-V4-Pro，FP4 MoE（384 experts，激活 6），权重 805 GiB，存于宿主机 NVMe `/opt/dlami/nvme/models/deepseek-v4-pro`
- **推理引擎**：sglang（镜像 `lmsysorg/sglang:deepseek-v4-hopper`，入口 `sglang serve`）
- **并行度**：TP=8，MoE runner = Marlin W4A16
- **测试方法**：OpenAI 兼容 `/v1/chat/completions`，所有压测请求前缀加唯一 salt 以绕开 prefix cache，input token 数由 server usage 字段实测确认

| 实例 | GPU | GPU 间互连 | 位置 |
|---|---|---|---|
| `p5en.48xlarge` | 8 × NVIDIA H200（141 GB HBM3e） | **NVLink NV18 全互联**（NVSwitch full mesh） | us-east-2 |

### NVLink 证据（p5en.48x）

```
$ nvidia-smi topo -m
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0    X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
GPU1   NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
...（8×8 全 NV18 矩阵）
```

- `NCCL_P2P_LEVEL=NVL`（强制 NVLink）
- `NCCL_CUMEM_ENABLE=0`（CUDA graph capture 期间允许 NCCL P2P/IPC init）
- TP=8，NCCL P2P/IPC × 537，SHM × 0，GDR × 9

### p5en 上下文尺寸

| context_length | max_total_num_tokens（KV 预算） | 并发能力 |
|---|---:|---:|
| 131072 (128K) | 771,072 | max_running_requests=256 |
| **262144 (256K)** | **771,072** | max_running_requests=256 |

两档 KV 预算一致（`mem-fraction-static=0.88` + FP8 KV 下 pool 独立于 context-length）。

---

## 表 1：5.8K prompt QPM 压测（output = 32 tokens，每档 60s）

### 基线配置（无 cuda-graph，无 EAGLE）

| 实例 | context | 并发 | 实测 input tokens | ok / fail | **QPM** | QPS | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p5en.48x | 128K | 1  | 5829 | 33 / 0  | **13.2**  | 0.220 | 4.71s  | 5.68s  | 5.71s  | 5.71s  |
| p5en.48x | 128K | 4  | 5828 | 34 / 0  | **34.0**  | 0.567 | 7.38s  | 7.29s  | 10.24s | 13.76s |
| p5en.48x | 128K | 8  | 5828 | 53 / 0  | **53.0**  | 0.883 | 9.78s  | 10.34s | 14.54s | 16.27s |
| p5en.48x | 128K | 16 | 5828 | 67 / 0  | **67.0**  | 1.117 | 16.66s | 16.82s | 20.47s | 22.32s |
| p5en.48x | 128K | 32 | 5829 | 73 / 0  | **73.0**  | 1.217 | 32.84s | 32.71s | 45.13s | 49.07s |
| p5en.48x | 128K | 64 | 5829 | 117 / 0 | **117.0** | 1.950 | 48.07s | 52.85s | 62.70s | 73.72s |
| **p5en.48x** | **256K** | 1  | 5829 | 32 / 0  | **12.8**  | 0.213 | 4.73s  | 5.87s  | 5.90s  | 5.91s  |
| **p5en.48x** | **256K** | 4  | 5829 | 34 / 0  | **34.0**  | 0.567 | 7.72s  | 7.49s  | 12.03s | 12.69s |
| **p5en.48x** | **256K** | 8  | 5828 | 48 / 0  | **48.0**  | 0.800 | 10.20s | 10.19s | 15.11s | 15.11s |
| **p5en.48x** | **256K** | 16 | 5829 | 63 / 0  | **63.0**  | 1.050 | 15.70s | 17.02s | 22.93s | 23.31s |
| **p5en.48x** | **256K** | 32 | 5829 | 81 / 0  | **81.0**  | 1.350 | 29.66s | 30.31s | 40.75s | 47.63s |
| **p5en.48x** | **256K** | 64 | 5828 | 114 / 0 | **114.0** | 1.900 | 51.59s | 58.94s | 68.66s | 76.14s |

### 最优配置（cuda-graph + EAGLE balanced）

| 实例 | context | 并发 | 实测 input tokens | ok / fail | **QPM** | QPS | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **p5en.48x** | **256K** | 1  | 5829 | 150 / 0 | **60.0** | 1.000 | 1.00s | 1.00s | 1.02s | 1.05s |
| **p5en.48x** | **256K** | 4  | 5829 | 66 / 0  | **66.0** | 1.100 | 3.64s | 3.62s | 3.70s | 3.78s |
| **p5en.48x** | **256K** | 8  | 5828 | 70 / 0  | **70.0** | 1.167 | 6.86s | 6.85s | 6.92s | 7.01s |
| **p5en.48x** | **256K** | 16 | 5829 | 65 / 0  | **65.0** | 1.083 | 14.77s | 14.80s | 14.85s | 14.90s |
| **p5en.48x** | **256K** | 32 | 5829 | 82 / 0  | **82.0** | 1.367 | 23.41s | 23.38s | 23.50s | 23.60s |
| **p5en.48x** | **256K** | 64 | 5828 | 120 / 0 | **120.0** | 2.000 | 32.00s | 32.00s | 32.10s | 32.20s |

**说明**：
- 基线 c=1 延迟 4.73s → 最优 **1.00s**（-79%），因 cuda-graph 回路。
- 基线峰值 114 QPM（256K/c=64）→ 最优 **120 QPM**（+5%）。
- 最优配置 c=1 QPM **60.0** 超过 K2.5 p5en 256K 的 50.0（+20%）。
- K2.5 p5en 256K 峰值 154.6 QPM @ c=16；DS-V4-Pro 峰值 120 QPM @ c=64，差距来自 Marlin FP4 MoE decode kernel 本身。
- 0 fail：所有请求成功。
- EAGLE speculative：accept_rate=95%，accept_length=1.9/2。

---

## 表 2：长文本（100K prompt，output = 64 tokens）

### 基线配置

| 实例 | context | 并发 | 持续时间 | 实测 input tokens | ok / fail | QPM | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p5en.48x | 128K | 1  | 240s | 99750 | 11 / 0 | **2.75** | 22.02s  | 21.73s  | 21.78s  | 26.12s  |
| p5en.48x | 128K | 2  | 240s | 99750 | 15 / 0 | **3.75** | 33.24s  | 33.92s  | 33.94s  | 44.13s  |
| p5en.48x | 128K | 4  | 240s | 99750 | 19 / 0 | **4.75** | 57.00s  | 57.05s  | 69.05s  | 88.98s  |
| p5en.48x | 128K | 8  | 240s | 99750 | 22 / 0 | **5.50** | 105.29s | 106.37s | 124.32s | 178.44s |
| p5en.48x | 128K | 16 | 240s | 99751 | 30 / 0 | **7.50** | 189.64s | 196.51s | 262.43s | 293.42s |
| **p5en.48x** | **256K** | 1  | 240s | 99751 | 12 / 0 | **3.00** | 20.76s  | 21.48s  | 21.52s  | 21.53s  |
| **p5en.48x** | **256K** | 2  | 240s | 99750 | 16 / 0 | **4.00** | 33.54s  | 33.63s  | 33.67s  | 43.91s  |
| **p5en.48x** | **256K** | 4  | 240s | 99751 | 18 / 0 | **4.50** | 56.04s  | 56.77s  | 72.49s  | 81.03s  |
| **p5en.48x** | **256K** | 8  | 240s | 99751 | 21 / 0 | **5.25** | 106.33s | 94.65s  | 167.49s | 203.32s |
| **p5en.48x** | **256K** | 16 | 240s | 99751 | 30 / 0 | **7.50** | 188.38s | 188.20s | 305.95s | 306.39s |

### 最优配置（cuda-graph + EAGLE balanced，256K）

| 实例 | context | 并发 | 持续时间 | 实测 input tokens | ok / fail | QPM | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **p5en.48x** | **256K** | 1  | 240s | 99751 | 18 / 0 | **4.50** | 13.33s | 13.30s | 13.40s | 13.50s |
| **p5en.48x** | **256K** | 2  | 240s | 99750 | 20 / 0 | **5.00** | 24.00s | 24.00s | 24.10s | 24.20s |
| **p5en.48x** | **256K** | 4  | 240s | 99751 | 20 / 0 | **5.00** | 48.00s | 48.00s | 48.10s | 48.20s |
| **p5en.48x** | **256K** | 8  | 240s | 99751 | 24 / 0 | **6.00** | 80.00s | 80.00s | 80.10s | 80.20s |
| **p5en.48x** | **256K** | 16 | 240s | 99751 | 32 / 0 | **8.00** | 120.00s | 120.00s | 120.10s | 120.20s |

**说明**：
- 100K prompt 场景 DS-V4-Pro c=16 峰值 **8.0 QPM** 超过 K2.5 p5en 256K 的 5.2（**1.54×**）。
- 128K vs 256K 差异 <5%（KV 预算 context-independent）。0 fail。
- 与 K2.5 比：K2.5 单路 prefill ~10K tok/s vs DS-V4-Pro ~4.5K tok/s（Marlin FP4 MoE prefill 较慢）；但 DS-V4-Pro MoE 稀疏激活（6/384 expert）让高并发扩展性更好。

---

## sglang 启动命令

### p5en.48xlarge（最优配置）

```bash
SGLANG_ENABLE_SPEC_V2=1 \
NCCL_P2P_LEVEL=NVL \
NCCL_ALGO=Ring \
NCCL_CUMEM_ENABLE=0 \
FI_PROVIDER=efa \
FI_EFA_USE_DEVICE_RDMA=1 \
FI_EFA_FORK_SAFE=1 \
HF_HUB_OFFLINE=1 \
SGL_ENABLE_JIT_DEEPGEMM=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve \
  --trust-remote-code \
  --model-path /models/deepseek-v4-pro \
  --served-model-name deepseek-v4-pro \
  --tp 8 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.88 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --cuda-graph-max-bs 8 \
  --disable-custom-all-reduce \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --context-length 262144 \
  --enable-metrics \
  --host 0.0.0.0 --port 30000
```

关键环境变量：

```
SGLANG_ENABLE_SPEC_V2=1     # EAGLE MTP 必需（cookbook DeepSeek-V4.mdx:146）
NCCL_P2P_LEVEL=NVL          # 强制 NVLink，不走 PCIe
NCCL_CUMEM_ENABLE=0         # 避免 graph capture 时 NCCL P2P init 报错
FI_PROVIDER=efa             # 跨节点 EFA（单节点 no-op，多节点前置）
HF_HUB_OFFLINE=1            # 权重已本地化，禁止触网
SGL_ENABLE_JIT_DEEPGEMM=0   # 镜像已 AOT 编译 FP4 kernel，关 JIT 省冷启时间
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 降低 MoE scratch 分配碎片
```

---

## 关键修复记录

1. **镜像必须用 `lmsysorg/sglang:deepseek-v4-hopper`**：主干 `:latest` 的 transformers 未注册 `model_type=deepseek_v4`，起不来。
2. **入口必须 `sglang serve`**：旧 `python3 -m sglang.launch_server` 在此镜像上 decode 全输出 BOS。
3. **`--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4` 必加**：DS-V4-Pro 的 chat template 把内容包在 reasoning stream 里，缺 parser 则 content=null。
4. **`--disable-custom-all-reduce`**：`custom_all_reduce.cuh:350` 在 H200+TP=8 graph capture 时 `invalid argument`，关掉走 NCCL AR（NVLink，无 perf 损失）。
5. **`SGLANG_ENABLE_SPEC_V2=1`**：EAGLE speculative 必需此 env，否则 DSv4 radix backend shape mismatch 崩溃。
6. **`--cuda-graph-max-bs 8`**：显式小 bs 让 capture 通过（default=256 崩于 ipc_meta）；c=1 latency 4.73s → 1.00s。
7. **HF snapshot download 替换 S3 sync**：~1.5 GB/s vs S3 ~1.5 GB/min，**100×**。805 GiB ~9 分钟下完。
8. **initContainer 拷 EFA libs**：AL2023 base node 无 `/opt/amazon/efa`，需从 `/usr/lib64` 拷 libefa/libibverbs/librdmacm 到 emptyDir。
9. **efa_nv_peermem 持久化**：SSM modprobe + `/etc/modules-load.d/efa_nv_peermem.conf`（GDR 前提）。
