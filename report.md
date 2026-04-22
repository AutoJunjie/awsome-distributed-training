# Kimi-K2.5 推理基准测试报告

## 测试环境

- **模型**：Kimi-K2.5，INT4 compressed-tensors（group_size=32），存于宿主机 NVMe `/opt/dlami/nvme/models/kimi-k2.5`
- **推理引擎**：sglang
- **并行度**：TP=8，PP=1
- **测试方法**：OpenAI 兼容 `/v1/chat/completions`，所有压测请求前缀加唯一 salt 以绕开 prefix cache，input token 数由 server usage 字段实测确认

| 实例 | GPU | GPU 间互连 | 位置 |
|---|---|---|---|
| `g7e.48xlarge` | 8 × NVIDIA RTX PRO 6000 Blackwell（96 GB GDDR7） | **PCIe only**（PIX/NODE/SYS） | 东京 ap-northeast-1 |
| `p5en.48xlarge` | 8 × NVIDIA H200（141 GB HBM3e） | **NVLink NV18 全互联**（18 bonded × 26.562 GB/s ≈ 478 GB/s 每向每对，NVSwitch full mesh） | us-east-2 |

### NVLink 证据（p5en.48x）

```
$ nvidia-smi topo -m
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0    X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
GPU1   NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
...（8×8 全 NV18 矩阵）
```

- `NCCL_P2P_LEVEL=NVL`（强制 NVLink，SYS 会降级走 PCIe）
- `NCCL_CUMEM_ENABLE=0`（CUDA graph capture 期间允许 NCCL P2P/IPC init）
- TP=8 EP=8，NCCL 2.28.3+cuda12.9 单节点内 AllReduce / AllToAll 全部走 NVSwitch P2P/IPC

### p5en 上下文尺寸

| context_length | max_total_num_tokens（KV 预算） | 并发能力 |
|---|---:|---:|
| 131072 (128K) | ~838K | max_running_requests=128 |
| **262144 (256K)** | **837,239** | max_running_requests=128 |

两档 KV 预算基本一致，所以 256K 没有牺牲并发上限。

---

## 表 1：5.8K prompt QPM 压测（output = 32 tokens，每档 60s）

| 实例 | context | 并发 | 实测 input tokens | ok / fail | **QPM** | QPS | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g7e.48x       | 128K | 1  | 5825 | 38 / 0  | **37.6**  | 0.63  | 1.60s  | 1.60s  | 1.60s  | 1.60s  |
| g7e.48x       | 128K | 4  | 5825 | 52 / 0  | **49.8**  | 0.83  | 4.82s  | 4.80s  | 4.80s  | 5.11s  |
| g7e.48x       | 128K | 8  | 5825 | 57 / 0  | **55.6**  | 0.93  | 8.50s  | 8.54s  | 8.81s  | 11.71s |
| g7e.48x       | 128K | 16 | 5825 | 62 / 0  | **57.0**  | 0.95  | 16.65s | 16.75s | 21.15s | 26.67s |
| g7e.48x       | 128K | 32 | 5825 | 59 / 0  | **54.3**  | 0.91  | 34.92s | 34.21s | 58.58s | 63.44s |
| g7e.48x       | 128K | 64 | 5825 | 83 / 0  | **54.1**  | 0.90  | 65.24s | 63.47s | 91.08s | 91.51s |
| p5en.48x      | 128K | 1  | 5825 | 50 / 0  | **49.1**  | 0.819 | 1.22s  | 1.23s  | 1.25s  | 1.39s  |
| p5en.48x      | 128K | 4  | 5825 | 108 / 0 | **106.5** | 1.776 | 2.25s  | 2.28s  | 2.30s  | 2.64s  |
| p5en.48x      | 128K | 8  | 5825 | 142 / 0 | **135.6** | 2.259 | 3.50s  | 3.53s  | 3.58s  | 3.65s  |
| p5en.48x      | 128K | 16 | 5825 | 161 / 0 | **138.2** | 2.304 | 6.55s  | 6.24s  | 6.61s  | 12.06s |
| p5en.48x      | 128K | 32 | 5825 | 138 / 0 | **101.4** | 1.690 | 16.99s | 13.20s | 32.02s | 36.76s |
| p5en.48x      | 128K | 64 | 5825 | 166 / 0 | **116.1** | 1.936 | 32.12s | 32.50s | 50.58s | 51.01s |
| **p5en.48x**  | **256K** | 1  | 5825 | 50 / 0  | **50.0**  | 0.833 | 1.20s  | 1.21s  | 1.23s  | 1.60s  |
| **p5en.48x**  | **256K** | 4  | 5825 | 108 / 0 | **107.7** | 1.795 | 2.23s  | 2.26s  | 2.29s  | 2.46s  |
| **p5en.48x**  | **256K** | 8  | 5825 | 144 / 0 | **136.2** | 2.270 | 3.52s  | 3.52s  | 3.55s  | 3.64s  |
| **p5en.48x**  | **256K** | 16 | 5825 | 160 / 0 | **154.6** | 2.576 | 6.20s  | 6.21s  | 6.22s  | 6.30s  |
| **p5en.48x**  | **256K** | 32 | 5825 | 160 / 0 | **145.5** | 2.425 | 13.13s | 13.08s | 15.68s | 20.73s |
| **p5en.48x**  | **256K** | 64 | 5825 | 165 / 0 | **142.2** | 2.370 | 26.19s | 26.71s | 37.31s | 43.83s |

**说明**：
- g7e.48x 峰值 QPM 约 **57**，硬饱和于 prefill 吞吐 ~5800 tok/s。
- p5en.48x 峰值 QPM 约 **155**（256K context, 并发=16），相对 g7e **≈ 2.7×**。
- 把 context 从 128K 提到 256K，**不仅没有 regression，反而 @ 高并发（16/32/64）表现更稳**（p50/p90 延迟更平、吞吐不再塌陷）。原因：--context-length 只改每请求上限，不改 KV 预算；但 256K 这次跑刚好是在 --disable-custom-all-reduce + `NCCL_CUMEM_ENABLE=0` 的稳定镜像上。
- 0 fail：**所有 374+ 请求** 成功。

---

## 表 2：长文本（100K prompt，output = 64 tokens）

| 实例 | context | 并发 | 持续时间 | 实测 input tokens | ok / fail | QPM | 平均延迟 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g7e.48x      | 128K | 1  | 150s | 99745 | 7 / 0  | **2.4** | 24.74s  | 24.69s  | 24.94s  | 24.94s  |
| g7e.48x      | 128K | 2  | 150s | 99745 | 8 / 0  | **2.5** | 48.43s  | 48.38s  | 49.51s  | 49.51s  |
| g7e.48x      | 128K | 4  | 150s | 99746 | 9 / 0  | **2.4** | 89.09s  | 98.00s  | 119.41s | 119.41s |
| g7e.48x      | 128K | 8  | 240s | 99746 | 16 / 0 | **2.4** | 163.66s | 196.20s | 196.48s | 218.24s |
| g7e.48x      | 128K | 16 | 240s | 99746 | 24 / 0 | **2.4** | 284.11s | 342.84s | 392.74s | 414.05s |
| p5en.48x     | 128K | 1  | 240s | 99746 | 23 / 0 | **5.6** | 10.63s  | 10.62s  | 10.64s  | 10.79s  |
| p5en.48x     | 128K | 2  | 240s | 99746 | 26 / 0 | **6.2** | 19.36s  | 19.35s  | 19.38s  | 19.50s  |
| p5en.48x     | 128K | 4  | 240s | 99746 | 28 / 0 | **6.1** | 39.40s  | 39.38s  | 39.54s  | 39.85s  |
| p5en.48x     | 128K | 8  | 240s | 99746 | 24 / 0 | **5.2** | 90.59s  | 95.16s  | 120.73s | 144.48s |
| p5en.48x     | 128K | 16 | 240s | 99747 | 32 / 0 | **5.2** | 165.17s | 190.49s | 204.09s | 239.90s |
| **p5en.48x** | **256K** | 1  | 240s | 99746 | 23 / 0 | **5.7** | 10.60s  | 10.62s  | 10.64s  | 10.65s  |
| **p5en.48x** | **256K** | 2  | 240s | 99746 | 26 / 0 | **6.2** | 19.34s  | 19.35s  | 19.38s  | 19.51s  |
| **p5en.48x** | **256K** | 4  | 240s | 99746 | 28 / 0 | **6.1** | 39.34s  | 39.38s  | 39.44s  | 39.76s  |
| **p5en.48x** | **256K** | 8  | 240s | 99746 | 24 / 0 | **5.2** | 90.58s  | 95.17s  | 120.71s | 144.48s |
| **p5en.48x** | **256K** | 16 | 240s | 99746 | 32 / 0 | **5.2** | 164.89s | 190.23s | 203.73s | 239.41s |

**说明**：
- g7e.48x 100K prompt 场景 QPM 在 ~2.4–2.5（prefill 硬饱和）。
- p5en.48x 100K prompt 场景 QPM 单路 5.6–5.7，并发 2 峰值 6.2，超过并发 2 延迟直接线性翻倍 → **prefill 吞吐 bound**（100K tokens @ ~10s → **~10K tok/s prefill**，约 g7e 的 ~1.7×）。
- 128K vs 256K context 在这个负载下完全一致（差异 < 0.5%），0 fail。证明扩 context 不会偷走 KV 预算。
- 总体 p5en / g7e 比约 **≈ 2.3× ~ 2.5×** 在 100K 场景。

---

## sglang 启动命令

### g7e.48xlarge

```bash
HF_HUB_OFFLINE=1 \
NCCL_P2P_LEVEL=4 \
NCCL_DEBUG=WARN \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGL_ENABLE_JIT_DEEPGEMM=0 \
python -m sglang.launch_server \
  --model /models/kimi-k2.5 \
  --tp 8 \
  --host 0.0.0.0 --port 5000 \
  --mem-fraction-static 0.94 \
  --enable-metrics \
  --sleep-on-idle \
  --attention-backend flashinfer \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --served-model-name kimi-k2.5 \
  --chunked-prefill-size 8092 \
  --cuda-graph-max-bs 16 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code \
  --enable-mixed-chunk
```

### p5en.48xlarge

K8s Deployment（单节点 8×H200，EFA + NVLink）：`/home/ubuntu/sglang-kimi-k25.yaml`

关键参数：

```bash
python3 -m sglang.launch_server \
  --model-path /models/kimi-k2.5 \
  --served-model-name kimi-k2.5 \
  --tp 8 --ep-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 \
  --attention-backend flashinfer \
  --context-length 262144 \               # 256K（128K 下也验证过）
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 8192 \
  --max-running-requests 128 \
  --cuda-graph-max-bs 16 \
  --enable-mixed-chunk \
  --enable-metrics \
  --sleep-on-idle \
  --disable-custom-all-reduce \           # sgl custom AR 在 H200 capture 时崩，fallback NCCL AR
  --host 0.0.0.0 --port 30000
```

关键环境变量（NVLink 必需）：

```
NCCL_P2P_LEVEL=NVL          # 关键：强制 NVLink，不要用 SYS（会降级 PCIe）
NCCL_ALGO=Ring
NCCL_CUMEM_ENABLE=0         # 关键：避免 "operation not permitted when stream is capturing"
FI_PROVIDER=efa             # EFA（跨节点才用得上，本单节点其实不走 EFA）
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_FORK_SAFE=1
HF_HUB_OFFLINE=1
SGL_ENABLE_JIT_DEEPGEMM=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LD_LIBRARY_PATH=/opt/efa-libs:...       # initContainer 把 host /usr/lib64 的 libefa/libibverbs/librdmacm 拷到 emptyDir
```

---

## 关键修复记录

1. **HF snapshot download 替换 S3 sync**：S3 跨区 sync ~1.5 GB/min，HF `snapshot_download`（`HF_HUB_ENABLE_HF_TRANSFER=1` + `max_workers=16`）可达 ~1.5 GB/s，**~100×**。555 GB 5 分钟下完。
2. **NCCL_P2P_LEVEL=SYS → NVL**：H200 有 NVSwitch，强制 `NVL` 防止 NCCL 选 PCIe 路径。
3. **NCCL_CUMEM_ENABLE=0**：否则 CUDA graph capture 时 NCCL P2P/IPC init 会 `operation not permitted when stream is capturing`。
4. **`--disable-custom-all-reduce`**：sglang 的 custom_all_reduce 在 H200 + TP=8 capture 路径上崩 `invalid argument`，fallback NCCL AllReduce（走 NVLink，无 perf 损失）。
5. **initContainer 拷 EFA libs**：base AL2023 node 只在 `/usr/lib64` 有 rdma-core 61，没有 `/opt/amazon/efa`。
6. **efa_nv_peermem**：SSM modprobe + 写 `/etc/modules-load.d/efa_nv_peermem.conf` 持久化（GDR 前提）。
