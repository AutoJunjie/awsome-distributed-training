---
name: eks-p5en-cr
description: Provision p5en.48xlarge nodes from a Capacity Reservation into the p5-training-eks EKS cluster (us-east-2), deploy GPU pods, and run multi-node workloads. Use when user asks to pull up / scale up / 拉起 / 部署 p5en nodes, attach a new CR, deploy training pods, run NCCL tests, or when `InsufficientInstanceCapacity` / `Capacity Reservation ... is not active` errors appear on this cluster.
---

# eks-p5en-cr — Skill

End-to-end runbook for bringing up `p5en.48xlarge` nodes from a Capacity Reservation into the `p5-training-eks` EKS cluster. Covers the common "new CR arrived → update LT → scale ASG → verify" flow, plus 5 known traps.

Target environment (hardcoded):
- Region: `us-east-2`
- Account: `955643200499`
- Cluster: `p5-training-eks`
- ASG: `p5-training-eks-p5-asg`
- Launch Template: `lt-0729374d13dad9998`
- Subnet: `subnet-05be9447665b2d50b` (us-east-2a)

If any of these differ (different region, different cluster), STOP and confirm with the user before proceeding.

---

## When to trigger this skill

- "拉起 / 部署 / 开 p5en 节点" / "bring up p5en" / "scale up p5en"
- 新 CR 到货，需要切换
- ASG scaling activities 报 `InsufficientInstanceCapacity`, `Capacity Reservation ... is not active`, `ReservationCapacityExceeded`
- `kubectl get nodes` 少节点，怀疑 CR / LT / ASG 问题
- Pod 因为 `nvidia.com/gpu` 或 `vpc.amazonaws.com/efa` 资源不足 Pending

Do NOT trigger for: SageMaker HyperPod clusters, p5 (non-en) instances, other regions, other accounts.

---

## Three questions to answer before any action

1. **What type is the CR?** `capacity-block` vs `default` — behaviors differ.
2. **Does the CR have a PlacementGroup?** `PlacementGroupArn = null` vs a specific ARN — LT must match.
3. **Is the CR past its `StartDate`?** A `capacity-block` before StartDate shows `active` + `Avail=N/N` but is **not consumable**.

If any of these are unknown, run §"Diagnose" first. Do not skip.

---

## Standard flow (7 steps)

```bash
NEW_CR=cr-xxxxxxxxxxxxxxxxx     # user's new CR id
LT_ID=lt-0729374d13dad9998
ASG=p5-training-eks-p5-asg
REGION=us-east-2
```

### Step 1 — Gather CR facts

```bash
aws ec2 describe-capacity-reservations --region $REGION \
  --capacity-reservation-ids $NEW_CR \
  --query "CapacityReservations[0].{Type:ReservationType,State:State,Match:InstanceMatchCriteria,PG:PlacementGroupArn,AZ:AvailabilityZone,Start:StartDate,End:EndDate,Avail:AvailableInstanceCount,Total:TotalInstanceCount}"
```

Record: `Type`, `PG`, `Start`, `AZ`. AZ must be `us-east-2a` (ASG subnet is pinned there).

### Step 2 — Gather current LT facts

```bash
aws ec2 describe-launch-template-versions --region $REGION \
  --launch-template-id $LT_ID --versions '$Default' \
  --query "LaunchTemplateVersions[0].{V:VersionNumber,CR:LaunchTemplateData.CapacityReservationSpecification,PG:LaunchTemplateData.Placement}"
```

Record: current default `VersionNumber` (the source-version for step 4), current CR id, current Placement.

### Step 3 — Align CR ↔ LT Placement Group

| CR `PlacementGroupArn` | LT `Placement.GroupName` |
|---|---|
| `null` | MUST be absent: use `"Placement":{}` (or preserve non-GroupName fields only) |
| `arn:...:placement-group/<name>` | MUST be `"GroupName":"<name>"` |

**Mismatch → targeted CR matching fails → EC2 silently falls through to on-demand → `InsufficientInstanceCapacity`.** CR looks healthy (`Avail=N/N`, `CapacityAllocations=[]`) but nothing is consumed.

### Step 4 — Create new LT version

Case A — LT has PG, CR does not (most common for capacity-block):
```bash
aws ec2 create-launch-template-version --region $REGION \
  --launch-template-id $LT_ID --source-version <V-from-step-2> \
  --version-description "CR $NEW_CR, drop PG (CR PG=null), AMI v20260423" \
  --launch-template-data "{
    \"ImageId\":\"ami-060e22c5411b4e739\",
    \"CapacityReservationSpecification\":{\"CapacityReservationTarget\":{\"CapacityReservationId\":\"$NEW_CR\"}},
    \"Placement\":{}
  }"
```

Case B — CR and LT both carry the same PG:
```bash
aws ec2 create-launch-template-version --region $REGION \
  --launch-template-id $LT_ID --source-version <V-from-step-2> \
  --version-description "CR $NEW_CR, AMI v20260423" \
  --launch-template-data "{
    \"ImageId\":\"ami-060e22c5411b4e739\",
    \"CapacityReservationSpecification\":{\"CapacityReservationTarget\":{\"CapacityReservationId\":\"$NEW_CR\"}}
  }"
```

**AMI**: `ami-060e22c5411b4e739` = `amazon-eks-node-al2023-x86_64-nvidia-1.33-v20260423` (k8s 1.33.11, 2026-04-24 build). Always use the latest 1.33 NVIDIA AMI — check with:
```bash
aws ec2 describe-images --region us-east-2 --owners amazon \
  --filters "Name=name,Values=amazon-eks-node-al2023-x86_64-nvidia-1.33-*" \
  --query "Images | sort_by(@, &CreationDate) | [-1].[Name,ImageId]" --output text
```

**Gotcha**: `"Placement":{}` wipes the entire Placement block. If step 2 showed fields other than GroupName in Placement (e.g. `Tenancy`, `AvailabilityZone`), preserve those. Check first:
```bash
aws ec2 describe-launch-template-versions --region $REGION \
  --launch-template-id $LT_ID --versions <V-from-step-2> \
  --query "LaunchTemplateVersions[0].LaunchTemplateData.Placement"
```

Record the returned `VersionNumber` as `NEW_V`.

### Step 5 — Set new version as default + verify

```bash
aws ec2 modify-launch-template --region $REGION \
  --launch-template-id $LT_ID --default-version $NEW_V

aws ec2 describe-launch-template-versions --region $REGION \
  --launch-template-id $LT_ID --versions '$Default' \
  --query "LaunchTemplateVersions[0].{V:VersionNumber,CR:LaunchTemplateData.CapacityReservationSpecification,PG:LaunchTemplateData.Placement}"
```

Verify: `CR.CapacityReservationTarget.CapacityReservationId == $NEW_CR` AND `PG` matches CR PG.

### Step 6 — Scale ASG

```bash
DESIRED=4     # match CR TotalInstanceCount

aws autoscaling update-auto-scaling-group --region $REGION \
  --auto-scaling-group-name $ASG \
  --min-size 0 --desired-capacity $DESIRED --max-size $DESIRED
```

Use `update-auto-scaling-group`, NOT `set-desired-capacity` — the latter errors if desired exceeds current max.

### Step 7 — Observe

Poll every 60–90s:
```bash
aws autoscaling describe-scaling-activities --region $REGION \
  --auto-scaling-group-name $ASG --max-records 5 \
  --query "Activities[].{T:StartTime,S:StatusCode,Desc:Description,Reason:StatusMessage}" --output table

aws ec2 describe-capacity-reservations --region $REGION \
  --capacity-reservation-ids $NEW_CR \
  --query "CapacityReservations[0].{Avail:AvailableInstanceCount,Total:TotalInstanceCount}"

kubectl get nodes -o wide
```

Normal timeline: RunInstances OK in 30s-2min → instance Running in +1min → nodeadm+kubelet+vpc-cni Ready in +4-7min total.

If still not Ready after 10 min, go to §"Diagnose".

---

## Diagnose (when things don't come up)

One-liner — run this first, compare outputs against the table below:

```bash
NEW_CR=<cr-id>; LT_ID=lt-0729374d13dad9998; REGION=us-east-2
echo "=== CR ==="
aws ec2 describe-capacity-reservations --region $REGION \
  --capacity-reservation-ids $NEW_CR \
  --query "CapacityReservations[0].{Type:ReservationType,State:State,Match:InstanceMatchCriteria,PG:PlacementGroupArn,AZ:AvailabilityZone,Start:StartDate,Avail:AvailableInstanceCount,Total:TotalInstanceCount,Alloc:CapacityAllocations}" --output json
echo "=== LT default ==="
aws ec2 describe-launch-template-versions --region $REGION \
  --launch-template-id $LT_ID --versions '$Default' \
  --query "LaunchTemplateVersions[0].{V:VersionNumber,CR:LaunchTemplateData.CapacityReservationSpecification,PG:LaunchTemplateData.Placement,IT:LaunchTemplateData.InstanceType}" --output json
echo "=== ASG recent ==="
aws autoscaling describe-scaling-activities --region $REGION \
  --auto-scaling-group-name p5-training-eks-p5-asg --max-records 3 \
  --query "Activities[].{T:StartTime,S:StatusCode,Reason:StatusMessage}" --output table
echo "=== date UTC ==="; date -u
```

| Observation | Root cause | Fix |
|---|---|---|
| CR `Avail=N`, `Alloc=[]`, ASG activity = `InsufficientInstanceCapacity` | **targeted match fails — LT PG ≠ CR PG** (trap 2, most common) | align PG, new LT version, set default, retry |
| ASG activity = `Capacity Reservation ... is not active` | LT still pinned to an expired CR (trap 1) | step 4/5 — push new LT version with correct CR id |
| `date -u` < CR `StartDate` | CR not yet usable (trap 3) | wait until Start time |
| RunInstances OK but node NotReady; console log shows `failed to ensure primary ENI only configuration` | someone bypassed ASG with `run-instances` and gave all 16 ENIs a subnet (trap 4) | don't bypass ASG; if forced, secondary ENIs must be `InterfaceType=efa-only` |
| `terminate-instances` recent, new launch returns `ReservationCapacityExceeded` | CR hasn't finished reclaiming (trap 5) | wait 5-15 min; verify with `describe-instances --filters capacity-reservation-id` that old instance is actually `terminated` |

---

## Five known traps (concise)

### Trap 1 — Stale CR in LT
Old LT version still points at an expired CR.
**Symptom**: ASG activity `Capacity Reservation ... is not active`.
**Fix**: new LT version with new CR id, set default.

### Trap 2 — ⭐ PG mismatch (most common, most misleading)
`capacity-block` + `InstanceMatchCriteria=targeted` requires bit-for-bit equality on every dimension, PG included.
- CR `PG=null` + LT `GroupName="foo"` → mismatch
- Both `null` → match
- Both the same ARN → match
**Symptom**: CR looks healthy (`Avail=N/N`, `Alloc=[]`), error is `InsufficientInstanceCapacity`. EC2 silently falls through to on-demand pool, which is typically empty for p5en in us-east-2a.
**Fix**: align in LT (Step 3/4). `"Placement":{}` to drop GroupName; keep other Placement fields.
**Why misleading**: CR displays as active, error mentions AWS capacity, same LT may have worked with a different CR.

### Trap 3 — CR before StartDate
`capacity-block` `StartDate` in the future: CR shows `State=active` + `Avail=N/N` but not consumable.
**Symptom**: `InsufficientInstanceCapacity` despite CR looking healthy.
**Fix**: `date -u` vs StartDate. Wait.

### Trap 4 — Bypassing ASG with run-instances
LT has 16 ENIs without SubnetId. ASG mode: AWS auto-injects subnet to primary (DeviceIndex=0) only. run-instances mode: API requires SubnetId on every ENI → adding subnet to all 16 → systemd-networkd takes over all 16 → nodeadm's "primary ENI only" check times out at 60s → node never joins.
**Fix**: always use ASG for this cluster. If absolutely must run-instances: secondary 15 ENIs MUST be `InterfaceType=efa-only` (no IP, only EFA device).

### Trap 5 — CR reclaim delay
After `terminate-instances`, capacity-block takes 5-15 min to go `shutting-down` → `terminated` → `Avail` returns.
**Symptom**: `ReservationCapacityExceeded` on retry.
**Fix**: don't retry immediately. Verify old instance is fully `terminated` before launching.

---

## Pod-side sanity after nodes are Ready

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=p5en.48xlarge -o wide
kubectl describe node <node-name> | grep -E "vpc.amazonaws.com/efa|nvidia.com/gpu"
# expect: vpc.amazonaws.com/efa: 16, nvidia.com/gpu: 8
```

Pod spec MUST include (p5en-specific):
```yaml
resources:
  limits:
    vpc.amazonaws.com/efa: "16"   # p5en is 16, NOT p5's 32
    nvidia.com/gpu: "8"
hostNetwork: true
hostIPC: true
securityContext:
  privileged: true
```

In-pod EFA verification:
```bash
kubectl exec -it <pod> -- fi_info -p efa | grep -c "domain name"
# expect: 16
```

---

## Teardown

```bash
kubectl delete -f /mnt/s3files/uccl-ep-mfu/pplx-workers-p5en.yaml

aws autoscaling update-auto-scaling-group --region us-east-2 \
  --auto-scaling-group-name p5-training-eks-p5-asg \
  --min-size 0 --desired-capacity 0 --max-size 4

# After 5-15 min, verify Avail returns to Total
aws ec2 describe-capacity-reservations --region us-east-2 \
  --capacity-reservation-ids <CR> \
  --query "CapacityReservations[0].{Avail:AvailableInstanceCount,Total:TotalInstanceCount}"
```

---

## Risky actions — confirm before running

These are destructive or hard to undo on shared infra. Ask the user first unless already authorized for this session:

- `update-auto-scaling-group` with `desired-capacity=0` (will terminate all running nodes)
- `modify-launch-template --default-version` (changes what future ASG launches use — low risk, reversible)
- `delete-launch-template-versions` (don't do this; keep history for rollback)
- Any `run-instances` / `terminate-instances` (bypasses ASG — see trap 4)

Safe to run without asking: all `describe-*` reads, `create-launch-template-version` (just adds a new version, doesn't make it default).

---

## Post-node-ready: fabricmanager check (may be fixed in v20260423)

AMI `v20260409` had a first-boot race: `nvidia-fabricmanager.service` failed on ~75% of nodes → pods get `cuInit()` error 802. AMI `v20260423` may have fixed this, but **always verify** after nodes come up:

```bash
INSTANCE_IDS=$(aws ec2 describe-instances --region us-east-2 \
  --filters "Name=tag:aws:autoscaling:groupName,Values=p5-training-eks-p5-asg" \
    "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].InstanceId" --output text)

# Check status first
aws ssm send-command --region us-east-2 --instance-ids $INSTANCE_IDS \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl is-active nvidia-fabricmanager"]'

# If any node reports "failed" or "inactive", restart:
aws ssm send-command --region us-east-2 --instance-ids $INSTANCE_IDS \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl restart nvidia-fabricmanager.service && systemctl is-active nvidia-fabricmanager"]'
```

Verify via SSM output: should print `active`. If all nodes pass on first check without restart, the v20260423 AMI has fixed the race — update this section accordingly.

---

## Deploy GPU pods

### Pod spec template (4-node example)

Key requirements for p5en pods on this cluster:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: workload-node0
  labels:
    app: workload
spec:
  restartPolicy: Never
  hostNetwork: true       # required for EFA multi-NIC
  hostIPC: true           # required for NCCL shared memory
  nodeName: <node-name>   # pin to specific node
  tolerations:
    - effect: NoSchedule
      key: nvidia.com/gpu
      operator: Exists
    - effect: NoSchedule
      key: vpc.amazonaws.com/efa
      operator: Exists
  containers:
    - name: main
      image: <image>
      command: ["sleep", "infinity"]
      securityContext:
        privileged: true
        capabilities:
          add: ["IPC_LOCK", "SYS_PTRACE"]
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: FI_PROVIDER
          value: "efa"
        - name: FI_EFA_USE_DEVICE_RDMA
          value: "1"
      resources:
        limits:
          cpu: "32"
          memory: 64Gi
          nvidia.com/gpu: "8"
          vpc.amazonaws.com/efa: "16"   # p5en = 16 EFA NICs (NOT 32 like p5)
        requests:
          cpu: "32"
          memory: 64Gi
          nvidia.com/gpu: "8"
          vpc.amazonaws.com/efa: "16"
      volumeMounts:
        - mountPath: /dev/shm
          name: shm
        - mountPath: /mnt/s3files
          name: s3files
  volumes:
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 200Gi    # NCCL needs large shm
    - name: s3files
      persistentVolumeClaim:
        claimName: fsx-pvc
```

### Image choices

| Image | Use case | Notes |
|---|---|---|
| `public.ecr.aws/hpc-cloud/nccl-tests:latest` | NCCL smoke tests | NCCL 2.27.7, nccl-tests 自带, 需 `LD_PRELOAD=/opt/nccl/build/lib/libnccl.so` |
| `763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` | PyTorch training | NCCL 2.28.9, CUDA 13.0, PyTorch 2.10, sshd 已装, 不含 nccl-tests (需现编 ~1min). ~9.2 GB, 同 digest 在 us-west-2 也有 |

### Deploy:

```bash
kubectl --context arn:aws:eks:us-east-2:955643200499:cluster/p5-training-eks \
  apply -f <your-pods.yaml>

# Verify all pods Running
kubectl get pods -l app=workload -o wide
```

---

## Setup inter-pod SSH (for mpirun)

`hostNetwork=true` 意味着 pod 直接用 node IP，sshd 必须用非 22 端口（宿主 sshd 占了 22）。

```bash
# 在每个 pod 内执行（或通过 kubectl exec 批量）:
# 1. 生成 / 复制共享 SSH 密钥
mkdir -p /root/.ssh && chmod 700 /root/.ssh
# (通过 kubectl cp 分发同一份 ed25519 keypair 到所有 pod)

# 2. 配置
cat > /root/.ssh/config <<'EOF'
Host *
  Port 2222
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
EOF
chmod 600 /root/.ssh/config

# 3. 启动 sshd (DLC 镜像已含; hpc-cloud 镜像已含)
/usr/sbin/sshd -p 2222

# 4. 验证互通
ssh root@<other-pod-ip> hostname
```

批量分发 SSH 密钥:
```bash
# 在 bastion 上生成密钥对
ssh-keygen -t ed25519 -f /tmp/nccl_key -N ""
# 打包
tar czf /tmp/ssh-keys.tar.gz -C /tmp nccl_key nccl_key.pub

# 分发到每个 pod
for pod in workload-node0 workload-node1 workload-node2 workload-node3; do
  kubectl cp /tmp/ssh-keys.tar.gz $pod:/tmp/ssh-keys.tar.gz
  kubectl exec $pod -- bash -c '
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    tar xzf /tmp/ssh-keys.tar.gz -C /root/.ssh/
    cp /root/.ssh/nccl_key /root/.ssh/id_ed25519
    cp /root/.ssh/nccl_key.pub /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/id_ed25519 /root/.ssh/authorized_keys
    /usr/sbin/sshd -p 2222
  '
done
```

---

## Run multi-node NCCL / training workload

### NCCL environment variables (validated on this cluster)

```bash
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

### mpirun command (4-node 32-GPU all_reduce example)

```bash
/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output \
  -mca plm_rsh_args "-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -np 32 -npernode 8 --bind-to none \
  --host <ip1>:8,<ip2>:8,<ip3>:8,<ip4>:8 \
  -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
  -x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 \
  -x NCCL_SOCKET_IFNAME=^docker,lo,veth \
  /opt/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 -w 5 -n 50
```

### Expected results (p5en 4-node)

| Metric | Expected |
|---|---|
| 8 GiB busbw | ~369 GB/s |
| GDR active | `GPU Direct RDMA Enabled` in logs, 0 fallback messages |
| EFA NICs per rank | 16 (`fi_info -p efa` shows 16 unique rdmap NICs) |

### Compile nccl-tests in DLC image (if needed)

```bash
cp -r /mnt/s3files/p5en-dlc-nccl-2026-04-25/bin/nccl-tests-source /tmp/nccl-tests
cd /tmp/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local/cuda CUDA_HOME=/usr/local/cuda -j16
# ~1 min, output: /tmp/nccl-tests/build/*_perf
```

---

## Full end-to-end checklist (from new CR to running workload)

1. [ ] Confirm CR: `describe-capacity-reservations` — check State, PG, Start, AZ
2. [ ] Create LT version with correct CR id + aligned Placement
3. [ ] Set LT default version
4. [ ] Scale ASG to desired count
5. [ ] Wait for nodes Ready (`kubectl get nodes`)
6. [ ] **Restart nvidia-fabricmanager** on all nodes via SSM
7. [ ] Apply pod YAML (hostNetwork, privileged, EFA 16, GPU 8, shm 200Gi)
8. [ ] Wait for pods Running
9. [ ] Distribute SSH keys + start sshd on port 2222
10. [ ] Run `mpirun` workload from node0 pod

---

## Filesystem caveat

| Location | Filesystem | Access |
|---|---|---|
| Bastion `/mnt/s3files` | EFS | Bastion 可读写 |
| Pod `/mnt/s3files` | FSx Lustre (PVC `fsx-pvc`) | Pod 可读写 |

这两个 **不是同一个文件系统**。跨侧传文件用 `kubectl cp`。

---

## Related files

- `/mnt/s3files/p5en-nccl-gdr-2026-04-25/handoff.md` — 完整栈版本记录
- `/mnt/s3files/p5en-dlc-nccl-2026-04-25/handoff-dlc.md` — DLC 镜像栈对比
- `/mnt/s3files/p5en-nccl-gdr-2026-04-25/yaml/nccl-gdr-4pods.yaml` — 4-pod YAML 模板
- `/mnt/s3files/p5-efa-diagnostics-2026-04-17.md` — EFA/GDR 诊断笔记
- `/mnt/s3files/uccl-ep-mfu/` — UCCL-EP / Megatron 实验 handoff
