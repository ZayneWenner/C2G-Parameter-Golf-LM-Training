# C2G 参数高尔夫 —— 缺口补齐清单（v2）

> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 关联：上一轮交付包位于桌面 `C2G参数高尔夫_作业交付/`

---

## 1. 上一轮被指出的缺口

| 缺口 | 具体问题 | 风险 |
|------|---------|------|
| **无真实训练运行** | 本机 `ready_for_training: false`，`submission.json` 中 `val_bpb / bytes_total / seeds` 全为 null | `scoreAchievement` 维度无法计分 |
| **虚假硬件申报** | `hardware: "8xH100 80GB SXM (planned)"` 未真实运行即填写 8xH100 | 被判定为虚假申报，可能 DQ |
| **未标注 non-record** | 未说明走 Non-record 通道的可能性 | 评审/提交入口可能直接拒绝 |
| **无自动采集脚本** | 需要手动解析日志、回填 JSON、打包 tar.gz | 易出错、不可复现 |

---

## 2. 本 v2 补齐内容

### 2.1 删除虚假硬件申报

`JiaJingwen_C2G_submission.json` 已更新：
- `hardware` 改为 `TBD — filled after actual single-GPU run (e.g. 1xRTX 4090 24GB, non-record)`
- `pytorch_version` 改为 `TBD — filled after actual run`
- 新增 `non_record: true`，`track` 默认 `non_record_16mb`
- 原 `_note` 模板大段说明已删除，改为字段内诚实标注

### 2.2 新增一键训练-采集-提交流水线

新增 `JiaJingwen_C2G_run_experiments.py`：
- **环境检查**：`nvidia-smi`、`torch.cuda.is_available()`、GPU 型号/显存
- **数据准备**：自动 clone 官方仓库，调用 `data/cached_challenge_fineweb.py` 下载 SP1024 与 SP8192
- **训练执行**：
  - E001 baseline（SP1024，3 seeds）
  - E003 SP8192 单点升级（3 seeds）
  - 单卡 `python` 直接运行，无需 `torchrun`
  - `MAX_WALLCLOCK_SECONDS=0`，non-record 不限 10 分钟
- **日志解析**：自动提取 `final_int8_zlib_roundtrip_exact` 的 `val_bpb`、`val_loss`、`bytes_total`、`bytes_code`
- **生成 `metrics.json`**：记录每次 seed 的完整结果与统计量
- **回填 `submission.json`**：真实 hardware / pytorch_version / val_bpb / val_bpb_std / bytes_total / bytes_code / seeds
- **打包 `submission.tar.gz`**：包含 `train_gpt.py`、`submission.json`、`README.md`、`metrics.json`、全部日志

### 2.3 README 更新

`JiaJingwen_C2G_README.md` 已重写：
- 文件清单加入 `run_experiments.py` 与本清单
- 复现步骤改为**推荐 RTX 4090 单卡 Non-record 路径**
- 说明：脚本默认 non-record，若改 record 通道需手动改 `track` 与 wallclock

---

## 3. 仍必须由你手动执行的 3 步

本地环境确实无 GPU，以下无法由 AI 代劳：

1. **租一张单卡 GPU**（推荐 RTX 4090 24GB，RunPod / Vast / AutoDL，约 ¥10–40）。
2. **把桌面 `C2G参数高尔夫_作业交付/` 整个目录上传到该机器**，执行：
   ```bash
   pip install torch sentencepiece numpy tqdm flash-attn
   python JiaJingwen_C2G_run_experiments.py
   ```
3. **下载产物**（`metrics.json`、`submission.json`、`submission_*.tar.gz`）并提交到挑战入口。

---

## 4. 诚实边界

- **未伪造 BPB**：`submission.json` 中的 `val_bpb` 仍待脚本真实跑完后回填；当前为 `null`。
- **未伪造硬件**：`hardware` 字段已改为 TBD，绝不再出现未运行即填 8xH100 的情况。
- **Non-record 合规**：官方明确接受 non-record 提交；脚本会把 `non_record: true` 写进 JSON。
- **可复现**：所有命令、环境变量、seed、数据下载方式均写入脚本与 README，任何人拿到目录即可复现。

---

## 5. 提交前自检

运行脚本后，确认 `submission.json` 满足：

```json
{
  "track": "non_record_16mb",
  "non_record": true,
  "val_bpb": <真实数字>,
  "val_bpb_std": <真实数字>,
  "bytes_total": <<=16000000>,
  "bytes_code": <真实数字>,
  "seeds": [1337, 1338, 1339],
  "hardware": "NVIDIA GeForce RTX 4090 24564MiB (non-record single GPU)",
  "pytorch_version": "2.x.x+cu121",
  "compliance": {
    "artifact_under_16mb": true,
    "three_seeds": true
  }
}
```

---

*补齐工作由 AI（WorkBuddy）在本轮完成；真实训练运行需用户在有 GPU 的环境执行。*
