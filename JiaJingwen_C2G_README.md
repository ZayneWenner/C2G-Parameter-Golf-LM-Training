# C2G 参数高尔夫 —— 作业交付包（JiaJingwen）

> 作者：贾静文（JiaJingwen）｜挑战 ID：ch-20260717031359-b8wyg0｜截止：2026-12-31
> 交付日期：2026-10-07（v2：补齐训练运行缺口）

---

## 0. 重要说明（请先读）

本挑战的 **35%–25% 分数来自真实 BPB 成绩**，那需要 **GPU + 下载 FineWeb + 真实训练运行**。当前本地工作环境**无 GPU、无法联网下载数据集、无法提交榜单**，因此：

- ✅ **已真实完成**：评审 5 维度中 4 个文档类维度（方法学 / 产物完整性 / AI 使用 / 复盘）的全部交付物，均基于**官方公开真实数据**写成，未编造任何数字。
- ✅ **已补齐的运行缺口**：
  - 删除 `submission.json` 中虚假的 `8xH100 80GB SXM (planned)` 硬件申报；
  - 新增 `JiaJingwen_C2G_run_experiments.py`，可在任意 CUDA 机器上一键完成环境检查 → 数据下载 → 3-seed 训练 → 日志解析 → 回填 `submission.json` → 打包 `submission.tar.gz`。
- ⏳ **仍需你手动执行一次**：在租用的 GPU 机器上跑 `python JiaJingwen_C2G_run_experiments.py`（约 ¥10–40 的 RTX 4090 单卡即可），回填真实 BPB。

**推荐路径：Non-record 单卡提交**

官方 README 明确接受 [Non-record Submissions](https://github.com/openai/parameter-golf#non-record-submissions)。单卡 RTX 4090 跑 3 seeds 约 ¥10–40，诚实地把 `track` 填为 `non_record_16mb`、硬件填为真实单卡，即可拿到**真实 BPB 分数**，不必追求 8×H100 的 10 分钟记录。

**本包所有 BPB 数字分两类：**
- 「官方实测」：来自 `openai/parameter-golf` 公开 `records/`，可复核。
- 「目标/计划值」：我计划跑出的，非实测，待回填。

---

## 1. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| **JiaJingwen_C2G_README.md** | 说明 | 本文件，串联全部交付物 |
| **JiaJingwen_C2G_方案草案.md** | 必交 | Level 1 算力申请门槛（4 问已答） |
| **JiaJingwen_C2G_方案设计.md** | 必交 | 目标 Level / 技术选型 / 实验矩阵 E001–E006 / 消融 / 预算 / 风险 |
| **JiaJingwen_C2G_拿来说明.md** | 必交 | 从 nanoGPT / 官方 repo / 历史 top PR 拿了什么、改了什么（署名） |
| **JiaJingwen_C2G_leaderboard.md** | 对照 | 官方真实 BPB 演进表 + 各 Level 阈值 + 我的目标行 |
| **JiaJingwen_C2G_AI日志.md** | 必交 | 真实 AI 协作记录 + 实验期规划工作流 |
| **JiaJingwen_C2G_AAR.md** | 必交 | 复盘：已做 / 待 GPU / 失败诚实记录 / 改进方案 |
| **JiaJingwen_C2G_缺口补齐清单.md** | 新增 | 本 v2 补齐内容、缺口原因、后续 3 步 |
| **JiaJingwen_C2G_train_gpt.py** | 代码 | 官方 NaiveBaseline 真实训练脚本（可运行基线） |
| **JiaJingwen_C2G_run_experiments.py** | 脚本 | 一键训练-采集-提交流水线（GPU 机器执行） |
| **JiaJingwen_C2G_submission.json** | 元数据 | 提交模板（已删除虚假硬件申报，运行后自动回填） |
| **REF_baseline_submission.json** | 参考 | 官方 NaiveBaseline 真实 submission.json（字段对照） |

---

## 2. 评审维度覆盖对照

| 维度（满分） | 覆盖交付物 | 状态 |
|------------|-----------|------|
| 成绩达成 (25) | `submission.json` + `run_experiments.py` + `leaderboard` 目标行 | ⏳ 脚本就绪，待你在 GPU 机器执行一次 |
| 方法学 (20) | 方案设计（实验矩阵 + 消融） | ✅ |
| 产物完整性 (15) | README + train_gpt.py + submission 模板 + run 脚本 | ✅ |
| AI 使用质量 (20) | AI日志 | ✅ |
| 复盘质量 (20) | AAR | ✅ |

---

## 3. 如何复现真实成绩（推荐：RTX 4090 单卡 Non-record）

把本目录整体上传到云端 GPU 机器（RunPod / Vast / AutoDL 等），然后执行：

```bash
# 1) 安装依赖
pip install torch sentencepiece numpy tqdm flash-attn

# 2) 先 dry-run 检查环境
python JiaJingwen_C2G_run_experiments.py --dry-run

# 3) 运行完整流水线（自动下载 FineWeb SP1024 + SP8192，跑 3 seeds，回填 submission.json，打包 tar.gz）
python JiaJingwen_C2G_run_experiments.py
```

脚本会：
1. 检查 `nvidia-smi` 与 `torch.cuda`；
2. 自动 clone `openai/parameter-golf` 并使用官方 `data/cached_challenge_fineweb.py` 下载数据；
3. 将 `JiaJingwen_C2G_train_gpt.py` 复制为官方入口，依次跑 **E001 baseline (SP1024)** 和 **E003 SP8192**，各 3 seeds；
4. 解析日志生成 `metrics.json`；
5. 自动把真实 `val_bpb`、`bytes_total`、`hardware`、`pytorch_version` 回填进 `submission.json`；
6. 打包 `JiaJingwen_C2G_submission_<timestamp>.tar.gz`。

> 单卡 RTX 4090 训练时间大概率超过 10 分钟，因此脚本默认把 `track` 填为 `non_record_16mb`，`non_record: true`，硬件填真实 GPU 名。这是官方允许的诚实通道。

如果你坚持要走 record 通道（8×H100 10 分钟），请把脚本里的 `track` 改回 `10min_16mb`、并把 `MAX_WALLCLOCK_SECONDS` 保持默认 600，但官方 record 需要 beats SOTA ≥0.005 nats，本方案只做 SP8192 单点改进，大概率只能冲 Level 2/3 而非刷新 record。

---

## 4. 下一步（对你）

1. **租一张 RTX 4090**（RunPod / Vast / AutoDL，约 ¥10–40，单卡 24GB 足够）。
2. **上传本目录并运行**：`python JiaJingwen_C2G_run_experiments.py`。
3. **下载产物**：`metrics.json`、`JiaJingwen_C2G_submission.json`、`JiaJingwen_C2G_submission_*.tar.gz`，然后提交到挑战入口或 GitHub PR（non-record）。

---

*本交付包由 AI（WorkBuddy）协助整理，所有外部数据均来自官方公开仓库并标注来源。未包含任何伪造的实验成绩。*
