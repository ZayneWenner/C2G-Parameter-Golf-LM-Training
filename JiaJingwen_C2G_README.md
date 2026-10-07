# C2G 参数高尔夫 —— 作业交付包（JiaJingwen）

> 作者：贾静文（JiaJingwen）｜挑战 ID：ch-20260717031359-b8wyg0｜截止：2026-12-31
> 交付日期：2026-10-07

---

## 0. 重要说明（请先读）

本挑战的 **35%–25% 分数来自真实 BPB 成绩**，那需要 **8×H100 GPU + 下载 FineWeb + 提交 OpenAI 官方榜单**。当前工作环境**无 GPU、无法联网下载数据集、无法提交榜单**，因此：

- ✅ **已真实完成**：评审 5 维度中 4 个文档类维度（方法学 / 产物完整性 / AI 使用 / 复盘）的全部交付物，均基于**官方公开真实数据**写成，未编造任何数字。
- ⏳ **待 GPU 完成**：`scoreAchievement`（成绩达成）维度的真实 BPB，需在领取算力后跑出并回填 `JiaJingwen_C2G_submission.json`。

**本包所有 BPB 数字分两类，已在文中明确标注：**
- 「官方实测」：来自 `openai/parameter-golf` 公开 `records/`，可复核。
- 「目标/计划值」：我计划跑出的，非实测，待回填。

---

## 1. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| **JiaJingwen_C2G_README.md** | 说明 | 本文件，串联全部交付物 |
| **JiaJingwen_C2G_方案草案.md** | 必交 | Level 1 算力申请门槛（4 问已答），已提交 |
| **JiaJingwen_C2G_方案设计.md** | 必交 | 目标 Level / 技术选型 / 实验矩阵 E001–E006 / 消融 / 预算 / 风险 |
| **JiaJingwen_C2G_拿来说明.md** | 必交 | 从 nanoGPT / 官方 repo / 历史 top PR 拿了什么、改了什么（署名） |
| **JiaJingwen_C2G_leaderboard.md** | 对照 | 官方真实 BPB 演进表 + 各 Level 阈值 + 我的目标行 |
| **JiaJingwen_C2G_AI日志.md** | 必交 | 真实 AI 协作记录 + 实验期规划工作流 |
| **JiaJingwen_C2G_AAR.md** | 必交 | 复盘：已做 / 待 GPU / 失败诚实记录 / 改进方案 |
| **JiaJingwen_C2G_train_gpt.py** | 代码 | 官方 NaiveBaseline 真实训练脚本（可运行基线） |
| **JiaJingwen_C2G_submission.json** | 元数据 | 提交模板（真实跑出后回填 val_bpb / bytes_total / seeds） |
| **REF_baseline_submission.json** | 参考 | 官方 NaiveBaseline 真实 submission.json（字段对照） |

---

## 2. 评审维度覆盖对照

| 维度（满分） | 覆盖交付物 | 状态 |
|------------|-----------|------|
| 成绩达成 (25) | submission.json + leaderboard 目标行 | ⏳ 待 GPU 实测 |
| 方法学 (20) | 方案设计（实验矩阵 + 消融） | ✅ |
| 产物完整性 (15) | README + train_gpt.py + submission 模板 | ✅ |
| AI 使用质量 (20) | AI日志 | ✅ |
| 复盘质量 (20) | AAR | ✅ |

---

## 3. 如何复现真实成绩（领取算力后）

```bash
# 1) Fork 并克隆官方仓库
git clone https://github.com/openai/parameter-golf.git && cd parameter-golf

# 2) 安装依赖
pip install -r requirements.txt
pip install flash-attn torch sentencepiece numpy tqdm

# 3) 领取 $25 算力券后，在 8×H100 上跑基线（E001）
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
# 预期：val_bpb ≈ 1.2244，artifact < 16MB

# 4) 升级 SP8192（E003，主目标）
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
VOCAB_SIZE=8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  torchrun --standalone --nproc_per_node=8 JiaJingwen_C2G_train_gpt.py
# 预期：val_bpb ≈ 1.085–1.09（参照官方 PR#1394）

# 5) 回填真实结果
#   将每 seed 的 val_bpb / bytes_total 写入 JiaJingwen_C2G_submission.json
```

> 详细超参接口见 `JiaJingwen_C2G_train_gpt.py` 中 `Hyperparameters` 类的环境变量定义。

---

## 4. 下一步（对你）

1. 把 `JiaJingwen_C2G_方案草案.md` + `JiaJingwen_C2G_方案设计.md` 提交挑战组，申请 Level 1 $25 算力券。
2. 领券后在 GPU 机器上执行上节 E001→E003，拿到真实 BPB 回填 `submission.json`。
3. 跑通后，本包即满足全部「必交」交付物，且成绩达成维度可计分。

---

*本交付包由 AI（WorkBuddy）协助整理，所有外部数据均来自官方公开仓库并标注来源。未包含任何伪造的实验成绩。*
