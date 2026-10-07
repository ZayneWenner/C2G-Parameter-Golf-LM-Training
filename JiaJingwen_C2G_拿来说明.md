# JiaJingwen_C2G_拿来说明（Provenance / "拿来主义"说明）

> 挑战：C2G 参数高尔夫
> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 目的：按 CHALLENGE.md「拿来主义原则」说明——**拿了什么、改了什么、为什么改**。所有外部来源均来自官方公开仓库 `openai/parameter-golf`（含 `records/` 真实提交记录）与 `KellerJordan/modded-nanogpt`。

---

## 0. 诚实边界声明

- 本说明中的代码与数值均**来自官方公开记录**，未做任何修改性声称。
- 我**尚未**在 GPU 上运行任何实验（环境无 8×H100、无法下载 FineWeb、无法提交榜单）。因此「我改了什么」目前是**计划态**（plan），待领取算力后落地并回填日志。
- SP8192 / SP4096 的 tokenizer 与预分词数据需在 GPU 机器上从 `kevclark/parameter-golf`（HuggingFace）拉取，本环境无法获取。

---

## 1. 从 nanoGPT / modded-nanoGPT 拿了什么

| 拿了什么 | 来源 | 用途 |
|---------|------|------|
| Transformer block 结构（Pre-LN、GQA、tied embedding） | `karpathy/nanoGPT` | 模型骨架 |
| 训练循环范式（分布式 DDP、`torchrun` 多卡、mixed precision） | `karpathy/nanoGPT` | 训练主流程 |
| **Muon 优化器** + Newton-Schulz 正交化（`zeropower_via_newtonschulz5`） | `KellerJordan/modded-nanogpt` → 官方 `train_gpt.py` 内已内置 | 优化器，取代 AdamW，提速/提质 |
| 评测思路（tokenizer-agnostic 压缩率） | modded-nanoGPT speedrun 社区 | 理解 BPB 作为底层能力度量 |

> 参考：`https://github.com/KellerJordan/modded-nanogpt`，Muon 博客 `https://kellerjordan.github.io/posts/muon/`

---

## 2. 从 OpenAI 官方 `train_gpt.py` 拿了什么

直接以官方 `train_gpt.py`（NaiveBaseline 快照，已复制为 `JiaJingwen_C2G_train_gpt.py`）为起点，未重写：

| 拿了什么 | 具体位置/接口 |
|---------|--------------|
| **I/O 与数据加载**：shard glob 读取 `fineweb_train_*.bin` / `fineweb_val_*.bin` | `Hyperparameters.data_path` / `train_files` / `val_files` |
| **Tokenizer 加载**：`sentencepiece` 加载 `.model` | `tokenizer_path = TOKENIZER_PATH` |
| **BPB 评测器**：`final_int8_zlib_roundtrip` 指标（int8 量化 + zlib 压缩往返，tokenizer 无关） | 脚本末尾 eval 段 |
| **提交格式**：`submission.json` 字段（`val_bpb` / `bytes_total` / `bytes_code` 等） | 见 `REF_baseline_submission.json` |
| **超参环境变量接口**：`VOCAB_SIZE` / `NUM_LAYERS` / `MODEL_DIM` / `NUM_KV_HEADS` / `MLP_MULT` / `TIE_EMBEDDINGS` / `QK_GAIN_INIT` / `MAX_WALLCLOCK_SECONDS` 等 | `Hyperparameters` 类 |
| **Muon 实现**：`zeropower_via_newtonschulz5` + `class Muon` | 脚本中段 |

> 这正是 CHALLENGE.md 推荐的「正确起点」：nanoGPT + 官方 `train_gpt.py` 脚手架。

---

## 3. 从官方历史 top 提交（PR lineage）拿了什么

下表基于 `records/` 真实 README 的 attribution 字段，列出我将复用的技术栈与对应原作者（**非我原创，明确署名**）：

| 技术组件 | 来源 PR / 作者 | 真实 BPB | 我在方案中的用法 |
|---------|--------------|---------|----------------|
| **SP8192 + GPTQ 嵌入 + SDClip + MuonEq-R + 深度循环** | PR #1394 @clarkkev | 1.0856 (5-seed) | **主基座**：作为我 E003 的 SP8192 实现参照 |
| **合法 Score-First TTT**（eval 期 score-before-update） | PR #549 @abaybektursun；PR #1413 @dexhunter | 叠加后 1.0828 | E006 进阶叠加，严格遵循 Issue #1017 四条件 |
| **3 层深度循环（layers 3-5 loop）** | PR #1331 / #1437 @dexhunter | 叠加后 1.0810 | E006 进阶 |
| **并行残差（layer 7+）** | PR #1204 @msisovic；PR #1412 @Robby955 | 叠加后 1.0810 | E006 进阶 |
| **QK-Gain 5.0 → 5.25**（可学习 per-head query 缩放） | PR #1217 @bigbag | 单旋钮 −0.0028 | E004 超参微调项 |
| **SP4096 + 4× MLP + 高 WD** | PR #1218 @clarkkev | 1.0979 | E002 过渡验证参照 |
| **Row-normalized Muon (MuonEq-R)** | PR #1260 @dexhunter | 1.0912+ | E005 优化器替换 |
| **Full-Hessian GPTQ + SDClip 量化** | PR #1019 @abaybektursun；PR #1394 @clarkkev | — | artifact 压缩至 ≤16MB |

> 证据出处：官方仓库 `records/track_10min_16mb/` 下各 `README.md`（NaiveBaseline、SP8192_QK5_LegalTTT、SP8192_3LayerRecur…）。2026-10-07 复核：当前 SOTA = **1.0565**（codemath3000, PR#2135, 2026-05-01）；1.0810 为 2026-04-09 时点值。

---

## 4. 我改了什么（计划态，待 GPU 落地）

> 以下为**计划**，非已执行。落地后在 `JiaJingwen_C2G_logs/` 与 `submission.json` 回填真实数据。

1. **Tokenizer 升级 BPE-1024 → SP8192**（E003 主线）
   - 改动：仅改 `TOKENIZER_PATH` + `VOCAB_SIZE=8192`，并重建数据缓存（删除旧 `manifest.json` 后用 `cached_challenge_fineweb.py --variant sp8192`）。
   - 为什么改：单点杠杆最大（真实记录 −0.14 BPB），且是 top 方案公共前缀。
   - 风险：tokenizer 模型文件更大，需确认 artifact ≤16MB（见方案设计 §7-B）。

2. **超参微调**（E004）：在 SP8192 基座上，仅调 `QK_GAIN_INIT`（4.0→5.0，参照 PR #1217）、`MATRIX_LR`、`WARMDOWN` 比例，验证是否进一步下降且非噪声。

3. **组合优化（E005/E006，进阶）**：SP8192 + MuonEq-R / + 深度循环 / + 并行残差 / + 合法 TTT，按方案设计 §4 消融逐一验证组件贡献。

4. **所有改动通过环境变量注入**，不修改脚本主体结构 → 保证他人可一键复现。

---

## 5. 为什么这样改（可解释性）

- **第一杠杆选 tokenizer 而非 optimizer/架构**：改动面最小、风险最低、预期收益最高、且是 SOTA 公共基座（三重印证，见 `方案设计.md` §2）。
- **组合时严格消融**：每个组件开关各跑 ≥3 seed，证明单独贡献与组合协同，避免 CHALLENGE.md 警告的「组合相互抵消甚至负向」。
- **合法 TTT 严格守规则**：仅对「已评分」的 val token 做 score-first 适配，满足 Issue #1017 条件 1–4（因果性 / 归一化分布 / 先评分后更新 / 单次评分），杜绝 SLOT / ETLB / n-gram cache 等违规技巧。

---

## 6. 引用清单（供评审核查）

- OpenAI Parameter Golf 官方仓库与 records：`https://github.com/openai/parameter-golf`
- NaiveBaseline 真实记录（val_bpb 1.2244）：`records/track_10min_16mb/2026-03-17_NaiveBaseline/`
- SP8192 + GPTQ + 深度循环（PR #1394, clarkkev, 1.0856）
- SP8192 + QK5 + 合法 TTT（dexhunter, 1.0828）
- SP8192 + 3层循环 + 并行残差 + TTT（bigbag, 1.0810，2026-04-09 时点 SOTA；2026-10-07 复核当前榜首 1.0565）
- Muon 优化器：Keller Jordan, `https://kellerjordan.github.io/posts/muon/`
- SentencePiece：Kudo & Richardson (2018), arXiv:1808.06209
- nanoGPT：Karpathy, `https://github.com/karpathy/nanoGPT`
- modded-nanoGPT：KellerJordan, `https://github.com/KellerJordan/modded-nanogpt`
