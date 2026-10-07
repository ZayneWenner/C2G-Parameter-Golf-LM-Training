# JiaJingwen_C2G_方案设计（Plan Design）

> 挑战：C2G 参数高尔夫 —— 极限约束下的语言模型训练
> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 关联文档：`JiaJingwen_C2G_方案草案.md`（Level 1 算力申请门槛，已提交）、`JiaJingwen_C2G_拿来说明.md`、`JiaJingwen_C2G_leaderboard.md`、`JiaJingwen_C2G_AI日志.md`、`JiaJingwen_C2G_AAR.md`
> 命名规范：`姓名拼音_C2G_内容描述.扩展名`

---

## 0. 本文档定位

`方案草案` 已回答算力申请 4 问门槛并申请 Level 1（$25）算力券。本文档在其基础上**扩展为完整实验设计**，覆盖 CHALLENGE.md「必须提交的文件」对「方案设计」的要求：目标 Level、技术选型、实验矩阵、消融设计、预算、时间线、风险预案、复现与提交格式。

**重要前提声明（诚实边界）：** 本环境无 8×H100 GPU、无法联网下载 FineWeb 评估集、无法提交 OpenAI 官方榜单，因此本文档中的 BPB 数值分为两类——
- **实测值（来自官方公开记录）**：标注「官方实测」，如 NaiveBaseline 1.2244、SP8192+TTT 1.0810。
- **目标/预测值（计划跑出的）**：标注「目标」，需在领取算力后真实跑出并回填 `JiaJingwen_C2G_submission.json`。
本文档**不伪造任何已测 BPB 数字**；真实成绩达成维度依赖后续 GPU 运行。

---

## 1. 目标定位（Target Level）

| 层级 | 目标 BPB | 本方案态度 | 对应交付 |
|------|---------|-----------|---------|
| **Level 1（Bronze）** | 复现 baseline **1.2244** (±0.005) | **必做（gate）** | baseline 训练日志 + 16MB artifact + submission.json |
| **Level 2（Silver）** | **< 1.18** | **主线目标** | + SP8192 单点改进 + 对照实验报告（≥3 seed） |
| **Level 3（Gold）** | **< 1.12**（前 50） | 进阶目标（若预算与时间允许） | 组合优化 + 消融表 + 显著性分析 |
| **Level 4（Platinum）** | < 1.085（前 20） | 仅作路线预留，不承诺 | 原创改动 + 英文 tech report |

**一句话：** 先把 Level 1 跑通拿到「入场券」，主线把第一个确定性杠杆 **Tokenizer 升级 SP8192** 做扎实冲 Level 2；Level 3 作为预算充裕时的进阶，路线已在第 3 节实验矩阵中预留。

---

## 2. 技术选型与理由（为什么先动 Tokenizer）

核心决策：**第一个改动只动 Tokenizer（BPE-1024 → SentencePiece-8192），不动 Transformer 本体。**

依据（来自官方公开记录的硬证据，详见 `leaderboard.md`）：

| 改动 | 真实 BPB | 相对 baseline 降幅 | 来源 |
|------|---------|------------------|------|
| baseline（SP-1024） | 1.2244 | — | openai 2026-03-18 |
| SP4096 + 深度循环 + 并行残差 + MuonEq-R | 1.0897 | −0.1347 | aryanbhosale 2026-04-04 |
| SP8192 + GPTQ emb + 深度循环 + SDClip | 1.0856 | −0.1388 | clarkkev PR#1394 2026-04-05 |
| SP8192 + QK5 + 合法 TTT | 1.0828 | −0.1416 | dexhunter 2026-04-06 |
| SP8192 + 3层循环 + 并行残差 + TTT | **1.0810** | −0.1434 | bigbag 2026-04-09（当前 SOTA） |

**机制直觉：** BPB = 模型对原始字节序列的压缩成本，越低越好。vocab 越大 → 单 token 覆盖字节越多 → 同等文本被切成更少 token → 预测步数更少 → BPB 下降。SentencePiece 的 unigram/BPE 风格无词表空格切分进一步减少 subword 碎片。

**为什么是它而不是其他三个方向（Optimizer / 架构 / 量化）：**
1. **改动面最小、风险最低**：只改数据预处理链路（tokenizer + 数据缓存重建），不动 Transformer 本体，先在 10 分钟约束下建立「改动→指标」可观测闭环。
2. **杠杆最大**：单点 SP8192 在真实记录里带来约 −0.14 BPB，远超 Optimizer/架构微调常见的 −0.01~−0.02。
3. **是 top 方案的公共前缀**：当前 SOTA（1.0810）与所有 1.08x 方案都带 `SP8192` 前缀，把它做通=为后续 +Muon / +循环 / +TTT 铺好地基。

---

## 3. 实验矩阵（先验证、后上分）

遵循 CHALLENGE.md 省钱原则：**80% 实验在单张 A100（小模型）验证，仅最后 ≤3 次正式提交上 8×H100**。

| 编号 | 配置 | 硬件 | seed 数 | 主看指标 | 目标 BPB | 目的 |
|------|------|------|---------|---------|---------|------|
| **E001** | baseline（naive_9l512d，BPE-1024） | 8×H100 | 1（复现） | 能否复现 1.2244±0.005；理解 `submission.json` 字段 | 1.2244 | Level 1 gate |
| **E002** | + SP4096（过渡，验证 pipeline） | 单 A100（小模型） | 3 | tokenizer 重建是否正确、artifact 体积 | ~1.09（参照） | 验证数据缓存重建链路 |
| **E003** | **+ SP8192（主目标）** | 单 A100→8×H100 | 3 | **主指标 BPB**；tokens/sec；wallclock；artifact≤16MB | **~1.085–1.09** | Level 2 主线 |
| **E004** | SP8192 + 轻微超参（lr/warmup/QK-gain） | 8×H100 | 3 | 是否进一步压 BPB（确认非噪声） | < 1.18 | 冲 Level 2 下限 |
| **E005**（进阶） | SP8192 + MuonEq-R | 8×H100 | 3 | 组合是否叠加、是否抵消 | < 1.15 | Level 3 路线预留 |
| **E006**（进阶） | SP8192 + 深度循环 + 并行残差 + 合法 TTT | 8×H100 | 3 | 组合消融、显著性 p<0.01 | < 1.12 | Level 3 路线预留 |

> 注：E005/E006 为路线预留，具体超参与组合以 E003/E004 实测结论为准，避免「四个方向平均用力」。

### 指标优先级（每个实验必录）
1. **BPB（主，越低越好）**——每实验 ≥3 seed 取均值 ±stderr，规避「同代码差 0.01」假阳性。
2. **训练 wallclock**——目标 ≤ 8:00（预留 20% buffer，因他人机器可能更慢）。
3. **artifact 体积**——训练循环末尾自动断言 ≤ 16,000,000 字节，超限立即告警。
4. **辅助诊断**：loss 曲线、grad norm、tokens/sec（判断稳定性、是否过拟合验证集；**绝不用验证集调超参**）。

---

## 4. 对照实验与消融设计（Ablation）

原则（CHALLENGE.md Level 3 要求）：**任何新 record 必须比当前 SOTA 低 ≥0.005 nats 且 p<0.01（≥3 seed）**；组合优化必须证明每个组件贡献。

| 组件 | 单独贡献验证 | 组合验证 |
|------|------------|---------|
| SP8192 tokenizer | E003 vs E001（同架构，仅换 tokenizer） | 作为所有后续实验的公共基座 |
| MuonEq-R optimizer | E005 vs E003（同 SP8192，仅换优化器） | E005 内再分 Muon vs AdamW |
| 深度循环（layers 4-5 loop） | 关/开 recurrence 对照 | 与并行残差交叉 |
| 并行残差（layer 7+） | 关/开对照 | 与循环交叉 |
| 合法 Score-First TTT | 关/开对照（eval 期） | 最后叠加，单独报 TTT gain |

**统计分析：** 每个组件开关各跑 ≥3 seed，报告均值 ±stderr；组合创新需满足 ΔBPB ≥ 0.005 nats 且 p<0.01（独立样本 t 检验或 bootstrap）。所有原始日志保留于 `JiaJingwen_C2G_logs/`。

---

## 5. 资源与预算

| 等级 | 赞助额度 | 申请条件（已满足/计划） |
|------|---------|----------------------|
| Level 1 | $25 算力券 | ✅ `方案草案.md` 已提交 |
| Level 2 | $100 算力券 | 待 Level 1 通过 + 单点改进 RFC（本文档即基础） |

**$25 分配（约 1–2 次完整 8×H100 跑）：**
- A100 小模型验证（≈$3–5）：E002/E003 前期在便宜硬件迭代 tokenizer 重建逻辑。
- 8×H100 正式（≈$20）：1 次 baseline 复现（E001）+ 3 次 SP8192 正式 seed（E003）+ 1–2 次超参微调（E004）冲 <1.18。
- 富余额度留作重跑（wallclock 踩边或 artifact 超限必须重跑）。

---

## 6. 时间线（截止 2026-12-31）

| 阶段 | 事项 | 里程碑 |
|------|------|--------|
| W1 | 领 $25 券 → Fork repo → 跑 E001 复现 baseline | Level 1 gate ✅ |
| W2 | E002/E003 验证 SP8192 pipeline（先 A100 后 8×H100） | 拿到首个 SP8192 BPB |
| W3 | E004 超参微调，冲 <1.18；写对照实验报告 | Level 2 ✅ + RFC 申请 $100 |
| W4（若有预算） | E005/E006 组合优化 + 消融 + 显著性 | Level 3 尝试 |
| 12-31 前 | 整理 submission.tar.gz、README、push GitHub、提交榜单 | 全部交付物齐全 |

---

## 7. 风险预案（失败怎么办）

| 场景 | 症状 | 对策 |
|------|------|------|
| **A. SP8192 反变差/持平** | BPB 不降 | 先排查：SP 模型是否正确加载、`fineweb_val` 是否用同一 tokenizer 重编码、data cache 是否重建（旧 BPE cache 未清最常见）。实现正确仍无收益→回退 baseline，提备选 **Muon optimizer**（与 tokenizer 正交，可叠加）。 |
| **B. artifact 超 16MB** | 提交被拒 | SP8192 tokenizer 比 BPE-1024 大。用 `sp_8192.model`；仍超则对 embedding 做 int8 量化腾预算（正是量化方向，提前铺 Level 3）。 |
| **C. baseline 复现不到 1.2244±0.005** | 数字对不上 | 逐级排查：① 数据下载完整性 ② seed/评测脚本一致性 ③ bf16 与 flash-attn 版本 ④ 8 卡 torchrun 是否真 8 卡都在跑。每步诊断写入 AI 日志。 |
| **D. 10 分钟踩边** | 本地 9:50，他人 10:20 | 目标训练时间压到 8:00，减 step 或增 batch；绝不为多训 30 秒赌墙钟。 |

**底线：** 任何一次实验都必须能解释 BPB 为何变好/变坏，解释不了就先停。

---

## 8. 复现性与提交格式

- **代码**：`JiaJingwen_C2G_train_gpt.py`（已复制官方 NaiveBaseline 真实脚本为基线，可运行）。所有改动通过环境变量接口（`VOCAB_SIZE` / `TOKENIZER_PATH` / `QK_GAIN_INIT` 等）注入，保证「干净环境一键复现」。
- **提交元数据**：`JiaJingwen_C2G_submission.json`（模板，字段对齐官方 `REF_baseline_submission.json`）。真实跑出后回填 `val_bpb` / `bytes_total` / `seeds`。
- **日志**：`JiaJingwen_C2G_logs/` 至少 3 独立 seed 完整日志。
- **16MB artifact**：`JiaJingwen_C2G_submission.tar.gz`（含代码 + 压缩权重 + tokenizer），训练末尾自动断言 ≤16,000,000 字节。

---

## 9. 与方案草案的关系

`方案草案` 回答了「往哪压 / 为何有效 / 怎么花算力 / 失败怎么办」4 个门槛问题，并选定 SP8192 为第一杠杆。本文档在其上扩展为：明确 Level 目标、细化实验矩阵（E001–E006）、补消融统计要求、预算时间线、复现格式。**两文档共同构成 Level 1→Level 2 的完整研究计划**，供挑战组审核与后续执行。
