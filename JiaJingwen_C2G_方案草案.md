# JiaJingwen_C2G_方案草案

> 算力申请门槛文档（Level 1 Gate）
> 挑战：C2G 参数高尔夫 —— 极限约束下的语言模型训练
> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 目标：先拿 Level 1 $25 算力券 → 复现 baseline（1.2244 BPB）→ 主线压向 SP8192 Tokenizer，冲 Level 2（BPB < 1.18）

---

## 0. 一句话总纲

我不追求"四个方向一起试"，而是按 CHALLENGE.md 的建议，**把第一个确定性杠杆——Tokenizer 升级到 SentencePiece 8192——做扎实**，用"改一个东西 → 跑 3 个 seed → 看 BPB 差值"的完整流程把 Level 1→Level 2 打通，再决定下一步组合。

---

## Q1：把 baseline 从 1.2244 往哪个方向压？

**明确选一个：Tokenizer 升级（BPE-1024 → SentencePiece-8192）。** 不选"都试试"。

理由（为什么是它而不是其他三个）：

- **改动面最小、风险最低**：改的是数据预处理链路（tokenizer + 数据缓存重建），**不动 Transformer 本体**。这让我能在 10 分钟约束下先建立一个"改动→指标"的可观测闭环，而不是一上来就动架构导致 debug 成本爆炸。
- **杠杆最大**：CHALLENGE.md 明确写着 Tokenizer 升级历史收益 **-0.03 ~ -0.05 BPB**，是所有单点方向里预期收益最高的；而 Optimizer/架构微调普遍只有 -0.01 ~ -0.02。
- **为后续组合铺路**：SP8192 是几乎所有当前 top 方案的公共前缀（见 Q2 证据），把它先做通，等于把后续 SP8192+Muon、SP8192+recurrence 的地基打好。

> 一句话：先做"最大确定收益 + 最小改动面"的那一件事，而不是在四个方向里平均用力。

---

## Q2：你怎么知道这个方向有效？（证据）

**直觉机制**：BPB = 模型对原始字节序列的压缩成本，越低越好。vocab 越大，单个 token 平均覆盖的字节越多 → 同样文本被切成更少的 token → 模型需要的"预测步数"更少 → BPB 下降。SentencePiece 的无词表空格切分（BPE 之外的 unigram/word-piece 风格）也减少了英文/代码里的 subword 碎片。

**论文证据（≥1 篇）**：
- Kudo & Richardson (2018), *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*, arXiv:1808.06209。这是 SP 的权威出处，论证了"language-independent subword 单元"在压缩率上的优势。

**实战/历史提交证据（≥1 条）**：
- **当前榜首 SP8192 + 3-Layer Recurrence ≈ 1.0810 BPB**（CHALLENGE.md 第 50–51 行）。注意：榜首方案的核心前缀就是 `SP8192`——说明 SentencePiece-8192 是 top 方案的公共基石，而非边缘技巧。
- **官方 `reference_top_submissions/` 中 `sp8192_*` 占据前 5 席中的多席**（`sp8192_3layer_recurrence`、`sp8192_parallel_residual`、`sp8192_qk_gain5`、`sp8192_hessian_sdclip`、`sp8192_gptq_embedding`）——5 个里 5 个都带 SP8192，这是比任何论文都硬的"社区共识"证据。
- **modded-nanoGPT 历史**：Keller Jordan 一脉的 speedrun 社区普遍把"替换/扩充 tokenizer"作为第一个可观测杠杆；OpenAI 官方 `train_gpt.py` 脚手架也已内置 `sp_8192.model` 作为可选 tokenizer，说明官方默认路径就包含它。

> 结论：方向有效既不是我拍脑袋，也不是孤立论文，而是"论文机制 + 榜首实证 + 官方脚手架内置"三重印证。

---

## Q3：你打算跑多少次实验？每次看什么指标？$25 怎么花？

**核心省钱原则**（CHALLENGE.md Step 6 提示）：80% 的实验在**单张 A100（或更便宜硬件）用小模型验证思路**，只有最后 3 次正式提交才上 **8×H100**。

### 实验矩阵（先验证、后上分）

| 编号 | 配置 | 硬件 | seed 数 | 看什么 |
|------|------|------|---------|--------|
| E001 | baseline（naive_9l512d，BPE-1024） | 8×H100 | 1（复现） | 能否复现 1.2244 ±0.005；理解 `submission.json` 每个字段 |
| E002 | + SP4096（过渡，验证 pipeline） | 单 A100（小模型） | 3 | tokenizer 重建是否正确、BPB 趋势、artifact 体积 |
| E003 | + SP8192（主目标） | 单 A100（小模型）→ 8×H100 | 3 | **主指标 BPB**；tokens/sec；训练时长（必须 < 8:00 留 20% buffer）；artifact 是否 ≤16MB |
| E004 | SP8192 + 轻微超参（lr/warmup 微调） | 8×H100 | 3 | 是否进一步压 BPB，确认不是噪声 |

### 指标优先级
1. **BPB（主，越低越好）**——每个实验跑 3 seed 取均值 ±stderr，避免"同样代码差 0.01"的假阳性。
2. **训练 wallclock**——必须 ≤ 8:00（预留 20% buffer，因为别人机器可能更慢）。
3. **artifact 体积**——训练循环末尾自动断言 ≤ 16,000,000 字节，超了立即告警。
4. **辅助诊断**：loss 曲线、grad norm、tokens/sec（用于判断训练是否稳定、是否过拟合验证集——**绝不用验证集调超参**）。

### $25 预算分配（约 1–2 次完整 8×H100 跑）
- **A100 小模型验证（约 $3–5）**：E002/E003 的前期验证在便宜硬件跑，迭代 tokenizer 重建逻辑。
- **8×H100 正式跑（约 $20）**：
  - 1 次 baseline 复现（E001）
  - 3 次 SP8192 正式 seed（E003）
  - 1–2 次 SP8192 + 超参微调（E004）冲 Level 2 的 <1.18
- 富余额度留作重跑（任何一次 wallclock 踩边或 artifact 超限都需重跑）。

---

## Q4：失败怎么办？（最坏情况预案）

**场景 A：SP8192 让 BPB 反而变差 / 持平**
- 先排除实现错误：检查 SentencePiece 模型是否正确加载、`fineweb_val` 是否用**同一个** tokenizer 重新编码、data cache 是否重建（旧 BPE cache 没清是最常见坑）。
- 若实现正确仍无收益：说明本数据集上 vocab 增益被其他因素抵消，立即**回退到 baseline**，把备选方向提上来——按 CHALLENGE.md，下一优先是 **Muon optimizer**（预期 -0.01~-0.02，且和 tokenizer 正交，后续可叠加）。

**场景 B：artifact 超 16MB**
- SP8192 的 tokenizer 模型文件比 BPE-1024 大。对策：用 `sp_8192.model` 而非更大 vocab；若仍超，对 embedding 做 int8 量化腾出预算（这正是 CHALLENGE.md 量化方向，提前为 Level 3 铺路）。

**场景 C：连 baseline 都复现不到 1.2244±0.005**
- 不急着改模型。逐级排查：① 数据下载是否完整（`download_fineweb.py` 缓存）② seed / 评测脚本是否和官方一致 ③ 混合精度（bf16）与 flash-attn 版本 ④ 8 卡 `torchrun` 通信是否真的 8 卡都在跑。把每一步诊断写进 AI 日志。

**场景 D：10 分钟踩边**
- 把目标训练时间压到 8:00，减少 step 数或增大 batch；绝不为了多训 30 秒赌墙钟。

> 这一条的本质：失败预案暴露的不是一个"Plan B 列表"，而是我是否真的想清楚了"每一步改动的可观测性"。我的底线是——**任何一次实验都必须能解释 BPB 为什么变好或变坏，解释不了就先停下来**。

---

## 附：AI-First 工作流（预审即体现）

- 写代码前，把 `train_gpt.py` 逐段喂给 AI 做 code review，先问清每一行对 BPB 的影响（CHALLENGE.md Step 2）。
- 每个实验假设先用 AI 生成对照设计 + 预期差值区间，再开跑。
- 训练日志（loss / grad norm / tokens-per-sec）用 AI 做可视化与异常诊断。
- 所有 AI 对话按 `ts >> log` 落地到 `JiaJingwen_C2G_AI日志.md`，供评审查时间戳与连贯性（避免"AI 日志造假"红牌）。

## 附：拿来说明（preview，正式版进 `JiaJingwen_C2G_拿来说明.md`）

- **nanoGPT**（Karpathy）：Transformer block / 训练循环骨架。
- **OpenAI 官方 `train_gpt.py`**：I/O、BPB 评估器、提交格式。
- **历史 top 方案**：`reference_top_submissions/sp8192_*` 作为 tokenizer 实现的参照 anchor。

---

*本草案满足 Level 1 算力申请 4 问门槛，承诺先复现 baseline、再做 SP8192 单点、再谈组合。审核通过后即领取 $25 算力券开跑。*
