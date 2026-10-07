# JiaJingwen_C2G_leaderboard（BPB 对比表）

> 挑战：C2G 参数高尔夫
> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 数据来源：官方 `openai/parameter-golf` 仓库 `README.md` Leaderboard 与 `records/` 各 submission.json（**全部为官方实测值**）。
> 复核：2026-10-07 重新抓取官方 README 逐行核对，补入 2026-04-09 之后新增的 16 条记录；**榜首已由 1.0810 更新为 1.0565**。
> 指标：**BPB（Bits-Per-Byte）越低越好**，tokenizer 无关，评估集为 FineWeb validation。

---

## 1. 官方公开榜单演进（真实实测，按 BPB 升序）

| 排名 | Run（方案） | BPB | 作者 | 日期 | 关键组件 |
|----:|------------|----:|------|------|---------|
| — | **Naive Baseline** | **1.2244** | openai | 2026-03-18 | 9L×512d, SP-1024, tied emb, 4 KV |
| 26 | fp16 Embed | 1.2197 | Renier Velazco | 2026-03-18 | FP16 tied emb + LR 调 |
| 17 | int6 mixed precision | 1.2147 | Nan Liu | 2026-03-19 | 10L int8/int6 |
| 14 | 2048 seq length | 1.2060 | Spokane Way | 2026-03-18 | 序列长度 |
| 13 | 4k seq length | 1.2014 | Spokane Way | 2026-03-19 | 序列长度 |
| 11 | LoRA TTT | 1.1928 | samacqua | 2026-03-19 | 测试时训练 |
| 10 | Sliding Window Eval | 1.1925 | Matthew Li | 2026-03-19 | 滑窗评估 |
| 9 | Muon WD + 10L | 1.1748 | notapplica | 2026-03-19 | Muon + 谱初始化 |
| 8 | Mixed Quant + Sliding | 1.1630 | aquariouseworkman | 2026-03-19 | int6+int8 |
| 7 | 10L Int6 QAT + Zstd | 1.1586 | yahya010 | 2026-03-19 | int6 QAT |
| 6 | Ternary Quant | 1.1570 | Ciprian-Florin Ifrim | 2026-03-24 | 1/0/-1 三值 |
| 5 | Int6 MLP3x + SmearGate | 1.1458 | Raahil Shah | 2026-03-20 | 3×MLP |
| 4 | 10L Int5-MLP + BigramHash | 1.1428 | thwu1 | 2026-03-20 | int5 |
| 3 | 11L Efficient XSA | 1.1307 | unnir | 2026-03-20 | XSA |
| 2 | 11L XSA4 + EMA + Int6 | 1.1271 | jfprincz | 2026-03-20 | XSA4 |
| 1 | 11L Partial RoPE + LN | 1.1248 | jfprincz | 2026-03-21 | Partial RoPE |
| 0 | 11L EMA + GPTQ-lite | 1.1228 | signalrush | 2026-03-22 | GPTQ-lite + warmdown |
| — | **11L AR Self-Gen GPTQ + XSA** | **1.1147** | abaybektursun (PR#1019) | 2026-03-25 | 自生成 GPTQ 校准 |
| — | **Parallel Resid + Mini Recurrence** | **1.1063** | Marko Sisovic (PR#1204) | 2026-03-31 | 并行残差 |
| — | **4096-Vocab + 4×MLP + 高 WD** | **1.0979** | clarkkev (PR#1218) | 2026-04-01 | SP4096 |
| — | **MuonEq-R + Recurrence + All-Int6** | **1.0912** | dexhunter (PR#1285) | 2026-04-03 | MuonEq-R |
| — | **SP4096 + 深度循环 + 并行残差 + MuonEq-R** | **1.0897** | aryanbhosale | 2026-04-04 | SP4096 栈 |
| — | **SP8192 + GPTQ Emb + 深度循环 + SDClip** | **1.0856** | clarkkev (PR#1394) | 2026-04-05 | **SP8192 栈（我的主基座）** |
| — | **SP8192 + Parallel Resid + Hessian SDClip** | **1.0835** | Robby Sneiderman (PR#1412) | 2026-04-06 | 并行残差 |
| — | **SP8192 + QK5 + 合法 TTT** | **1.0828** | dexhunter (PR#1413) | 2026-04-06 | +TTT |
| — | **SP8192 + Parallel Resid + Score-First TTT** | **1.0822** | aryanbhosale (PR#1477) | 2026-04-08 | +并行残差 |
| — | **SP8192 + 3层循环 + 并行残差 + QK5.25 + TTT** | **1.0810** | bigbag (PR#1493) | 2026-04-09 | 3层循环+并行残差+QK5.25+合法 TTT（**2026-04-09 时点 SOTA**） |
| — | **SP8192 + Muon 0.97 + Legal Score-First TTT** | **1.0798** | dexhunter (PR#1514) | 2026-04-09 | SP8192 + Muon 0.97 + 合法 score-first TTT（3-seed，p=0.020 vs #1493） |
| — | **Improved Parallel Residuals + CUTLASS EVT + Legal TTT** | **1.0758** | msisovic (PR#1529) | 2026-04-11 | 双通道并行残差（START=8）+ CUTLASS EVT（修正后 3-seed） |
| — | **VarLen Attention + Fused MLP + Doc-Independent Legal TTT** | **1.0734** | samacqua (PR#1530) | 2026-04-11 | 变长 FA3 注意力 + 融合 Triton MLP |
| — | **VarLenAttn + PhasingTTT** | **1.0728** | romeerp (PR#1610) | 2026-04-13 | 分阶段 TTT（在已评分 chunk 上） |
| — | **VarLen Attention + Fused MLP + Multi-Phase Global SGD TTT** | **1.0719** | dexhunter (PR#1626) | 2026-04-14 | 多阶段全局 SGD TTT + int7 嵌入 |
| — | **SmearGate + Attention Output Gate + Legal TTT** | **1.0714** | MarioPaerle (PR#1667) | 2026-04-16 | SmearGate + 注意力输出门 |
| — | **CaseOps Tokenizer + Tapered WD + Phased TTT** | **1.0678** | romeerp (PR#1729) | 2026-04-19 | CaseOps 无损大小写变换 + 字节 sidecar 记账 |
| — | **SP8192 + CaseOps + GatedAttn + QuantGate + Loop45 + Phased TTT** | **1.0655** | dexhunter (PR#1736) | 2026-04-19 | SP8192 底座 + CaseOps + 门控注意力（本档起脱离 1.08x） |
| — | **CaseOps + MLPClip12 + SmearGate / LoRA-TTT** | **1.0645** | dexhunter (PR#1769) | 2026-04-22 | MLPClip12（5-seed） |
| — | **PR1736 + PolarNS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT** | **1.0634** | nprime06 (PR#1787) | 2026-04-23 | PolarNS + SparseAttnGate + 融合 softcapped CE |
| — | **BOS-Fixed SmearGate + LQER Asymmetric + PR1787 SparseAttn + Phased TTT** | **1.0614** | aquariouseworkman (PR#1851) | 2026-04-27 | BOS 边界修正 + LQER 非对称 |
| — | **BOS-Fixed SmearGate + LQER + SparseAttnGate + 9-Hparam Stack** | **1.0611** | codemath3000 (PR#1855) | 2026-04-27 | 9 项贪心超参覆盖（3-seed） |
| — | **AWQ-Lite GPTQ + AsymLogit on PR1855 Stack** | **1.0594** | alertcat (PR#1945) | 2026-04-29 | AWQ-lite 混合 GPTQ + AsymLogit（3-seed） |
| — | **Long-Context No-Q/V TTT + QK-Gain 5.25** | **1.0586** | andrewbaggio1 (PR#1953) | 2026-04-30 | 2560 评估/TTT 上下文 + 无 Q/V TTT mask |
| — | **Progressive Context Growth + Short-Doc Score-First TTT** | **1.0576** | simonbissonnette (PR#2014) | 2026-04-30 | 渐进上下文增长到 3k（3-seed） |
| — | **Calib32 Token-Only N-gram + AsymLogit Stack** | **1.0565** | codemath3000 (PR#2135) | 2026-05-01 | **当前 SOTA**（3-seed mean 1.05651，官方标注 grace 政策内） |

**时点说明（2026-10-07 复核）**：官方挑战窗口为 2026-03-18 ～ 04-30；上表 1.0810 一行是 **2026-04-09 时点**的榜首，其后两周记录密集刷新，最终榜首为 **1.0565**（codemath3000，PR#2135，2026-05-01，官方标注处于 grace 政策下，3-seed mean 1.05651，p=0.014 vs PR#2014）。本表按时间顺序完整保留全部记录，**引用时不要把 1.0810 当作当前 SOTA**；§2 的 Level 阈值与「Elite 20 建议线」仍按 2026-04-09 时点榜单估算，实际入选难度已明显提高。

---

## 2. 各 Level 阈值与目标对照

| 关卡 | BPB 要求 | 状态/我的目标 |
|------|---------|--------------|
| Level 1（Bronze） | 复现 **1.2244** (±0.005) | 🎯 目标 E001 |
| Level 2（Silver） | **< 1.18** | 🎯 主线 E003/E004（SP8192 预期 ~1.085–1.09，远超门槛） |
| Level 3（Gold，前 50） | **< 1.12** | 🎯 进阶 E005/E006 |
| Level 4（Platinum，前 20） | **< 1.085** | 路线预留（**2026-10-07 复核：官方榜首已为 1.0565**，1.085 仅相当于 2026-04-09 时点水平，见 §1 时点说明） |
| Elite 20 核心候选建议线 | < 1.15 | 由 SP8192 单点即可达到 |

> 关键观察：**仅将 tokenizer 从 SP-1024 升到 SP8192（baseline→PR#1394），真实降幅即达 −0.1388 BPB（1.2244→1.0856）**，一步跨过 Level 2/Level 3 门槛。这印证了方案设计「先做最大确定杠杆」的选型。

---

## 3. 我的目标行（计划态，待 GPU 实测回填）

> 以下为**计划目标值**，非实测。领取算力并在 8×H100 跑通后，将真实 BPB 填入 `JiaJingwen_C2G_submission.json`。

| 我的实验 | 配置 | 目标 BPB | 对应关卡 |
|---------|------|---------|---------|
| E001 | baseline（SP-1024） | 1.2244（复现） | Level 1 |
| E003 | + SP8192 | ~1.085–1.09 | Level 2/3 |
| E004 | SP8192 + 超参微调 | < 1.18（保底）/ 争取 <1.12 | Level 2/3 |
| E005（进阶） | SP8192 + MuonEq-R | < 1.15 | Level 3 |
| E006（进阶） | SP8192 + 循环 + 并行残差 + TTT | < 1.12 | Level 3 |

---

## 4. 统计显著性说明（Level 3 门槛）

官方规则：新 record 须比当前 SOTA 低 **≥0.005 nats** 且 **p<0.01**（≥3 独立 seed）。当前 SOTA = **1.0565**（codemath3000，PR#2135，2026-05-01；3-seed mean 1.05651，p=0.014 vs PR#2014）；1.0810 仅为 2026-04-09 时点值。我的 E006 若冲击 Level 3，必须满足该显著性阈值，并在 `JiaJingwen_C2G_AAR.md` / 消融报告中给出 t 检验或 bootstrap 结果。

---

## 5. 数据可信度

- 所有「官方实测」数值直接来自 `openai/parameter-golf` 公开 `submission.json` 与 README，可复核。
- 本文件**无任何伪造 BPB**；第 3 节目标行为计划值，明确标注。
- **2026-10-07 复核**：对照官方 README 逐行重抓，补齐 2026-04-09 之后新增的 **16 条**记录（1.0798 → 1.0565），并修正全文 5 处把 1.0810 标为「当前 SOTA」的表述。
