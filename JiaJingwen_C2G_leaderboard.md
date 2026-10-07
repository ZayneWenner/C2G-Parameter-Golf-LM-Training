# JiaJingwen_C2G_leaderboard（BPB 对比表）

> 挑战：C2G 参数高尔夫
> 作者：贾静文（JiaJingwen）
> 日期：2026-10-07
> 数据来源：官方 `openai/parameter-golf` 仓库 `README.md` Leaderboard 与 `records/` 各 submission.json（**全部为官方实测值**）。
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
| — | **SP8192 + 3层循环 + 并行残差 + QK5.25 + TTT** | **1.0810** | bigbag | 2026-04-09 | **当前 SOTA** |

---

## 2. 各 Level 阈值与目标对照

| 关卡 | BPB 要求 | 状态/我的目标 |
|------|---------|--------------|
| Level 1（Bronze） | 复现 **1.2244** (±0.005) | 🎯 目标 E001 |
| Level 2（Silver） | **< 1.18** | 🎯 主线 E003/E004（SP8192 预期 ~1.085–1.09，远超门槛） |
| Level 3（Gold，前 50） | **< 1.12** | 🎯 进阶 E005/E006 |
| Level 4（Platinum，前 20） | **< 1.085** | 路线预留（当前 SOTA 1.0810） |
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

官方规则：新 record 须比当前 SOTA 低 **≥0.005 nats** 且 **p<0.01**（≥3 独立 seed）。当前 SOTA = 1.0810（bigbag，3 seed std 0.0002）。我的 E006 若冲击 Level 3，必须满足该显著性阈值，并在 `JiaJingwen_C2G_AAR.md` / 消融报告中给出 t 检验或 bootstrap 结果。

---

## 5. 数据可信度

- 所有「官方实测」数值直接来自 `openai/parameter-golf` 公开 `submission.json` 与 README，可复核。
- 本文件**无任何伪造 BPB**；第 3 节目标行为计划值，明确标注。
