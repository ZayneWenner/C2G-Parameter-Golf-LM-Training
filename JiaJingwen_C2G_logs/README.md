# JiaJingwen_C2G_logs —— 训练日志目录

> 作者：贾静文（JiaJingwen）｜ 挑战：C2G 参数高尔夫 ｜ 更新：2026-10-07

## 当前状态（诚实声明）

**本目录目前只有规范与模板，没有任何真实训练日志。** 本机无 GPU（`nvidia-smi` 127）、无 torch、无 FineWeb 预分词数据，一次训练都没跑过。`metrics.json` 中所有数值位为 `null` —— 回填工具会（正确地）拒绝这种不全的数据，因此不会出现"空值被写进成绩"的情况。

## 目录内容

| 文件 | 作用 |
|---|---|
| `README.md` | 本规范：命名、必采字段、汇总口径 |
| `metrics.json.template` | 汇总模板，字段与 `JiaJingwen_C2G_score_backfill.py` 严格对齐 |
| `seed00_train.log.template` | 单 seed 日志模板，展示必须保存的内容与格式 |

## 命名规范

```
E<实验号>_seed<种子>_<配置简称>.log
例：E003_seed0_sp8192.log
    E001_seed0_baseline_bpe1024.log
```
- 一个 seed 一个文件，**不合并**；
- 文件名里的配置简称必须与 `JiaJingwen_C2G_ablation.md` §1 实验矩阵一致；
- 训练与评测分两个文件时，后缀 `_train.log` / `_eval.log`。

## 每个 log 必须包含（缺一项该次实验不可复现）

1. **完整命令行**（`torchrun --nproc_per_node=... train_gpt.py --config ...`）；
2. **代码版本**（`git rev-parse HEAD` 输出，以及 `train_gpt.py` 的 sha256）；
3. **硬件**（GPU 型号与显存、卡数，例如 `8xH100 80GB SXM`）；
4. **seed 值**；
5. **训练 wallclock 秒数**（必须 < 600s 才有资格进 10min 记录；建议压到 480s 留缓冲）；
6. **评测 wallclock 秒数**（同样 < 600s）；
7. **final val_bpb**（official eval 口径，final int8+zlib roundtrip）；
8. **artifact 字节数**（必须 ≤ 16,000,000）；
9. **训练期诊断要点**：loss 曲线、grad norm、tokens/sec 的关键节点（用于判断是否稳定、是否过拟合）；
10. **异常与告警原文**（OOM、超时、模块版本冲突等，不删不改）。

## 汇总口径

1. 先有 log，后有 `metrics.json` —— **不许反过来**：不许先写一个想要的数字再补 log。
2. `metrics.json` 的 `seeds` 至少 3 个，且 `seed_results` 与 log 一一对应；
3. 均值与 std 从逐 seed 结果算出，保留到 log 给出的有效位，不顺手上调；
4. log 定稿后**不事后编辑**；需要修正就新增文件并写明原因。

## 从 log 到成绩（回填命令）

```bash
# 1) 先体检：GPU / torch / tokenizer / 数据 / submission.json
python JiaJingwen_C2G_score_backfill.py check

# 2) 只打印不落盘，确认数字无误
python JiaJingwen_C2G_score_backfill.py backfill metrics.json --dry-run

# 3) 真写入（自动备份 .bak，并回读校验）
python JiaJingwen_C2G_score_backfill.py backfill metrics.json
```

工具行为保证：必填项缺失或不合规（`val_bpb` 不在 (0,5)、`bytes_total` 超 16MB、seed 少于 3 个、`hardware` 为空、`train_seconds`/`eval_seconds` 非正数）→ **拒绝写入并以退出码 2 结束，`submission.json` 不被改动**。

## 单卡 / 非 8×H100 的跑法

若最终用的是单卡或消费级卡，**照样按本规范留 log**，但：
- `hardware` 字段如实填写（例如 `1xA100-40G (non-record)`），不得写成 8×H100；
- 回填后 `compliance` 里会出现 `false`，工具会提示"该结果只能进官方 Unlimited-Compute / non-record 通道"——这是如实标注，不是失败。
