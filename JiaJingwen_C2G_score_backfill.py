#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2G 参数高尔夫 —— 成绩回填工具（诚实模式 / honest mode）

本工具**绝不编造成绩**。它只做一件事：把一次**真实 GPU 跑分**产生的数字，
从 metrics.json 原样搬进 submission.json。任何必填项缺失或不合规，
它直接拒绝并**不碰** submission.json（退出码 2）。

用法:
  python JiaJingwen_C2G_score_backfill.py check
  python JiaJingwen_C2G_score_backfill.py backfill metrics.json              # 真写入 submission.json（自动备份 .bak）
  python JiaJingwen_C2G_score_backfill.py backfill metrics.json --dry-run    # 只打印，不写盘
  python JiaJingwen_C2G_score_backfill.py backfill metrics.json --out /tmp/x.json

metrics.json 需在**真跑过的那台 GPU 机器上**生成（每个 seed 一条）:
{
  "hardware": "8xH100 80GB SXM",
  "pytorch_version": "2.5.0+cu124",
  "protocol": "10min_16mb",
  "train_seconds": 480, "eval_seconds": 60,
  "seeds": [0, 1, 2],
  "seed_results": {"0": {"val_bpb": 1.0809, "val_loss": 1.826}, "1": {...}, "2": {...}},
  "val_bpb": 1.0810,        # >=3 seed 均值，final int8+zlib roundtrip 指标
  "val_bpb_std": 0.0002,
  "bytes_total": 15863489,  # 提交包 artifact 字节数，必须 <= 16000000
  "bytes_code": null,       # 可留空 -> 自动按本目录 *.py 求和
  "name": "SP8192 + QK-Gain 5.0", "technique_summary": "..."
}
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "JiaJingwen_C2G_submission.json"
ART_CAP = 16_000_000   # artifact 上限（字节）
SEC_CAP = 600          # 训练 / 评测各 10 分钟


def say(*a):
    print(*a)


def sh(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, ((p.stdout or "") + (p.stderr or ""))
    except Exception as e:  # 命令不存在 / 超时
        return 127, "%s: %s" % (type(e).__name__, e)


def cmd_check():
    """在跑之前先体检：GPU / torch / tokenizer / 数据 / submission.json。"""
    ok = {}
    rc, txt = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    gpus = [l.strip() for l in txt.splitlines() if l.strip() and "," in l]
    ok["gpu"] = rc == 0 and len(gpus) > 0
    say("[gpu] rc=%s count=%d %s" % (rc, len(gpus), gpus or "--> NO GPU VISIBLE"))

    rc, txt = sh([sys.executable, "-c",
                  "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"])
    ok["torch"] = rc == 0
    say("[torch] " + (txt.strip().splitlines()[-1] if txt.strip() else "not installed"))

    rc, txt = sh([sys.executable, "-c", "import sentencepiece;print(sentencepiece.__version__)"])
    ok["sentencepiece"] = rc == 0
    say("[tokenizer-lib] sentencepiece " + (txt.strip().splitlines()[-1] if ok["sentencepiece"] else "MISSING"))

    models = glob.glob(str(ROOT / "**" / "*.model"), recursive=True)
    say("[tokenizer-file] %d 个 .model: %s" % (len(models), models[:3]))
    tr = glob.glob(str(ROOT / "**" / "fineweb_train_*.bin"), recursive=True)
    va = glob.glob(str(ROOT / "**" / "fineweb_val_*.bin"), recursive=True)
    ok["data"] = bool(tr) and bool(va)
    say("[data] train shards=%d val shards=%d  (各需 >=1)" % (len(tr), len(va)))

    ok["submission_json"] = SUB.exists()
    say("[submission.json] %s" % ("present" if ok["submission_json"] else "MISSING"))
    ready = all(ok.values())
    say("\nready_for_training: %s" % str(ready).lower())
    if not ready:
        say("blocking: " + ", ".join(k for k, v in ok.items() if not v))
    return 0


def _py_code_bytes():
    """bytes_code 按本目录 *.py 实测求和（可用 metrics.json 的 bytes_code 覆盖）。"""
    tot, brk = 0, []
    for p in sorted(ROOT.glob("*.py")):
        n = p.stat().st_size
        tot += n
        brk.append("%s=%d" % (p.name, n))
    return tot, brk


def _validate(m):
    """只接受真实跑分。任一项不成立即返回错误列表，绝不猜、不补、不四舍五入。"""
    errs = []
    vb = m.get("val_bpb")
    if isinstance(vb, bool) or not isinstance(vb, (int, float)) or not (0.0 < float(vb) < 5.0):
        errs.append("val_bpb 必须是真实测得的数值且落在 (0,5)，当前=%r" % (vb,))
    bt = m.get("bytes_total")
    if isinstance(bt, bool) or not isinstance(bt, int) or not (0 < bt <= ART_CAP):
        errs.append("bytes_total 必须是真实整数且落在 (0,%d]，当前=%r" % (ART_CAP, bt))
    seeds = m.get("seeds")
    if not isinstance(seeds, list) or len({str(s) for s in seeds}) < 3:
        errs.append("seeds 必须列出 >=3 个真实 seed，当前=%r" % (seeds,))
    sr = m.get("seed_results")
    if sr is not None and (not isinstance(sr, dict) or len(sr) < 3):
        errs.append("seed_results 若给出则必须 >=3 条，当前 %d 条" % len(sr or {}))
    if not m.get("hardware"):
        errs.append("hardware 必填（写实际跑分的那台机器，不能用 planned）")
    for k in ("train_seconds", "eval_seconds"):
        v = m.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            errs.append("%s 必填且为真实测得秒数，当前=%r" % (k, v))
    return errs


def _build(m):
    code_tot, brk = _py_code_bytes()
    bt = int(m["bytes_total"])
    seeds = list(m["seeds"])
    ts, es = float(m["train_seconds"]), float(m["eval_seconds"])
    std = m.get("val_bpb_std")
    base = None
    if SUB.exists():
        try:
            base = json.loads(SUB.read_text(encoding="utf-8")).get("baseline_reference")
        except Exception:
            base = None
    doc = {
        "_note": ("REAL measured values, copied verbatim from metrics.json by "
                  "JiaJingwen_C2G_score_backfill.py. Nothing here is estimated, "
                  "rounded up, or fabricated."),
        "author": "JiaJingwen",
        "github_id": "ZayneWenner",
        "name": m.get("name", "SP8192 + single-point tokenizer upgrade"),
        "track": m.get("protocol", "10min_16mb"),
        "val_bpb": round(float(m["val_bpb"]), 7),
        "val_bpb_std": (round(float(std), 7)
                        if isinstance(std, (int, float)) and not isinstance(std, bool) else None),
        "bytes_total": bt,
        "bytes_code": int(m.get("bytes_code") or code_tot),
        "seeds": seeds,
        "seed_results": m.get("seed_results") or {},
        "hardware": m["hardware"],
        "pytorch_version": m.get("pytorch_version", "TBD"),
        "train_seconds": ts,
        "eval_seconds": es,
        "technique_summary": m.get("technique_summary", "See JiaJingwen_C2G_方案设计.md E003."),
        "compliance": {
            "train_under_600s": ts <= SEC_CAP,
            "artifact_under_16mb": bt <= ART_CAP,
            "eval_under_600s": es <= SEC_CAP,
            "three_seeds": len(seeds) >= 3,
        },
        "baseline_reference": base,
    }
    return doc, brk


def cmd_backfill(args):
    src = Path(args.metrics)
    if not src.exists():
        say("REFUSED: metrics 文件不存在 -> %s" % src)
        say("先在真跑过的那台 GPU 机器上生成 metrics.json（格式见本脚本头部注释）。")
        return 2
    m = json.loads(src.read_text(encoding="utf-8"))
    errs = _validate(m)
    if errs:
        say("REFUSED: 数据不全或不合规，submission.json 未被改动。")
        for e in errs:
            say("  - " + e)
        return 2
    doc, brk = _build(m)
    say("code_bytes_breakdown: " + ", ".join(brk))
    say("val_bpb=%s +- %s  bytes_total=%s  seeds=%s"
        % (doc["val_bpb"], doc["val_bpb_std"], doc["bytes_total"], doc["seeds"]))
    for k, v in doc["compliance"].items():
        say("  compliance.%s = %s" % (k, v))
    if not all(doc["compliance"].values()):
        say("NOTE: 存在 false 的合规项 -> 该结果只能进官方 Unlimited-Compute / non-record 通道，不能算 10min_16mb 记录。")
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    if args.dry_run:
        say("DRY_RUN -- 以下为将写入的内容，未落盘:\n" + text)
        return 0
    tgt = Path(args.out) if args.out else SUB
    if tgt.exists() and not args.out:
        shutil.copyfile(str(tgt), str(tgt) + ".bak")
        say("backup -> %s.bak" % tgt.name)
    tgt.write_text(text, encoding="utf-8")
    back = json.loads(tgt.read_text(encoding="utf-8"))  # 回读校验，确认真的落地
    same = (back.get("val_bpb") == doc["val_bpb"] and back.get("bytes_total") == doc["bytes_total"])
    say("written -> %s (%d bytes)  readback_match=%s" % (tgt, tgt.stat().st_size, same))
    return 0 if same else 6


def main():
    ap = argparse.ArgumentParser(description="C2G score backfill (honest mode)")
    ap.add_argument("mode", choices=["check", "backfill"])
    ap.add_argument("metrics", nargs="?", default="metrics.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    return cmd_check() if a.mode == "check" else cmd_backfill(a)


if __name__ == "__main__":
    sys.exit(main())
