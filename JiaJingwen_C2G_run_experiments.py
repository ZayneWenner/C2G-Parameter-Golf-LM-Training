#!/usr/bin/env python3
"""
JiaJingwen C2G Parameter Golf — 真实训练-采集-提交流水线

用途：在有 GPU 的机器上一键完成
  1) 环境/数据检查  2) 跑 baseline + SP8192（各 3 seeds）
  3) 解析日志生成 metrics.json  4) 回填 submission.json
  5) 打包 submission.tar.gz

设计原则：
- 当前环境无 GPU，所以本脚本不在本地训练，而是生成到 GPU 机器执行。
- 默认走 Non-record 单卡路径（RTX 4090 / 1xH100），训练时间可能超过 10min，
  但 honest 标注 non-record 后仍可拿到真实 BPB 分数。
- 所有敏感占位（hardware / pytorch_version / val_bpb）均由实际运行回填，
  脚本拒绝写入虚假数据。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# -----------------------------
# USER CONFIG — 按需修改
# -----------------------------

REPO_URL = "https://github.com/openai/parameter-golf.git"
REPO_DIR = Path("openai-parameter-golf")          # 官方仓库克隆目录
OFFICIAL_TRAIN_GPT = REPO_DIR / "train_gpt.py"     # 官方入口（会被覆盖为我们的提交脚本）

MY_TRAIN_GPT = Path("JiaJingwen_C2G_train_gpt.py") # 你的提交脚本（与本文档同目录）
SUBMISSION_JSON = Path("JiaJingwen_C2G_submission.json")
README_MD = Path("JiaJingwen_C2G_README.md")

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "E001_baseline_sp1024",
        "env": {
            "VOCAB_SIZE": "1024",
            "TOKENIZER_PATH": str(REPO_DIR / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
            "DATA_PATH": str(REPO_DIR / "data" / "datasets" / "fineweb10B_sp1024"),
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "0",   # non-record：不掐 10min
        },
        "data_variant": "sp1024",
        "seeds": [1337, 1338, 1339],
    },
    {
        "name": "E003_sp8192",
        "env": {
            "VOCAB_SIZE": "8192",
            "TOKENIZER_PATH": str(REPO_DIR / "data" / "tokenizers" / "fineweb_8192_bpe.model"),
            "DATA_PATH": str(REPO_DIR / "data" / "datasets" / "fineweb10B_sp8192"),
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "0",
        },
        "data_variant": "sp8192",
        "seeds": [1337, 1338, 1339],
    },
]


def log(msg: str) -> None:
    print(f"[C2G] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def run(cmd: list[str] | str, env: dict[str, str] | None = None, cwd: Path | None = None,
        check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """运行 shell 命令并打印；超时与失败会抛出异常。"""
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    log(f"RUN: {' '.join(cmd)}")
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd, env=merged_env, cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )


def check_gpu() -> dict[str, Any]:
    """检查 GPU 与 torch 是否可用，返回硬件信息。"""
    log("检查 GPU 环境...")
    try:
        out = run(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader"],
                  check=True)
        gpu = out.stdout.strip().split(", ")
        gpu_name, gpu_mem = gpu[0], gpu[1]
    except Exception as e:
        log(f"ERROR: nvidia-smi 失败 -> {e}")
        sys.exit(1)

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        if not cuda_available:
            log("ERROR: torch.cuda.is_available() == False")
            sys.exit(1)
    except ImportError:
        log("ERROR: torch 未安装")
        sys.exit(1)

    info = {
        "gpu_name": gpu_name,
        "gpu_memory": gpu_mem,
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
    }
    log(f"GPU={gpu_name} ({gpu_mem}), torch={torch_version}, cuda={cuda_version}")
    return info


def clone_or_check_repo() -> None:
    """克隆或检查官方仓库。"""
    if REPO_DIR.exists():
        log(f"官方仓库已存在: {REPO_DIR}")
        return
    log(f"克隆官方仓库到 {REPO_DIR} ...")
    run(["git", "clone", REPO_URL, str(REPO_DIR)])


def prepare_data(variant: str) -> None:
    """使用官方脚本下载/缓存指定 tokenizer 的 FineWeb 数据。"""
    data_path = REPO_DIR / "data" / "datasets" / f"fineweb10B_{variant}"
    tok_path = REPO_DIR / "data" / "tokenizers" / f"fineweb_{variant}_bpe.model"
    if data_path.exists() and tok_path.exists():
        log(f"数据已存在: {data_path}")
        return
    log(f"下载 FineWeb {variant} ...")
    script = REPO_DIR / "data" / "cached_challenge_fineweb.py"
    if not script.exists():
        log(f"ERROR: 官方数据脚本不存在: {script}")
        sys.exit(1)
    run([sys.executable, str(script), "--variant", variant], cwd=REPO_DIR, timeout=3600)


@dataclass
class RunResult:
    seed: int
    val_bpb: float | None
    val_loss: float | None
    train_time_ms: float | None
    quant_file_bytes: int | None
    code_bytes: int | None
    total_submission_bytes: int | None
    log_path: Path | None
    error: str | None = None


def install_my_train_script() -> None:
    """把你的 train_gpt.py 复制进官方仓库作为运行入口，并备份原文件。"""
    if not MY_TRAIN_GPT.exists():
        log(f"ERROR: 未找到你的训练脚本 {MY_TRAIN_GPT}")
        sys.exit(1)
    backup = OFFICIAL_TRAIN_GPT.with_suffix(".py.official_backup")
    if OFFICIAL_TRAIN_GPT.exists() and not backup.exists():
        shutil.copy2(OFFICIAL_TRAIN_GPT, backup)
        log(f"已备份官方 train_gpt.py -> {backup}")
    shutil.copy2(MY_TRAIN_GPT, OFFICIAL_TRAIN_GPT)
    log(f"已复制 {MY_TRAIN_GPT} -> {OFFICIAL_TRAIN_GPT}")


def run_single_experiment(exp: dict[str, Any], seed: int, output_root: Path) -> RunResult:
    """跑一次训练并解析日志。"""
    run_dir = output_root / f"{exp['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = {k: str(v) for k, v in exp["env"].items()}
    env["SEED"] = str(seed)
    env["RUN_ID"] = f"{exp['name']}_seed{seed}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # 复制代码到 run_dir，便于打包时自包含
    shutil.copy2(MY_TRAIN_GPT, run_dir / "train_gpt.py")

    cmd = [
        sys.executable, str(OFFICIAL_TRAIN_GPT),
    ]
    logfile = run_dir / f"{env['RUN_ID']}.log"

    try:
        log(f"开始训练 {exp['name']} seed={seed}，日志 -> {logfile}")
        proc = run(cmd, env=env, cwd=REPO_DIR, check=False, timeout=7200)
        with open(logfile, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        if proc.returncode != 0:
            err = f"训练进程退出码 {proc.returncode}"
            log(f"WARNING: {err}")
            return RunResult(seed=seed, log_path=logfile, error=err)
    except subprocess.TimeoutExpired as e:
        with open(logfile, "w", encoding="utf-8") as f:
            f.write(e.stdout or "")
        return RunResult(seed=seed, log_path=logfile, error=f"超时 (> {e.timeout}s)")

    return parse_log(logfile, seed)


def parse_log(logfile: Path, seed: int) -> RunResult:
    """从 train_gpt.py 日志中解析最终指标。"""
    text = logfile.read_text(encoding="utf-8")
    res = RunResult(seed=seed, log_path=logfile)

    # final_int8_zlib_roundtrip_exact val_loss:1.22436571 val_bpb:1.22436571
    m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.eE+\-.]+) val_bpb:([0-9.eE+\-.]+)", text)
    if m:
        res.val_loss = float(m.group(1))
        res.val_bpb = float(m.group(2))
    else:
        res.error = "未找到 final_int8_zlib_roundtrip_exact 行"

    # Total submission size int8+zlib: 15863489 bytes
    m = re.search(r"Total submission size int8\+zlib: (\d+) bytes", text)
    if m:
        res.total_submission_bytes = int(m.group(1))
    else:
        m = re.search(r"Serialized model int8\+zlib: (\d+) bytes", text)
        if m:
            res.quant_file_bytes = int(m.group(1))

    # Code size: 47642 bytes
    m = re.search(r"Code size: (\d+) bytes", text)
    if m:
        res.code_bytes = int(m.group(1))

    # 训练时间：取最后一条 val_bpb 日志的 train_time
    times = re.findall(r"step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+) train_time:(\d+)ms", text)
    if times:
        # 最终 val_bpb 应以 final_int8_zlib 为准；训练时间取最后一次 periodic val 或 final
        res.train_time_ms = float(times[-1][1])

    if res.val_bpb is None:
        res.error = "解析失败：未得到 val_bpb"
    return res


def compute_stats(results: list[RunResult]) -> dict[str, Any] | None:
    """计算多 seed 均值与标准差。"""
    bpbs = [r.val_bpb for r in results if r.val_bpb is not None]
    if len(bpbs) < 2:
        return None
    import statistics
    return {
        "mean_val_bpb": sum(bpbs) / len(bpbs),
        "std_val_bpb": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
        "min_val_bpb": min(bpbs),
        "max_val_bpb": max(bpbs),
        "n_seeds": len(bpbs),
    }


def build_metrics(results: dict[str, list[RunResult]], gpu_info: dict[str, Any]) -> dict[str, Any]:
    """生成 metrics.json。"""
    metrics: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "experiments": {},
    }
    for exp_name, runs in results.items():
        metrics["experiments"][exp_name] = {
            "runs": [asdict(r) for r in runs],
            "stats": compute_stats(runs),
        }
    return metrics


def assert_compliance(sub: dict[str, Any]) -> None:
    """提交前合规断言，失败则退出。"""
    errors = []
    if sub.get("hardware", "").startswith("8xH100"):
        errors.append("hardware 仍包含虚假的 8xH100 申报，请先真实运行或明确标注 non-record")
    for k in ["val_bpb", "val_bpb_std", "bytes_total", "bytes_code"]:
        if sub.get(k) is None:
            errors.append(f"{k} 为 null，请先跑训练")
    if not sub.get("seeds"):
        errors.append("seeds 为空")
    comp = sub.get("compliance", {})
    if not all(comp.get(k) for k in ["train_under_600s", "artifact_under_16mb", "eval_under_600s", "three_seeds"]):
        errors.append("compliance 未全部通过")
    if errors:
        log("提交前合规检查失败：")
        for e in errors:
            log(f"  - {e}")
        sys.exit(2)
    log("提交前合规检查通过")


def backfill_submission(metrics: dict[str, Any], track: str = "non_record_16mb") -> dict[str, Any]:
    """用 metrics.json 回填 submission.json。"""
    if not SUBMISSION_JSON.exists():
        log(f"ERROR: {SUBMISSION_JSON} 不存在")
        sys.exit(1)
    sub: dict[str, Any] = json.loads(SUBMISSION_JSON.read_text(encoding="utf-8"))

    # 选用 E003_sp8192 作为主提交结果；如未成功则回退到 E001_baseline
    best_exp = None
    for cand in ["E003_sp8192", "E001_baseline_sp1024"]:
        if cand in metrics["experiments"] and metrics["experiments"][cand]["stats"]:
            best_exp = cand
            break
    if best_exp is None:
        log("ERROR: 没有成功实验可供回填")
        sys.exit(1)

    stats = metrics["experiments"][best_exp]["stats"]
    runs = metrics["experiments"][best_exp]["runs"]

    sub["track"] = track
    sub["val_bpb"] = round(stats["mean_val_bpb"], 8)
    sub["val_bpb_std"] = round(stats["std_val_bpb"], 8)

    # bytes_total = code + compressed model；取各 run 最大值，确保不超 16MB
    total_bytes = [r.get("total_submission_bytes") for r in runs if r.get("total_submission_bytes")]
    code_bytes = [r.get("code_bytes") for r in runs if r.get("code_bytes")]
    if not total_bytes or not code_bytes:
        log("ERROR: 未解析到 bytes_total / bytes_code")
        sys.exit(1)
    sub["bytes_total"] = max(total_bytes)
    sub["bytes_code"] = max(code_bytes)

    sub["seeds"] = [r["seed"] for r in runs if r["val_bpb"] is not None]
    sub["seed_results"] = {
        str(r["seed"]): {
            "val_bpb": r["val_bpb"],
            "val_loss": r["val_loss"],
            "train_time_ms": r["train_time_ms"],
            "total_submission_bytes": r["total_submission_bytes"],
        }
        for r in runs if r["val_bpb"] is not None
    }

    # 硬件与合规：由真实运行决定
    gpu = metrics["gpu"]
    sub["hardware"] = f"{gpu['gpu_name']} {gpu['gpu_memory']} (non-record single GPU)"
    sub["pytorch_version"] = gpu["pytorch_version"]
    sub["non_record"] = True
    sub["compliance"] = {
        "train_under_600s": False,          # non-record 允许超 10min
        "artifact_under_16mb": sub["bytes_total"] <= 16_000_000,
        "eval_under_600s": False,           # 单卡 eval 通常也超；non-record 不计该条
        "three_seeds": len(sub["seeds"]) >= 3,
    }
    # 删除过时的模板注释字段
    sub.pop("_note", None)

    # 保存前先检查；non-record 不把 train_under_600s / eval_under_600s 当硬性失败
    if sub["bytes_total"] > 16_000_000:
        log(f"ERROR: artifact {sub['bytes_total']} > 16MB")
        sys.exit(1)
    if len(sub["seeds"]) < 3:
        log("WARNING: seed 数不足 3，建议补跑")

    SUBMISSION_JSON.write_text(json.dumps(sub, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"已回填 {SUBMISSION_JSON}")
    return sub


def package_submission(sub: dict[str, Any]) -> Path:
    """打包 submission.tar.gz。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tar_path = Path(f"JiaJingwen_C2G_submission_{timestamp}.tar.gz")
    log(f"打包提交包 -> {tar_path}")

    with tarfile.open(tar_path, "w:gz") as tar:
        # 必须文件
        tar.add(MY_TRAIN_GPT, arcname="train_gpt.py")
        tar.add(SUBMISSION_JSON, arcname="submission.json")
        if README_MD.exists():
            tar.add(README_MD, arcname="README.md")
        # 日志 + metrics
        if Path("metrics.json").exists():
            tar.add("metrics.json")
        # 把每个 seed 的日志也打包
        for logf in sorted(Path(".").glob("runs/*_seed*/**/*.log"), recursive=True):
            tar.add(logf)
    log(f"打包完成: {tar_path} ({tar_path.stat().st_size} bytes)")
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description="C2G Parameter Golf 真实训练流水线")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 E001 baseline，直接跑 E003 SP8192")
    parser.add_argument("--only-baseline", action="store_name", help="只跑 E001 baseline 验证环境")
    parser.add_argument("--dry-run", action="store_true", help="只检查环境，不启动训练")
    args = parser.parse_args()

    log("=== JiaJingwen C2G 真实训练流水线启动 ===")
    gpu_info = check_gpu()
    clone_or_check_repo()
    install_my_train_script()

    experiments = list(EXPERIMENTS)
    if args.skip_baseline:
        experiments = [e for e in experiments if e["name"] != "E001_baseline_sp1024"]
    if args.only_baseline:
        experiments = [e for e in experiments if e["name"] == "E001_baseline_sp1024"]

    for exp in experiments:
        prepare_data(exp["data_variant"])

    if args.dry_run:
        log("DRY RUN 完成，环境就绪，未启动训练")
        return

    output_root = Path("runs")
    output_root.mkdir(exist_ok=True)
    all_results: dict[str, list[RunResult]] = {}

    for exp in experiments:
        log(f"--- 实验 {exp['name']} ---")
        results: list[RunResult] = []
        for seed in exp["seeds"]:
            r = run_single_experiment(exp, seed, output_root)
            results.append(r)
            if r.error:
                log(f"seed={seed} 失败: {r.error}")
            else:
                log(f"seed={seed} val_bpb={r.val_bpb} bytes={r.total_submission_bytes}")
        all_results[exp["name"]] = results

    # 生成 metrics.json
    metrics = build_metrics(all_results, gpu_info)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log("已生成 metrics.json")

    # 回填 submission.json（默认 non_record_16mb）
    sub = backfill_submission(metrics, track="non_record_16mb")

    # 打包
    tar_path = package_submission(sub)

    log("=== 流水线完成 ===")
    log(f"提交包: {tar_path}")
    log(f"最终 BPB: {sub['val_bpb']} ± {sub['val_bpb_std']}")
    log(f"下一步：把 {SUBMISSION_JSON} 与 {tar_path} 上传到 GitHub PR 或挑战提交入口")


if __name__ == "__main__":
    main()
