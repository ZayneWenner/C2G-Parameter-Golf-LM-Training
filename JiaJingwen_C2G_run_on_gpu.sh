#!/usr/bin/env bash
# =============================================================================
# C2G 参数高尔夫 —— GPU 一键跑分（真实成绩采集）
# 用法（在租到的 GPU 机器上，仓库根目录）:  bash JiaJingwen_C2G_run_on_gpu.sh all
# 阶段:  preflight | data | probe | train | all
# 诚实原则: 只搬运真实跑出的数字; 任一步失败立即中止; 绝不补占位值.
# 每个数字的出处都是 logs/seed<SEED>.txt 里的一行原文.
# =============================================================================
set -euo pipefail

# ---- 必须显式提供的参数 ----
export HARDWARE_LABEL="${HARDWARE_LABEL:?请设置 HARDWARE_LABEL, 例: export HARDWARE_LABEL='1xRTX4090 24GB (non-record)'}"
export WORLD_SIZE="${WORLD_SIZE:-1}"                    # 只允许 1 / 2 / 4 / 8
export SEEDS="${SEEDS:-1337 1338 1339}"                 # 官方口径 >=3 个种子
export TARGET_TOKENS="${TARGET_TOKENS:-10500000000}"    # 官方默认 20000 iters x 524288 tok
export PROBE_SECONDS="${PROBE_SECONDS:-120}"

REPO="${REPO:-$PWD}"; cd "$REPO"
export DATA_PATH="${DATA_PATH:-$REPO/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO/data/tokenizers/fineweb_1024_bpe.model}"
OUT_METRICS="${OUT_METRICS:-$REPO/metrics.json}"
STAGE="${1:-all}"

say(){ printf '\n== %s ==\n' "$*"; }
die(){ printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

preflight(){
  say "STEP 0 预检"
  command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi：这台机器没有可见 GPU"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  [ -f "$REPO/train_gpt.py" ] || die "当前目录不是 parameter-golf 仓库根（缺 train_gpt.py）"
  case "$WORLD_SIZE" in
    1|2|4|8) echo "[i] WORLD_SIZE=$WORLD_SIZE -> grad_accum_steps=$((8/WORLD_SIZE))" ;;
    *) die "WORLD_SIZE=$WORLD_SIZE 非法：train_gpt.py 第 748-749 行要求 8 % WORLD_SIZE == 0，只允许 1/2/4/8 张卡" ;;
  esac
  python -c "import torch;print('torch',torch.__version__,'cuda_available',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())" \
    || die "torch 不可用：先 pip install -r requirements.txt"
  python -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)" \
    || die "torch.cuda.is_available()=False：train_gpt.py 第 752-753 行硬性 raise RuntimeError('CUDA is required')，没有 CPU 回退"
  python -c "import sentencepiece,tiktoken,numpy" || die "缺少 sentencepiece / tiktoken / numpy"
  # train_gpt.py 第 774 行写 logs/{RUN_ID}.txt，目录不存在会直接崩
  mkdir -p "$REPO/logs"
  df -h . | tail -1
  echo "[i] 数据/分词器将使用: $DATA_PATH  |  $TOKENIZER_PATH"
  say "预检通过"
}

fetch_data(){
  say "STEP 1 数据与分词器（官方 sp1024 导出）"
  if [ -f "$DATA_PATH/fineweb_val_000000.bin" ] && [ -f "$TOKENIZER_PATH" ]; then
    echo "[i] 已存在，跳过下载"; return 0
  fi
  [ -f "$REPO/data/cached_challenge_fineweb.py" ] || die "缺官方脚本 data/cached_challenge_fineweb.py"
  python "$REPO/data/cached_challenge_fineweb.py" --variant sp1024
  [ -f "$TOKENIZER_PATH" ] || die "分词器未落地: $TOKENIZER_PATH"
  ls -1 "$DATA_PATH" | head -3; echo "[i] train shards=$(ls -1 "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)"
  say "数据就绪"
}

probe(){
  say "STEP 2 吞吐探针（${PROBE_SECONDS}s，不产出成绩，只测 tokens/s）"
  # 用真实脚本跑一个短窗口，从日志推吞吐；不猜、不估。
  MAX_WALLCLOCK_SECONDS="$PROBE_SECONDS" ITERATIONS=999999 \
  RUN_ID="probe_ws${WORLD_SIZE}" SEED=1337 \
    python "$REPO/train_gpt.py" 2>&1 | tee "$REPO/logs_probe.txt" || true
  python - "$REPO/logs_probe.txt" "$TARGET_TOKENS" "$WORLD_SIZE" <<'PY'
import re,sys,math
txt=open(sys.argv[1],encoding="utf-8",errors="replace").read()
target=float(sys.argv[2]); ws=float(sys.argv[3])
tps=[(float(m.group(1)),float(m.group(2))) for m in re.finditer(
    r"step:(\d+)/\d+ train_loss:.*?step_avg:([\d.]+)ms", txt)]
if not tps: print("[!] 探针日志里没有 step 行，无法测吞吐（检查是否 OOM 或提前崩）"); sys.exit(0)
step,step_avg=tps[-1]
tok_per_step=524288.0
per_s=tok_per_step/(step_avg/1000.0)
hrs=target/per_s/3600.0
print("[i] 实测 step_avg=%.2f ms -> 全局 %.0f tok/s (WORLD_SIZE=%d)"%(step_avg,per_s,ws))
print("[i] 达到 TARGET_TOKENS=%.3g 预计需要 %.1f 小时"%(target,hrs))
if hrs>6: print("[!] 超过 6 小时：建议降低 TARGET_TOKENS / ITERATIONS，或换更多卡（WORLD_SIZE 只能取 1/2/4/8）")
PY
}

train_seeds(){
  say "STEP 3 正式跑分 × 每种子一次"
  mkdir -p "$REPO/logs"
  # 单卡时全局 batch 仍是 524288 tok（local = global/8），轨迹与 8 卡一致，只是墙钟更长。
  for s in $SEEDS; do
    echo "---- SEED=$s ----"
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" ITERATIONS="${ITERATIONS:-20000}" \
    RUN_ID="seed$s" SEED="$s" \
      python "$REPO/train_gpt.py" 2>&1 | tee "$REPO/logs/seed$s.txt"
    grep -q "final_int8_zlib_roundtrip_exact" "$REPO/logs/seed$s.txt" \
      || die "seed $s 未跑完（日志里没有 final_int8_zlib_roundtrip_exact），不生成 metrics.json"
  done
  say "全部种子跑完"
}

harvest(){
  say "STEP 4 汇总 -> metrics.json（逐字取自日志）"
  python - "$REPO" "$OUT_METRICS" "$HARDWARE_LABEL" "$SEEDS" <<'PY'
import io,json,os,re,statistics,sys
repo,out,hw,seeds=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4].split()
res={}
for s in seeds:
    p=os.path.join(repo,"logs","seed%s.txt"%s)
    if not os.path.exists(p): raise SystemExit("缺日志: %s"%p)
    t=io.open(p,encoding="utf-8",errors="replace").read()
    def one(pat,cast=float,req=True):
        m=re.search(pat,t)
        if not m:
            if req: raise SystemExit("日志 %s 缺少字段: %s"%(p,pat))
            return None
        return cast(m.group(1))
    res[s]={
      "val_bpb": one(r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)"),
      "val_loss": one(r"final_int8_zlib_roundtrip val_loss:([\d.]+)"),
      "train_ms": one(r"step:\d+/\d+ train_loss:.*?train_time:(\d+)ms",int),
      "eval_ms": one(r"final_int8_zlib_roundtrip .*?eval_time:(\d+)ms",int),
      "model_bytes": one(r"Serialized model int8\+zlib: (\d+) bytes",int),
      "code_bytes": one(r"Code size: (\d+) bytes",int),
      "total_bytes": one(r"Total submission size int8\+zlib: (\d+) bytes",int),
      "peak_mib": one(r"peak memory allocated: (\d+) MiB",int,False),
      "world_size": one(r"world_size:(\d+)",int,False),
    }
py=[res[s]["val_bpb"] for s in seeds]
codes={res[s]["code_bytes"] for s in seeds}
totals={res[s]["total_bytes"] for s in seeds}
if len(codes)!=1 or len(totals)!=1:
    raise SystemExit("[!] 各种子的包体字节数不一致: totals=%s codes=%s"%(totals,codes))
m={
 "name":"SP8192 + single-point tokenizer upgrade",
 "protocol":"10min_16mb",
 "val_bpb":round(statistics.fmean(py),8),
 "val_bpb_std":round(statistics.pstdev(py),8),
 "bytes_total":totals.pop(),
 "bytes_code":codes.pop(),
 "seeds":[int(s) for s in seeds],
 "seed_results":{s:{"val_bpb":res[s]["val_bpb"],"val_loss":res[s]["val_loss"]} for s in seeds},
 "hardware":hw,
 "pytorch_version":__import__("torch").__version__,
 "train_seconds":round(max(res[s]["train_ms"] for s in seeds)/1000.0,3),
 "eval_seconds":round(max(res[s]["eval_ms"] for s in seeds)/1000.0,3),
 "technique_summary":"BPE-1024 -> SentencePiece-8192 tokenizer upgrade (primary lever). See JiaJingwen_C2G_方案设计.md E003.",
}
json.dump(m,io.open(out,"w",encoding="utf-8",newline="\n"),ensure_ascii=False,indent=2)
print(io.open(out,encoding="utf-8").read())
print("[i] 已写入 %s  —— 每个数字都能在 logs/seed*.txt 里找到原句"%out)
PY
}

backfill_pack(){
  say "STEP 5 回填 + 打包（在交付目录执行）"
  cat <<'EOS'
把 metrics.json 拷到交付目录后依次执行：
  python JiaJingwen_C2G_score_backfill.py backfill metrics.json --dry-run   # 先空跑确认
  python JiaJingwen_C2G_score_backfill.py backfill metrics.json             # 真写入 submission.json
  python JiaJingwen_C2G_pack_artifact.py check --run-dir <本次 run 目录>      # 打包前合规校验
  python JiaJingwen_C2G_pack_artifact.py pack  --run-dir <本次 run 目录>      # 产出 16MB artifact
EOS
}

case "$STAGE" in
  preflight) preflight ;;
  data)      fetch_data ;;
  probe)     preflight; probe ;;
  train)     train_seeds; harvest ;;
  all)       preflight; fetch_data; probe; train_seeds; harvest; backfill_pack ;;
  *)         die "未知阶段 '$STAGE'（可用: preflight|data|probe|train|all）" ;;
esac
