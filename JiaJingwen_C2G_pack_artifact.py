#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JiaJingwen_C2G_pack_artifact.py
C2G 参数高尔夫 —— 16MB artifact 打包与合规校验（纯标准库，不联网、不训练、不造数）

为什么需要它：
  挑战正文 line 301-303 把 `姓名_C2G_submission.tar.gz`（含代码 + 压缩权重 + tokenizer）
  列为「必须提交的文件」，但官方 `train_gpt.py` 只写出 `final_model.int8.ptz`，
  脚本里没有任何 tarfile / gzip 打包代码（已实测复核：tarfile=0、gzip=0 次出现）。
  所以打包这一步必须自己做——本脚本只做这一步。

用法：
  python JiaJingwen_C2G_pack_artifact.py check [--run-dir DIR] [--json]
  python JiaJingwen_C2G_pack_artifact.py pack  [--run-dir DIR] [--json]

  可选：--log PATH 指定训练日志；--tokenizer PATH 指定 tokenizer；
        --out-dir DIR 指定产物目录（默认交付目录）；--allow-no-log-proof 跳过日志取证。

合规判据（全部来自挑战正文，不自行发明）：
  line 44  : 模型文件上限 16 MB (16,000,000 字节)，含代码 + 压缩权重 + tokenizer
  line 439 : 超 16MB 却没发现 -> 提交时被拒
  脚本自报 : "Total submission size int8+zlib: N bytes"（N = 代码 + int8+zlib 权重）
  两条线都查，任一越界即拒绝——宁可拒，不可交出超限或来源不明的包。

退出码：0 = 通过；2 = 拒绝（缺件 / 超限 / 无真实跑分取证）
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile

LIMIT_BYTES = 16_000_000
CHALLENGE_ID = "ch-20260717031359-b8wyg0"
REPO_URL = "https://github.com/ZayneWenner/C2G-Parameter-Golf-LM-Training"
CODE_NAME = "JiaJingwen_C2G_train_gpt.py"
WEIGHTS_NAME = "final_model.int8.ptz"
TOKENIZER_NAME = "fineweb_1024_bpe.model"
ARTIFACT_NAME = "JiaJingwen_C2G_submission.tar.gz"

RE_TOTAL = re.compile(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes")
RE_BPB = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.eE+-]+)\s+val_bpb:([0-9.eE+-]+)"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def human(n):
    return "%d bytes (%.2f MB)" % (n, n / 1000000.0)


def find_first(root, name):
    for dirpath, _dirnames, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def find_log(root):
    if not os.path.isdir(root):
        return None
    hits = []
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".log", ".txt")) and "log" in fn.lower():
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)[0] if hits else None


def parse_log(path):
    """从真实训练日志里取证：官方脚本自报的总体积 + int8 往返后的 val_bpb。"""
    if not path or not os.path.isfile(path):
        return {"ok": False, "reason": "log_not_found", "log": path}
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    m_total = RE_TOTAL.findall(txt)
    m_bpb = RE_BPB.findall(txt)
    if not m_total or not m_bpb:
        return {
            "ok": False,
            "reason": "log_missing_proof_lines",
            "has_total_line": bool(m_total),
            "has_bpb_line": bool(m_bpb),
            "log": path,
        }
    loss, bpb = m_bpb[-1]
    return {
        "ok": True,
        "log": path,
        "script_reported_total_bytes": int(m_total[-1]),
        "val_loss": float(loss),
        "val_bpb": float(bpb),
    }


def build_targz(members):
    """确定性打包：成员排序、mtime=0、uid/gid=0、gzip mtime=0，同输入必然同字节。"""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for arcname, src in sorted(members):
            ti = tf.gettarinfo(src, arcname=arcname)
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(src, "rb") as fh:
                tf.addfile(ti, fh)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("mode", choices=["check", "pack"])
    ap.add_argument("--run-dir", default=None,
                    help="训练跑分目录（含 final_model.int8.ptz 与日志）")
    ap.add_argument("--log", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out-dir", default=None, help="产物目录，默认交付目录")
    ap.add_argument("--allow-no-log-proof", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    deliv = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else deliv
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else deliv

    res = {"mode": args.mode, "run_dir": run_dir, "limit_bytes": LIMIT_BYTES,
           "challenge_id": CHALLENGE_ID, "problems": [], "warnings": []}

    code_path = os.path.join(deliv, CODE_NAME)
    weights_path = os.path.join(run_dir, WEIGHTS_NAME)
    if not os.path.isfile(weights_path):
        weights_path = find_first(run_dir, WEIGHTS_NAME) or weights_path
    tok_path = args.tokenizer or find_first(run_dir, TOKENIZER_NAME) or \
        os.path.join(run_dir, "data", "tokenizers", TOKENIZER_NAME)

    for label, p in (("code", code_path), ("weights", weights_path), ("tokenizer", tok_path)):
        if not os.path.isfile(p):
            res["problems"].append("missing_%s: %s" % (label, p))

    proof = parse_log(args.log or find_log(run_dir))
    res["proof"] = proof
    if not proof.get("ok"):
        msg = "no_real_run_proof (%s)" % proof.get("reason")
        if args.allow_no_log_proof:
            res["warnings"].append(msg + " -- 已按 --allow-no-log-proof 放行")
        else:
            res["problems"].append(msg)

    if not res["problems"]:
        code_bytes = os.path.getsize(code_path)
        w_bytes = os.path.getsize(weights_path)
        t_bytes = os.path.getsize(tok_path)
        res["sizes"] = {"code_bytes": code_bytes, "weights_bytes": w_bytes,
                        "tokenizer_bytes": t_bytes,
                        "script_reported_total_bytes": proof.get("script_reported_total_bytes")}
        if proof.get("script_reported_total_bytes") is not None and \
                proof["script_reported_total_bytes"] > LIMIT_BYTES:
            res["problems"].append("script_reported_total_over_limit: %d > %d" % (
                proof["script_reported_total_bytes"], LIMIT_BYTES))
        if code_bytes + w_bytes > LIMIT_BYTES:
            res["problems"].append("code_plus_weights_over_limit: %d > %d" % (
                code_bytes + w_bytes, LIMIT_BYTES))

    if not res["problems"] and args.mode == "pack":
        manifest = os.path.join(run_dir, "_MANIFEST.txt")
        with io.open(manifest, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("challenge: %s\nrepo: %s\n" % (CHALLENGE_ID, REPO_URL))
            fh.write("train_script: %s (%d bytes, sha256=%s)\n" % (
                CODE_NAME, res["sizes"]["code_bytes"], sha256_file(code_path)))
            fh.write("weights: %s (%d bytes)\n" % (WEIGHTS_NAME, res["sizes"]["weights_bytes"]))
            fh.write("tokenizer: %s (%d bytes)\n" % (TOKENIZER_NAME, res["sizes"]["tokenizer_bytes"]))
            fh.write("script_reported_total_bytes: %d\n" % res["sizes"]["script_reported_total_bytes"])
            fh.write("val_bpb: %.8f\n" % proof["val_bpb"])
            fh.write("limit_bytes: %d\n" % LIMIT_BYTES)
            fh.write("generated_by: JiaJingwen_C2G_pack_artifact.py\n")
            fh.write("note: 本包由真实训练产物打包，未做任何数值臆测。\n")
        blob = build_targz([(CODE_NAME, code_path), (WEIGHTS_NAME, weights_path),
                            (TOKENIZER_NAME, tok_path), ("MANIFEST.txt", manifest)])
        if len(blob) > LIMIT_BYTES:
            res["problems"].append("artifact_over_limit: %d > %d" % (len(blob), LIMIT_BYTES))
        else:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, ARTIFACT_NAME)
            with open(out_path, "wb") as fh:
                fh.write(blob)
            got = os.path.getsize(out_path)
            with tarfile.open(out_path, mode="r:gz") as tf:
                names = sorted(tf.getnames())
            res["artifact"] = {"path": out_path, "bytes": got, "human": human(got),
                               "sha256": sha256_file(out_path), "members": names,
                               "headroom_bytes": LIMIT_BYTES - got}
            if got != len(blob):
                res["problems"].append("write_verify_mismatch")

    res["ok"] = not res["problems"]
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("mode=%s  run_dir=%s" % (args.mode, run_dir))
        print("limit=%s" % human(LIMIT_BYTES))
        for k in ("code_bytes", "weights_bytes", "tokenizer_bytes",
                  "script_reported_total_bytes"):
            if "sizes" in res:
                print("  %-28s %s" % (k, res["sizes"][k]))
        if proof.get("ok"):
            print("  proof: val_bpb=%.8f (log %s)" % (proof["val_bpb"], proof["log"]))
        for w in res["warnings"]:
            print("  WARN  %s" % w)
        for p in res["problems"]:
            print("  REJECT  %s" % p)
        if "artifact" in res:
            a = res["artifact"]
            print("  artifact: %s" % a["human"])
            print("  sha256:   %s" % a["sha256"])
            print("  headroom: %d bytes" % a["headroom_bytes"])
            print("  members:  %s" % ", ".join(a["members"]))
        print("RESULT: %s" % ("OK" if res["ok"] else "REJECTED"))
    sys.exit(0 if res["ok"] else 2)


if __name__ == "__main__":
    main()
