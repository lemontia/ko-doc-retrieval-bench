"""기계 칸 — 모델마다 색인 시간 · 질문 하나 지연 · 메모리 최대 증가분 · 저장 공간을 같은 방법으로 잰다.

  python3 tools/machine.py [평가판] [모델 ...]      기본: 법령검색-v1 · bm25 gemma2 bge-m3 kure1 kure2

- 색인부터 다시: 모델마다 빈 캐시 폴더(BENCH_CACHE)로 bench.py run 을 돌린다(결과 줄은 results.jsonl 에 그대로 더해진다)
- 메모리: vm_stat 을 1초마다 읽어 실행 직전 3초 평균 대비 최대 증가분. wired(GPU·고정) 와 used(wired+active+압축) 둘 다
  — 10-15 PolyStrata 측정과 같은 방법. 프로세스 RSS 는 MPS 할당을 못 잡아서 쓰지 않는다
- 질문 하나 지연: bench.py 의 _latency (질문 10개를 하나씩, 코퍼스 전체 점수까지, 중앙값)
- 열 식히기: 모델 사이 120초 쉰다(10-15 연속 부하에서 클럭이 내려간 실측)
- 기계 상태(다른 앱·LM Studio 에 올라간 모델)를 줄마다 남긴다 — 다른 일을 다 내린 '단독'이 아니면 그렇게 적힌다
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "machine.jsonl"


def vm():
    t = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", t)[1])
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", t)[1]) * page / 2**30
    wired = g("Pages wired down")
    return {"wired": wired, "used": wired + g("Pages active") + g("Pages occupied by compressor")}


def state():
    lms = subprocess.run(["lms", "ps"], capture_output=True, text=True).stdout.strip().splitlines()[1:]
    top = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True).stdout.splitlines()
    top = sorted(((int(l.split(None, 1)[0]) / 2**20, l.split(None, 1)[1].rsplit("/", 1)[-1]) for l in top if l.strip()), reverse=True)[:5]
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    return {"vm": {k: round(v, 1) for k, v in vm().items()}, "lms_loaded": [l.split()[0] for l in lms if l.strip()],
            "top_rss_gb": [(round(a, 1), b) for a, b in top], "therm": therm}


def measure(suite, model):
    base = []
    for _ in range(3):
        base.append(vm())
        time.sleep(1)
    b = {k: sum(x[k] for x in base) / 3 for k in base[0]}
    peak, stop = dict(b), threading.Event()

    def sample():
        while not stop.is_set():
            v = vm()
            for k in peak:
                peak[k] = max(peak[k], v[k])
            time.sleep(1)

    th = threading.Thread(target=sample)
    th.start()
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="bench-cache-") as cache:
        p = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "bench.py"), "run", "--model", model, "--suite", suite],
                           env={**os.environ, "BENCH_CACHE": cache}, capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    th.join()
    if p.returncode:
        raise SystemExit(f"{model}: {p.stderr[-1500:]}")
    row = [json.loads(l) for l in (ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    row = [r for r in row if r["model"] == model and r["suite"] == suite][-1]
    m = row["meta"]
    return {"wall_seconds": round(wall, 1), "index_seconds": m.get("index_seconds"), "query_ms_median": m.get("query_ms_median"),
            "store_mb": m.get("store_mb_fp16") or m.get("store_mb"), "token_vectors": m.get("token_vectors"),
            "peak_wired_gb": round(peak["wired"] - b["wired"], 2), "peak_used_gb": round(peak["used"] - b["used"], 2),
            "r@10": row["metrics"]["전체"]["r@10"], "results_at": row["at"]}


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "법령검색-v1"
    models = sys.argv[2:] or ["bm25", "gemma2", "bge-m3", "kure1", "kure2"]
    for i, model in enumerate(models):
        if i:
            time.sleep(120)
        st = state()
        r = {"at": datetime.now().isoformat(timespec="seconds"), "suite": suite, "model": model, "state": st, **measure(suite, model)}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(model, {k: r[k] for k in ("index_seconds", "query_ms_median", "store_mb", "peak_wired_gb", "peak_used_gb", "r@10")},
              "LM Studio:", st["lms_loaded"], flush=True)
