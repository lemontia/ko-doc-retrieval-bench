"""덕랩 임베딩 벤치 — 모델 하나 = models/<이름>.yaml, 평가판 하나 = suites/<이름>-v<판>/.

  uv run bench.py run --model kure2 [--suite 법령검색-v1]   모델 하나로 평가판 전부(또는 하나) → results.jsonl 에 한 줄씩 더한다
  uv run bench.py compare kure2 gemma2                      평가판별 지표 차이 + McNemar (각 모델의 가장 최근 줄)
  uv run bench.py export                                    public: yes 인 평가판만 export/<평가판>.json 으로

순위: 조각(unit) 점수 → 문서 점수는 조각 최고값 → 정답 문서의 순위. 점수가 같으면 정답을 뒤로 센다(비관적).
results.jsonl 은 지우지 않는다. 다시 재면 줄이 더해진다.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
CACHE = Path(os.environ.get("BENCH_CACHE") or ROOT / "cache")   # 기계 칸을 잴 땐 빈 폴더를 줘서 색인부터 다시 (tools/machine.py)
RESULTS = ROOT / "results.jsonl"


# ---------- 읽기 ----------

def load_model(name):
    m = yaml.safe_load((ROOT / "models" / f"{name}.yaml").read_text())
    m["name"] = name
    return m


def load_suite(dirname):
    d = ROOT / "suites" / dirname
    s = yaml.safe_load((d / "suite.yaml").read_text())
    s["dir"] = dirname
    s["public"] = s.get("public") is True   # 빠졌거나 yes 가 아니면 비공개
    s["units"] = [json.loads(l) for l in (d / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    s["questions"] = [json.loads(l) for l in (d / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    hp = d / "questions.hidden.jsonl"   # 숨긴 문항: 점수는 「숨김」 묶음으로 내고, 문항·순위는 export 로 안 나간다
    s["hidden"] = [json.loads(l) for l in hp.read_text(encoding="utf-8").splitlines()] if hp.exists() else []
    s["corpus_hash"] = hashlib.sha256((d / "corpus.jsonl").read_bytes()).hexdigest()[:12]
    # 정오표(tools/errata.py): 문항 파일은 쓴 그대로 두고 여기서 적용한다 — 제외 · 정답 추가 · 정답 교체
    ep = d / "errata.yaml"
    e = (yaml.safe_load(ep.read_text(encoding="utf-8")) or {}) if ep.exists() else {}
    s["errata_hash"] = hashlib.sha256(ep.read_bytes()).hexdigest()[:12] if ep.exists() else None
    drop, add, rep = e.get("drop") or {}, e.get("add_targets") or {}, e.get("replace_targets") or {}
    for part in ("questions", "hidden"):
        out = []
        for q in s[part]:
            if q["qid"] in drop:
                continue
            q = dict(q, targets=list(rep.get(q["qid"], q["targets"])))
            q["targets"] += [t for t in add.get(q["qid"], []) if t not in q["targets"]]
            out.append(q)
        s[part] = out
    docs = {u["doc"] for u in s["units"]}
    missing = [(q["qid"], t) for q in s["questions"] + s["hidden"] for t in q["targets"] if t not in docs]
    assert not missing, f"{dirname}: 코퍼스에 없는 정답 {missing[:5]}"
    return s


def suite_dirs():
    return sorted(p.parent.name for p in (ROOT / "suites").glob("*/suite.yaml"))


# ---------- 넣는 형식 ----------

def doc_text(m, u):
    fmt = m.get("doc_by_kind", {}).get(u["kind"], m["doc"])
    return fmt.format(title=u["title"], body=u["body"])


def query_text(m, q):
    return m["query"].format(q=q)


# ---------- 런타임: 모두 (문서 쪽 상태, 질의 → 조각 점수 배열) 을 만든다 ----------

def toks(t):
    """법지도 engine/index.py 와 같다: 한글·숫자·영문만 남기고, 두 글자 이상은 bigram."""
    t = re.sub(r"[^가-힣0-9a-zA-Z]", " ", t)
    out = []
    for w in t.split():
        out += [w] if len(w) < 2 else [w[i:i + 2] for i in range(len(w) - 1)]
    return out


def run_bm25(m, units, qs, key):
    k1, b = m.get("k1", 1.5), m.get("b", 0.75)
    tk = [toks(u["lex"]) for u in units]
    n, avgdl = len(units), sum(map(len, tk)) / len(units)
    df = Counter(t for x in tk for t in set(x))
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}   # Qdrant idf modifier 식
    post = {}
    for i, x in enumerate(tk):
        dl, tf = len(x), Counter(x)
        for t, f in tf.items():
            post.setdefault(t, []).append((i, f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))))
    def one(q):
        s = np.zeros(n)
        for t in set(toks(q)):
            for i, v in post.get(t, []):
                s[i] += idf[t] * v
        return s
    return [one(q) for q in qs], {"query_ms_median": _latency(one, qs)}


def _by_length(texts, encode, bs, lens=None):
    """길이 순으로 묶어 넣고(패딩 줄임) 원래 순서로 돌려놓는다. lens 를 안 주면 글자 수.
    묶음 구성이 바뀌면 MPS 에서 벡터가 조금 달라진다(KURE 토큰 벡터 최대 0.095, 10-18 확인) — 묶는 기준도 시험 조건이다"""
    order = np.argsort([len(t) for t in texts] if lens is None else lens, kind="stable")
    out = [None] * len(texts)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        for j, v in zip(idx, encode([texts[j] for j in idx])):
            out[j] = v
    return out


def _latency(one, qs, n=10):
    """질문 하나를 따로 넣어 코퍼스 전체 점수까지 내는 시간(ms) — 앞 n개의 중앙값. 처음 한 번은 데우기로 버린다."""
    one(qs[0])
    ts = []
    for q in qs[:n]:
        t0 = time.perf_counter()
        one(q)
        ts.append((time.perf_counter() - t0) * 1000)
    return round(sorted(ts)[len(ts) // 2], 1)


def _cached_dense(m, units, key, encode, bs):
    path = CACHE / f"{key}.npy"
    meta_p = path.with_suffix(".json")
    if path.exists():
        D = np.load(path)
        return D, {**json.loads(meta_p.read_text()), "store_mb": round(D.nbytes / 2**20, 1)}
    t0 = time.time()
    D = np.asarray(_by_length([doc_text(m, u) for u in units], encode, bs), dtype=np.float32)
    meta = {"index_seconds": round(time.time() - t0, 1), "store_mb": round(D.nbytes / 2**20, 1)}
    np.save(path, D)
    meta_p.write_text(json.dumps(meta))
    return D, meta


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_st(m, units, qs, key):
    import torch
    from sentence_transformers import SentenceTransformer
    kw = {"model_kwargs": {"dtype": getattr(torch, m["dtype"])}} if m.get("dtype") else {}
    st = SentenceTransformer(m["id"], revision=m["revision"], device=_device(), **kw)
    if m.get("max_seq"):
        st.max_seq_length = m["max_seq"]
    enc = lambda xs: st.encode(xs, prompt="", batch_size=len(xs), normalize_embeddings=True)
    D, meta = _cached_dense(m, units, key, enc, m.get("batch", 16))
    Q = enc([query_text(m, q) for q in qs])
    meta["max_seq"], meta["dtype"] = st.max_seq_length, str(next(st.parameters()).dtype)
    meta["query_ms_median"] = _latency(lambda q: D @ enc([query_text(m, q)])[0], qs)
    return [D @ q for q in Q], meta


def run_lmstudio(m, units, qs, key):
    import urllib.request

    def enc(xs):
        req = urllib.request.Request(m["url"], data=json.dumps({"model": m["id"], "input": xs}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())["data"]
        v = np.array([x["embedding"] for x in sorted(data, key=lambda x: x["index"])], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    D, meta = _cached_dense(m, units, key, enc, m.get("batch", 16))
    Q = enc([query_text(m, q) for q in qs])
    meta["query_ms_median"] = _latency(lambda q: D @ enc([query_text(m, q)])[0], qs)
    return [D @ q for q in Q], meta


def run_pylate(m, units, qs, key):
    """ColBERT(MaxSim). 문서 토큰 벡터는 fp16 으로 저장, 전부 직접 계산(근사 색인 없음)."""
    from pylate import models
    cm = models.ColBERT(model_name_or_path=m["id"], revision=m.get("revision"), device=_device())
    path = CACHE / f"{key}.npz"
    meta_p = path.with_suffix(".json")
    if path.exists():
        z, meta = np.load(path), json.loads(meta_p.read_text())
        flat, off = z["flat"], z["off"]
    else:
        t0 = time.time()
        enc = lambda xs: [np.asarray(e, dtype=np.float16) for e in cm.encode(xs, is_query=False, batch_size=len(xs))]
        texts = [doc_text(m, u) for u in units]
        lens = [len(x) for x in cm.tokenizer(texts, add_special_tokens=True)["input_ids"]]   # 법지도 10-17 과 같이 토큰 길이 순
        embs = _by_length(texts, enc, m.get("batch", 8), lens)
        off = np.cumsum([0] + [len(e) for e in embs])
        flat = np.concatenate(embs)
        meta = {"index_seconds": round(time.time() - t0, 1), "token_vectors": int(len(flat)),
                "store_mb_fp16": round(flat.nbytes / 2**20, 1)}
        np.savez(path, flat=flat, off=off)
        meta_p.write_text(json.dumps(meta))
    T = flat.astype(np.float32)
    score = lambda qv: np.maximum.reduceat(np.asarray(qv, dtype=np.float32) @ T.T, off[:-1], axis=1).sum(0).astype(np.float64)
    out = [score(q) for q in cm.encode([query_text(m, q) for q in qs], is_query=True)]
    meta["store_mb_fp16"] = round(flat.nbytes / 2**20, 1)
    meta["query_ms_median"] = _latency(lambda q: score(cm.encode([query_text(m, q)], is_query=True)[0]), qs)
    return out, meta


RUNTIMES = {"bm25": run_bm25, "st": run_st, "lmstudio": run_lmstudio, "pylate": run_pylate}


# ---------- 순위·지표 ----------

def doc_ranks(unit_scores, units, questions):
    docs = sorted({u["doc"] for u in units})
    pos = {d: i for i, d in enumerate(docs)}
    ui = np.array([pos[u["doc"]] for u in units])
    out = []
    for s, q in zip(unit_scores, questions):
        ds = np.full(len(docs), -np.inf)
        np.maximum.at(ds, ui, s)
        best = max(ds[pos[t]] for t in q["targets"])
        out.append(int((ds > best).sum() + (ds == best).sum()))   # 동점은 정답에 불리하게
    return out


def metrics(ranks, recall_at, mrr_at):
    n = len(ranks)
    out = {f"r@{k}": sum(r <= k for r in ranks) / n for k in recall_at}
    out["mrr"] = sum(1 / r for r in ranks if mrr_at is None or r <= mrr_at) / n
    return out


def mcnemar(a, b):
    """짝지은 정확 이항 양측 p — a·b 는 문항별 적중 여부. (A만, B만, p)"""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return n01, n10, p


# ---------- 명령 ----------

def versions():
    out = {"python": platform.python_version(), "machine": platform.machine(), "node": platform.node()}
    for lib in ("numpy", "torch", "sentence_transformers", "transformers", "pylate"):
        mod = sys.modules.get(lib)
        if mod is not None:
            out[lib] = getattr(mod, "__version__", "?")
    return out


def cmd_run(a):
    m = load_model(a.model)
    if m.get("requires") and not os.environ.get("BENCH_CHILD"):
        # 모델마다 라이브러리 버전이 다르다(pylate 1.6 은 sentence-transformers 5.3 고정, gemma2 원본은 6.x 필요).
        # 모델 파일의 requires 로 띄운 임시 환경에서 같은 명령을 다시 돈다
        cmd = ["uv", "run", "--no-project", "--with", "numpy", "--with", "pyyaml"]
        for r in m["requires"]:
            cmd += ["--with", r]
        cmd += ["python", str(Path(__file__).resolve()), *sys.argv[1:]]
        os.execvpe("uv", cmd, {**os.environ, "BENCH_CHILD": "1"})
    for sd in [a.suite] if a.suite else suite_dirs():
        s = load_suite(sd)
        fmt = json.dumps([m.get("doc"), m.get("doc_by_kind"), m.get("query"), m.get("max_seq"), m.get("dtype")],
                         ensure_ascii=False)
        key = f"{m['name']}_{s['corpus_hash']}_{hashlib.sha256((str(m.get('id')) + str(m.get('revision')) + fmt).encode()).hexdigest()[:8]}"
        t0 = time.time()
        allq = s["questions"] + s["hidden"]
        # 공개·숨김을 따로 넣는다 — 질의도 묶어서 벡터로 바꾸므로, 섞으면 묶음이 바뀌어 공개 문항 값이 흔들린다(bf16·MPS)
        scores, meta = RUNTIMES[m["runtime"]](m, s["units"], [q["question"] for q in s["questions"]], key)
        if s["hidden"]:
            scores = list(scores) + list(RUNTIMES[m["runtime"]](m, s["units"], [q["question"] for q in s["hidden"]], key)[0])
        ranks = doc_ranks(scores, s["units"], allq)
        meta["total_seconds"] = round(time.time() - t0, 1)
        npub = len(s["questions"])
        groups = {"전체": list(range(npub))}   # 「전체」는 공개 문항만 — 숨긴 문항이 생기기 전 기록과 계속 비교된다
        for i, q in enumerate(s["questions"]):
            groups.setdefault(q.get("group", "전체"), []).append(i)
        if s["hidden"]:
            groups["숨김"] = list(range(npub, len(allq)))
        row = {"at": datetime.now().isoformat(timespec="seconds"), "model": m["name"],
               "model_spec": {k: v for k, v in m.items() if k != "name"}, "suite": sd, "public": s["public"],
               "corpus_hash": s["corpus_hash"], "errata_hash": s["errata_hash"], "n": npub, "n_hidden": len(s["hidden"]),
               "metrics": {g: metrics([ranks[i] for i in ix], s["recall_at"], s.get("mrr_at")) for g, ix in groups.items()},
               "ranks": {q["qid"]: r for q, r in zip(allq, ranks)}, "meta": meta, "env": versions()}
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(sd, m["name"], {g: {k: round(v, 3) for k, v in x.items()} for g, x in row["metrics"].items()}, meta, flush=True)


def latest(model, suite):
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()] if RESULTS.exists() else []
    rows = [r for r in rows if r["model"] == model and r["suite"] == suite]
    return rows[-1] if rows else None


def cmd_compare(a):
    for sd in suite_dirs():
        s = load_suite(sd)
        A, B = latest(a.a, sd), latest(a.b, sd)
        if not (A and B):
            print(f"\n## {sd} — 기록 없음 ({a.a if not A else a.b})")
            continue
        if A["corpus_hash"] != B["corpus_hash"] or A["corpus_hash"] != s["corpus_hash"] or \
                A.get("errata_hash") != s["errata_hash"] or B.get("errata_hash") != s["errata_hash"]:
            print(f"\n## {sd} — 코퍼스·정오표 해시가 다르다 ({A['corpus_hash']} / {B['corpus_hash']} / 지금 {s['corpus_hash']}). 다시 잴 것")
            continue
        print(f"\n## {sd} (n={A['n']}, 공개 {'예' if s['public'] else '아니오'})\n")
        keys = list(A["metrics"]["전체"])
        print("| 묶음 | 모델 | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
        for g in [g for g in A["metrics"] if g in B["metrics"]]:   # 숨김이 생기기 전 줄과 비교해도 깨지지 않게
            for r in (A, B):
                print(f"| {g} | {r['model']} | " + " | ".join(f"{r['metrics'][g][k]:.3f}" for k in keys) + " |")
        for label, part in (("공개", s["questions"]), ("숨김", s["hidden"])):
            qids = [q["qid"] for q in part if q["qid"] in A["ranks"] and q["qid"] in B["ranks"]]
            if not qids:
                continue
            print(f"\n| McNemar {label} (n={len(qids)}) | {a.a} 만 맞힘 | {a.b} 만 맞힘 | p |\n|---|---|---|---|")
            for k in s["recall_at"]:
                n01, n10, p = mcnemar([A["ranks"][q] <= k for q in qids], [B["ranks"][q] <= k for q in qids])
                print(f"| @{k} | {n01} | {n10} | {p:.2g} |")


def public_row(r, s):
    """기록 한 줄에서 공개해도 되는 것만. 순위는 공개 문항 것만 남긴다 — 숨긴 문항은 「숨김」 묶음 점수만 나간다."""
    pub = {q["qid"] for q in s["questions"]}
    out = {k: r[k] for k in ("model", "at", "metrics", "meta")}
    out["model_spec"] = {k: v for k, v in r["model_spec"].items() if k != "url"}   # LM Studio 주소(사설망 IP)
    out["env"] = {k: v for k, v in r["env"].items() if k != "node"}                 # 기계 호스트 이름
    out["ranks"] = {q: v for q, v in r["ranks"].items() if q in pub}
    out["n_hidden"] = r.get("n_hidden", 0)
    return out


def public_errata(sd, s):
    """정오표 중 공개해도 되는 것 — 숨긴 문항은 qid·이유를 빼고 개수만."""
    ep = ROOT / "suites" / sd / "errata.yaml"
    e = (yaml.safe_load(ep.read_text(encoding="utf-8")) or {}) if ep.exists() else {}
    qf = ROOT / "suites" / sd / "questions.jsonl"   # 공개 문항 파일 그대로(정오표로 뺀 것 포함)
    pub = {json.loads(l)["qid"] for l in qf.read_text(encoding="utf-8").splitlines()}
    keep = lambda m: {k: v for k, v in (m or {}).items() if k in pub}
    hid = lambda m: sum(1 for k in (m or {}) if k not in pub)
    return {"drop": keep(e.get("drop")), "add_targets": keep(e.get("add_targets")), "replace_targets": keep(e.get("replace_targets")),
            "removed_docs": len(e.get("removed_docs") or []),
            "hidden_changed": hid(e.get("drop")) + hid(e.get("add_targets")) + hid(e.get("replace_targets"))}


def cmd_export(a, out_dir=None):
    """공개 평가판만 내보낸다. 비공개가 섞이지 않게: 공개 목록을 먼저 정하고, 그 목록 밖의 이름은 아예 읽지 않는다."""
    out_dir = Path(out_dir) if out_dir else ROOT / "export"   # 검사는 임시 폴더를 준다 — 실제 export/ 를 덮어쓰지 않게
    out_dir.mkdir(exist_ok=True)
    public = [sd for sd in suite_dirs() if load_suite(sd)["public"]]
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()] if RESULTS.exists() else []
    for sd in public:
        s = load_suite(sd)
        assert s["public"] is True
        mine = [r for r in rows if r["suite"] == sd and r["public"] is True and r["corpus_hash"] == s["corpus_hash"] and r.get("errata_hash") == s["errata_hash"]]
        last = {}
        for r in mine:
            last[r["model"]] = r   # 모델마다 가장 최근 줄
        doc = {"suite": sd, "name": s["name"], "version": s["version"], "desc": s["desc"], "corpus_hash": s["corpus_hash"],
               "overlap": s.get("overlap"),
               "questions": s["questions"],
               "errata": public_errata(sd, s),
               "models": [public_row(r, s) for r in last.values()]}
        (out_dir / f"{sd}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print("export", sd, len(last), "models")
    return public


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--suite")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    sub.add_parser("export")
    a = ap.parse_args(argv)
    CACHE.mkdir(exist_ok=True)
    {"run": cmd_run, "compare": cmd_compare, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
