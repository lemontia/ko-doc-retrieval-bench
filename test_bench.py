"""uv run python test_bench.py — 실패하면 AssertionError."""
import json
from pathlib import Path

import bench


def test_export_public_only():
    private = [sd for sd in bench.suite_dirs() if not bench.load_suite(sd)["public"]]
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bench-export-"))
    out = bench.cmd_export(None, tmp)
    assert not set(out) & set(private), out
    private_qids = {q["qid"] for sd in private for q in bench.load_suite(sd)["questions"]}
    for p in tmp.glob("*.json"):
        assert p.stem not in private, p
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert not {q["qid"] for q in doc["questions"]} & private_qids
        assert all(not set(m["ranks"]) & private_qids for m in doc["models"])


def test_export_no_internal_info():
    """공개 export 에 사설망 IP(Tailscale 100.x)·기계 호스트 이름이 없어야 한다."""
    import platform, re, tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bench-export-"))
    bench.cmd_export(None, tmp)
    host = platform.node().split(".")[0]
    for p in tmp.glob("*.json"):
        t = p.read_text(encoding="utf-8")
        assert not re.search(r"\b100\.\d+\.\d+\.\d+", t), p
        assert host not in t, (p, host)


def test_hidden_not_exported():
    """숨긴 문항의 qid·질문 글이 export 어디에도 없어야 한다(점수만 나간다)."""
    s = {"questions": [{"qid": "E001"}], "hidden": [{"qid": "H001", "question": "숨긴 질문 글"}]}
    r = {"model": "m", "model_spec": {}, "at": "t", "meta": {}, "env": {},
         "metrics": {"전체": {"r@1": 1.0}, "숨김": {"r@1": 0.0}}, "ranks": {"E001": 1, "H001": 9}, "n_hidden": 1}
    out = json.dumps(bench.public_row(r, s), ensure_ascii=False)
    assert "H001" not in out and "숨긴 질문 글" not in out, out
    assert '"숨김"' in out and '"E001"' in out
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bench-export-"))
    bench.cmd_export(None, tmp)   # 지금 코드로 새로 내보낸 것을 본다(옛 export/ 파일이 아니라)
    for sd in bench.suite_dirs():
        st = bench.load_suite(sd)
        p = tmp / f"{sd}.json"
        if st["hidden"] and p.exists():
            text = p.read_text(encoding="utf-8")
            for q in st["hidden"]:
                assert f'"{q["qid"]}"' not in text and q["question"] not in text, (sd, q["qid"])


def test_public_flag_strict(tmp="_t"):
    d = bench.ROOT / "suites" / tmp
    d.mkdir()
    try:
        (d / "corpus.jsonl").write_text("")
        (d / "questions.jsonl").write_text("")
        for text, want in [("public: yes", True), ("public: no", False), ("public: '예'", False), ("", False)]:
            (d / "suite.yaml").write_text(text + "\nname: t\n")
            assert bench.load_suite(tmp)["public"] is want, text
    finally:
        for p in d.iterdir():
            p.unlink()
        d.rmdir()


def test_ranks_ties_and_max_per_doc():
    import numpy as np
    units = [{"doc": "a"}, {"doc": "a"}, {"doc": "b"}, {"doc": "c"}]
    s = [np.array([0.1, 0.9, 0.9, 0.5])]   # a 의 최고 0.9 = b 0.9 → 동점이면 정답을 뒤로
    assert bench.doc_ranks(s, units, [{"targets": ["a"]}]) == [2]
    assert bench.doc_ranks(s, units, [{"targets": ["c"]}]) == [3]
    assert bench.metrics([1, 3, 60], [1, 5], 50) == {"r@1": 1 / 3, "r@5": 2 / 3, "mrr": (1 + 1 / 3) / 3}


def test_mcnemar():
    assert bench.mcnemar([True] * 38 + [False] * 62, [False] * 100)[2] < 1e-10
    assert bench.mcnemar([True, False], [False, True]) == (1, 1, 1.0)


def test_errata_applied(tmp="_e"):
    """정오표: 제외한 문항은 사라지고, 정답 추가·교체가 순위에 반영된다."""
    import numpy as np
    d = bench.ROOT / "suites" / tmp
    d.mkdir()
    try:
        (d / "suite.yaml").write_text("name: t\npublic: yes\n")
        (d / "corpus.jsonl").write_text("".join(json.dumps({"id": x, "doc": x, "kind": "판례", "title": "", "body": "", "lex": ""}) + "\n" for x in "abc"))
        (d / "questions.jsonl").write_text("".join(json.dumps({"qid": q, "question": "", "targets": [t]}) + "\n" for q, t in [("Q1", "a"), ("Q2", "a"), ("Q3", "a")]))
        (d / "errata.yaml").write_text("drop: {Q1: 불량}\nadd_targets: {Q2: [b]}\nreplace_targets: {Q3: [c]}\n")
        s = bench.load_suite(tmp)
        assert [q["qid"] for q in s["questions"]] == ["Q2", "Q3"]
        assert s["questions"][0]["targets"] == ["a", "b"] and s["questions"][1]["targets"] == ["c"]
        sc = [np.array([0.1, 0.9, 0.5])] * 2      # b 가 1위 → Q2 는 추가 정답으로 1위, Q3 정답 c 는 2위
        assert bench.doc_ranks(sc, s["units"], s["questions"]) == [1, 2]
        (d / "errata.yaml").write_text("add_targets: {Q2: [없는문서]}\n")
        try:
            bench.load_suite(tmp)
            raise SystemExit("코퍼스에 없는 정답을 막지 못함")
        except AssertionError:
            pass
    finally:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


def test_fixed_suites_unchanged():
    """고정한 평가판의 코퍼스·문항은 바뀌면 안 된다(정오표는 더할 수 있으니 보지 않는다)."""
    import hashlib
    import yaml
    for sd in bench.suite_dirs():
        y = yaml.safe_load((bench.ROOT / "suites" / sd / "suite.yaml").read_text())
        fh = y.get("fixed_hashes")
        if not fh:
            continue
        d = bench.ROOT / "suites" / sd
        for k, f in (("corpus", "corpus.jsonl"), ("questions", "questions.jsonl"), ("hidden", "questions.hidden.jsonl")):
            if not (d / f).exists():   # 공개판엔 숨긴 문항 파일이 없다
                continue
            assert hashlib.sha256((d / f).read_bytes()).hexdigest()[:12] == fh[k], (sd, k)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
