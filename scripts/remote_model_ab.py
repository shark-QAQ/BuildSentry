#!/usr/bin/env python
"""远程微调模型 A/B 对照测试（LoRA vs 基座）。

连到 10.1.4.86:8124 的 vLLM 服务，用同一批段落分别打基座模型与 LoRA 适配器，
对比四项指标：

1. JSON 解析成功率 —— 微调收益最明确的地方（训练报告 §5.3 ⑦）
2. 判定一致率     —— has_violation 是否与人工标注一致
3. 编造条款数     —— 引用了不在 prompt 依据里的条款（红线指标，应 = 0）
4. 输出长度/耗时   —— 吞吐类指标必须与输出长度一起看（训练报告 §5.4 的教训）

用法:
    python scripts/remote_model_ab.py
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.audit_contract import (  # noqa: E402
    OutputParseError,
    build_prompt,
    parse_model_output,
)

# 所有样本共用同一组依据。这样「引用不在依据里的条款」= 编造，
# 判定干净、无需人工判读。这也顺带考察模型能否从多条依据中挑出相关的那条。
CLAUSES = [
    {
        "content": "基坑周边1.2米范围内严禁堆载。",
        "metadata": {"spec_code": "JGJ 59-2011", "article_id": "3.11.3"},
    },
    {
        "content": "脚手架连墙件应按规定设置，竖向间距不应大于建筑物层高，且不应大于4米。",
        "metadata": {"spec_code": "JGJ 130-2011", "article_id": "6.3.4"},
    },
    {
        "content": "模板支架立杆间距不应大于1.2米，步距不应大于1.8米。",
        "metadata": {"spec_code": "JGJ 162-2008", "article_id": "5.2.2"},
    },
    {
        "content": "高处作业人员应正确佩戴安全带，安全带应高挂低用。",
        "metadata": {"spec_code": "JGJ 80-2016", "article_id": "4.1.3"},
    },
]

# 依据里出现过的条号。模型引用的条号若不在此集合内，即判为编造。
KNOWN_ARTICLE_IDS = {c["metadata"]["article_id"] for c in CLAUSES}

# (段落, 人工标注: 是否违规, 预期条号)
SAMPLES = [
    (
        "基坑开挖深度4.5米，采用放坡开挖，坡率1:0.5。基坑周边0.8米范围内临时堆放"
        "钢筋原材约15吨，堆放高度不超过1.5米。基坑边坡设置排水沟，坡顶设置1.2米高防护栏杆。",
        True,
        "3.11.3",
    ),
    (
        "基坑开挖深度3.2米，采用放坡开挖，坡率1:0.75。基坑周边2.0米范围外集中堆放"
        "周转材料，堆放高度1.0米，堆载不超过设计允许值。坡顶设置1.2米高防护栏杆并悬挂警示标志。",
        False,
        None,
    ),
    (
        "外脚手架搭设高度24米，采用扣件式钢管脚手架。连墙件按两步三跨设置，"
        "竖向间距6.0米，水平间距4.5米，连墙件采用刚性连接。脚手架外侧满挂密目安全网。",
        True,
        "6.3.4",
    ),
    (
        "外脚手架搭设高度18米，采用扣件式钢管脚手架。连墙件按两步三跨设置，"
        "竖向间距3.5米，水平间距4.0米，连墙件采用刚性连接并靠近主节点。外侧满挂密目安全网。",
        False,
        None,
    ),
    (
        "现浇混凝土楼板模板支架采用扣件式钢管支架，立杆纵横间距1.5米，步距1.8米，"
        "立杆底部设置可调底座。支架搭设完成后经验收合格方可浇筑混凝土。",
        True,
        "5.2.2",
    ),
    (
        "现浇混凝土楼板模板支架采用扣件式钢管支架，立杆纵横间距1.0米，步距1.5米，"
        "立杆底部设置可调底座。支架搭设完成后经验收合格方可浇筑混凝土。",
        False,
        None,
    ),
    (
        "主体结构施工至6层，外防护采用悬挑脚手架。高处作业人员统一佩戴安全带，"
        "作业时将安全带低挂高用，挂设在下方横杆上。临边洞口设置防护栏杆和盖板。",
        True,
        "4.1.3",
    ),
    (
        "主体结构施工至6层，外防护采用悬挑脚手架。高处作业人员统一佩戴安全带，"
        "作业时将安全带高挂低用，挂设在上方牢固的挂点上。临边洞口设置防护栏杆和盖板。",
        False,
        None,
    ),
    (
        "基坑开挖深度5.0米，采用放坡开挖。基坑周边1.0米范围内堆放土方约20立方米。"
        "高处作业人员下坑作业时佩戴安全带，安全带低挂高用挂在坡顶栏杆上。坡顶设置1.2米高防护栏杆。",
        True,
        "3.11.3",
    ),
    (
        "基坑开挖深度4.0米，采用放坡开挖，坡率1:1.0。基坑周边1.5米范围外堆放材料，"
        "堆载不超过10kPa。坡顶设置1.2米高防护栏杆，坑内设置上下专用通道。降水井间距15米。",
        False,
        None,
    ),
    (
        "地下室顶板模板支架搭设高度5.2米，立杆纵距1.8米，横距1.5米，步距2.0米。"
        "支架四周设置竖向剪刀撑，中部每隔4跨设置一道竖向剪刀撑。支架与框架柱刚性连接。",
        True,
        "5.2.2",
    ),
    (
        "地下室顶板模板支架搭设高度5.2米，立杆纵距0.9米，横距0.9米，步距1.5米。"
        "支架四周设置竖向剪刀撑，中部每隔4跨设置一道竖向剪刀撑。支架与框架柱刚性连接。",
        False,
        None,
    ),
]

_FULLWIDTH = str.maketrans("０１２３４５６７８９．", "0123456789.")


def normalize(text: str) -> str:
    """归一化条号文本：全角转半角、去「第/条/款」修饰字。

    注意**不能去分隔符**：规范编号与条号之间靠空格/短横线分隔，
    一旦压掉，"JGJ 59-2011 3.11.3" 会粘成 "JGJ59-20113.11.3"，
    抽出条号就成了 20113.11.3，把所有正常引用误判成编造。
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_FULLWIDTH)
    return text.replace("第", "").replace("条", "").replace("款", "")


def cited_article_ids(returned_code: str) -> set[str]:
    """从模型返回的 clause_code 里抽出所有条号片段。

    模型可能返回 "JGJ 59-2011 3.11.3"，也可能只返回 "3.11.3"，
    两种都要能识别，故直接扫出所有形如 数字.数字(.数字) 的片段。
    规范编号形如 "59-2011"（含短横线）不会被这个正则命中，天然被排除。
    """
    return set(re.findall(r"\d+(?:\.\d+)+", normalize(returned_code)))


def is_grounded(cited: str, known: set[str]) -> bool:
    """cited 是否为依据里出现过的条号。

    容忍模型只写 "3.11" 或写成 "3.11.3.1" 这类前后缀差异
    ——差异不该被算成编造，编造指的是引出依据里根本没提的条号。
    """
    for k in known:
        if cited == k or k.startswith(cited + ".") or cited.startswith(k + "."):
            return True
    return False


def http_post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(base_url: str, timeout: int = 15) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["id"] for m in data.get("data", [])]


def evaluate_one(base_url: str, model: str, paragraph: str, expect_violation: bool,
                 expect_clause: str | None, max_tokens: int, timeout: int) -> dict:
    """跑一条样本，返回结构化结果。异常不抛出，记为 error 字段。"""
    prompt = build_prompt(paragraph, CLAUSES)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # Qwen3 默认开思考；思考内容里的花括号会让 parse_model_output 取错位置
        "chat_template_kwargs": {"enable_thinking": False},
    }

    started = time.time()
    try:
        resp = http_post(f"{base_url}/v1/chat/completions", payload, timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        return {"error": f"HTTP {e.code}: {body}", "elapsed": time.time() - started}
    except Exception as e:  # noqa: BLE001 - 连通性测试，任何异常都记录下来
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - started}

    elapsed = time.time() - started
    choice = resp["choices"][0]
    raw = choice["message"]["content"]
    usage = resp.get("usage", {})

    parsed, parse_error = None, None
    try:
        parsed = parse_model_output(raw)
    except OutputParseError as e:
        parse_error = str(e)

    fabricated: list[str] = []
    matched_expected = False
    if parsed is not None:
        for v in parsed:
            ids = cited_article_ids(str(v.get("clause_code", "")))
            # 一条引用都没抽出条号，或引出的条号依据里全没有 → 编造
            if not ids or not any(is_grounded(i, KNOWN_ARTICLE_IDS) for i in ids):
                fabricated.append(str(v.get("clause_code", "")))
            if expect_clause and any(is_grounded(i, {expect_clause}) for i in ids):
                matched_expected = True

    return {
        "error": None,
        "elapsed": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "raw": raw,
        "parse_error": parse_error,
        "reported_violation": bool(parsed) if parsed is not None else None,
        "num_reported": len(parsed) if parsed is not None else None,
        "fabricated": fabricated,
        "expected_violation": expect_violation,
        "expected_clause": expect_clause,
        "matched_expected": matched_expected,
    }


def run_model(base_url: str, model: str, args) -> list[dict]:
    """并发跑完所有样本，结果按输入顺序返回（不要用 as_completed 的顺序）。"""
    results: list[dict] = [{}] * len(SAMPLES)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                evaluate_one, base_url, model, para, exp, exp_clause,
                args.max_tokens, args.timeout,
            ): idx
            for idx, (para, exp, exp_clause) in enumerate(SAMPLES)
        }
        done = 0
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            print(f"    [{model}] {done}/{len(SAMPLES)} 完成", flush=True)
    return results


def summarize(model: str, results: list[dict]) -> dict:
    n = len(results)
    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    parsed = [r for r in ok if r["parse_error"] is None]

    tp = sum(1 for r in parsed if r["reported_violation"] and r["expected_violation"])
    fp = sum(1 for r in parsed if r["reported_violation"] and not r["expected_violation"])
    fn = sum(1 for r in parsed if not r["reported_violation"] and r["expected_violation"])
    tn = sum(1 for r in parsed if not r["reported_violation"] and not r["expected_violation"])

    fabricated = sum(len(r["fabricated"]) for r in parsed)
    total_reported = sum(r["num_reported"] or 0 for r in parsed)
    clause_hits = sum(1 for r in parsed if r["expected_violation"] and r["matched_expected"])
    expected_violations = sum(1 for r in parsed if r["expected_violation"])

    return {
        "model": model,
        "n": n,
        "errors": len(errors),
        "parse_ok": len(parsed),
        "parse_rate": len(parsed) / n if n else 0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(parsed) if parsed else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "fabricated": fabricated,
        "clause_hit_rate": clause_hits / expected_violations if expected_violations else 0,
        "avg_tokens": sum(r["completion_tokens"] for r in ok) / len(ok) if ok else 0,
        "avg_reported": total_reported / len(parsed) if parsed else 0,
        "avg_elapsed": sum(r["elapsed"] for r in ok) / len(ok) if ok else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.1.4.86:8124")
    ap.add_argument("--lora", default="audit")
    ap.add_argument("--base-model", default="qwen3-32b-audit")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true", help="打印每条样本的明细")
    ap.add_argument("--save", default=None, help="把原始结果存成 JSON，便于复看/复算")
    args = ap.parse_args()

    print(f"目标服务: {args.base_url}")
    try:
        models = list_models(args.base_url)
    except Exception as e:  # noqa: BLE001
        print(f"[失败] 无法访问服务: {e}")
        return 1
    print(f"可用模型: {models}")
    print(f"样本数: {len(SAMPLES)}（违规 {sum(1 for _, e, _ in SAMPLES if e)} / "
          f"合规 {sum(1 for _, e, _ in SAMPLES if not e)}）\n")

    all_results: dict[str, list[dict]] = {}
    for label, model in (("LoRA", args.lora), ("基座", args.base_model)):
        if model not in models:
            print(f"=== {label} [{model}] 不在服务上，跳过 ===\n")
            continue
        print(f"=== 正在跑 {label} [{model}] ===")
        all_results[label] = run_model(args.base_url, model, args)
        print()

    if not all_results:
        print("没有任何模型跑成功")
        return 1

    if args.save:
        payload = {
            label: {"summary": summarize(label, res), "results": res}
            for label, res in all_results.items()
        }
        Path(args.save).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"原始结果已存: {args.save}\n")

    summaries = [summarize(label, res) for label, res in all_results.items()]

    print("=" * 78)
    print("汇总对比")
    print("=" * 78)
    header = f"{'指标':<22}" + "".join(f"{s['model']:>16}" for s in summaries)
    print(header)
    print("-" * 78)

    def row(name: str, fmt, values):
        print(f"{name:<22}" + "".join(f"{fmt(v):>16}" for v in values))

    row("请求成功", lambda v: f"{v['n'] - v['errors']}/{v['n']}", summaries)
    row("JSON 解析成功率", lambda v: f"{v['parse_rate']:.1%}", summaries)
    row("判定一致率", lambda v: f"{v['accuracy']:.1%}", summaries)
    row("违规召回率", lambda v: f"{v['recall']:.1%}", summaries)
    row("违规精确率", lambda v: f"{v['precision']:.1%}", summaries)
    row("命中预期条号率", lambda v: f"{v['clause_hit_rate']:.1%}", summaries)
    row("编造条款数(红线)", lambda v: str(v["fabricated"]), summaries)
    row("平均报出违规数", lambda v: f"{v['avg_reported']:.2f}", summaries)
    row("平均输出 tok", lambda v: f"{v['avg_tokens']:.0f}", summaries)
    row("平均耗时", lambda v: f"{v['avg_elapsed']:.1f}s", summaries)
    print()

    print(f"tp/fp/fn/tn: " + " | ".join(
        f"{s['model']}: {s['tp']}/{s['fp']}/{s['fn']}/{s['tn']}" for s in summaries))

    if args.verbose:
        print("\n" + "=" * 78)
        print("逐样本明细")
        print("=" * 78)
        for label, results in all_results.items():
            print(f"\n--- {label} ---")
            for i, r in enumerate(results):
                _, exp, exp_clause = SAMPLES[i]
                gt = f"违规({exp_clause})" if exp else "合规"
                if r.get("error"):
                    print(f"  #{i + 1:>2} gt={gt:<14} [请求失败] {r['error'][:60]}")
                elif r["parse_error"]:
                    print(f"  #{i + 1:>2} gt={gt:<14} [解析失败] {r['parse_error'][:60]}")
                else:
                    got = f"违规×{r['num_reported']}" if r["reported_violation"] else "合规"
                    mark = "✓" if (r["reported_violation"] == exp) else "✗"
                    fab = f" 编造:{r['fabricated']}" if r["fabricated"] else ""
                    print(f"  #{i + 1:>2} gt={gt:<14} 模型={got:<10} {mark}{fab}")
                    if mark == "✗":
                        # 判错的样本把原文打出来，便于判断是模型的问题还是标注的问题
                        print("       " + r["raw"].replace("\n", " ")[:300])

    return 0


if __name__ == "__main__":
    sys.exit(main())
