# -*- coding: utf-8 -*-
"""
knowledge_base.py — 露天矿安全知识库 (RAG 检索模块)
====================================================
构建矿山安全领域知识文档库, 提供本地 BM25 检索(纯 numpy, 零外部依赖)。
知识来源: 行业法规、技术规范、灾害机理、预警分级、处置措施、典型案例。

用途:
  1. 多智能体 pipeline: 检测结果 → 构造查询 → 检索相关知识 → 注入 prompt (RAG)
  2. 前端"知识库查询"演示功能

用法:
  from knowledge_base import KnowledgeBase
  kb = KnowledgeBase()
  chunks = kb.retrieve("边坡 滑坡 红色预警 处置措施", k=3)
"""
import re
import math
import json
from pathlib import Path
from collections import Counter

# 知识条目由来源登记表加载；法规摘要、元数据和项目说明明确分开。
KNOWLEDGE_DOCS = []  # populated from the versioned, source-linked catalog below

# ── BM25 检索器 (纯 numpy, 零依赖) ────────────────────
class BM25Retriever:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.corpus = [self._tokenize(d["title"] + " " + d["title"] + " " +
                                      d.get("keywords", "") + " " + d["content"]) for d in docs]
        self.avgdl = sum(len(t) for t in self.corpus) / max(len(self.corpus), 1)
        self.idf = {}
        n_docs = len(self.corpus)
        df = Counter()
        for toks in self.corpus:
            df.update(set(toks))
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        self.term_freqs = [Counter(t) for t in self.corpus]

    @staticmethod
    def _tokenize(text):
        """中英文分词: 英文按词, 中文按2-gram (简单有效)"""
        tokens = []
        text = text.lower()
        for word in re.findall(r"[a-z0-9]+", text):
            tokens.append(word)
        cn = re.findall(r"[一-鿿]+", text)
        for seg in cn:
            if len(seg) == 1:
                tokens.append(seg)
            else:
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i + 2])
        stop = {"什么", "如何", "怎么", "是否", "进行", "关于", "可以", "我们", "需要"}
        return [token for token in tokens if token not in stop]

    def score(self, query_tokens, tf, doc_len):
        score = 0.0
        for term in query_tokens:
            if term not in self.idf or term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[term] * f * (self.k1 + 1) / max(denom, 1e-9)
        return score

    def retrieve(self, query, k=3):
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scored = []
        for i, tf in enumerate(self.term_freqs):
            s = self.score(q_tokens, tf, sum(tf.values()))
            # A queried standard identifier outranks a neighboring standard that
            # mentions it only in the body. Years are not document identifiers.
            for number in re.findall(r"(?<!\d)\d{5}(?!\d)", query):
                if number in self.docs[i]["title"]:
                    s += 20
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda item: (-item[0], self.docs[item[1]]["id"]))
        chosen, remaining = [], []
        by_source = Counter()
        # Keep relevant neighboring chunks without letting one source fill the whole context.
        for _, i in scored:
            doc = self.docs[i]
            if k >= 3 and by_source[doc.get("source_id")] >= 2:
                remaining.append(doc)
                continue
            chosen.append(doc)
            by_source[doc.get("source_id")] += 1
            if len(chosen) >= k:
                break
        return (chosen + remaining)[:max(0, k)]


class KnowledgeBase:
    def __init__(self, data_dir=None):
        root = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.sources = json.loads((root / "knowledge_sources.json").read_text(encoding="utf-8"))
        chunks = json.loads((root / "knowledge_chunks.json").read_text(encoding="utf-8"))
        self.docs = []
        seen = set()
        for chunk in chunks:
            identifier = chunk.get("id", "")
            if not re.fullmatch(r"[A-Z]+-\d{3}", identifier) or identifier in seen:
                raise ValueError(f"Invalid/duplicate knowledge ID: {identifier}")
            seen.add(identifier)
            src = self.sources[chunk["source_id"]]
            for key in ("source_title", "publisher", "checked_at", "evidence_type", "scope", "verification"):
                if not src.get(key):
                    raise ValueError(f"Missing provenance {key}: {identifier}")
            if not chunk.get("content") or not chunk.get("locator"):
                raise ValueError(f"Missing content/locator: {identifier}")
            merged = dict(src, **chunk)
            merged["related_sources"] = [dict(self.sources[key], source_id=key)
                                         for key in chunk.get("related_source_ids", [])]
            self.docs.append(merged)
        self.retriever = BM25Retriever(self.docs)

    def retrieve(self, query, k=3):
        return self.retriever.retrieve(query, k)

    def format_context(self, chunks):
        """检索结果 → 可注入 prompt 的文本块"""
        parts = []
        for c in chunks:
            provenance = (f"[{c['id']}] [{c['category']}] {c['title']}\n"
                          f"资料类型：{c['evidence_type']}；适用范围：{c['scope']}\n"
                          f"出处：{c['source_title']}；定位：{c['locator']}；"
                          f"核对日期：{c['checked_at']}\n")
            if c.get("source_url"):
                provenance += f"原文链接：{c['source_url']}\n"
            if c.get("verification") == "metadata_only":
                provenance += "仅核验编号、名称和状态，未读取标准全文；不能推定具体条文。\n"
            parts.append(provenance + c["content"])
        return "\n\n".join(parts)


# Backward-compatible export for scripts that inspect the catalog directly.
KNOWLEDGE_DOCS = KnowledgeBase().docs


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    kb = KnowledgeBase()
    queries = ["边坡滑坡红色预警如何处置", "中小矿山监测成本", "滑坡的遥感识别特征"]
    for q in queries:
        print(f"\n查询: {q}")
        for c in kb.retrieve(q, k=2):
            print(f"  → [{c['id']}] {c['title']}")
