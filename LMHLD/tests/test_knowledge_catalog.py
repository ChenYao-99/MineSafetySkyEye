import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase


class KnowledgeCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()

    def test_source_provenance_and_legacy_ids(self):
        identifiers = {d["id"] for d in self.kb.docs}
        self.assertTrue({"REG-001", "REG-002", "REG-003", "KNOW-001", "KNOW-002", "WARN-001",
                         "WARN-002", "ACT-001", "ACT-002", "CASE-001", "CASE-002"} <= identifiers)
        for d in self.kb.docs:
            self.assertTrue(d["locator"] and d["checked_at"] and d["evidence_type"])
            self.assertTrue(d["source_url"] or d["verification"] == "implementation")

    def test_exact_standard_and_correct_name(self):
        results = self.kb.retrieve("GB/T 37573是什么标准", k=3)
        self.assertEqual(results[0]["id"], "REG-002")
        self.assertIn("年度评价", results[0]["source_title"])

    def test_retrieval_covers_real_questions(self):
        for query, expected in [("200米采场边坡在线监测", "REG-009"),
                                ("RGB图片能否计算NDVI", "DATA-004"),
                                ("两期影像配准", "GEO-004"),
                                ("72小时位移加速", "REG-017"),
                                ("煤矿2026年重大事故隐患标准", "REG-014")]:
            with self.subTest(query=query):
                self.assertIn(expected, {d["id"] for d in self.kb.retrieve(query, k=3)})

    def test_context_carries_ids_and_scope_and_metadata_limit(self):
        doc = next(d for d in self.kb.docs if d["id"] == "REG-002")
        context = self.kb.format_context([doc])
        self.assertIn("[REG-002]", context)
        self.assertIn(doc["source_url"], context)
        self.assertIn("未读取标准全文", context)

    def test_no_match_or_empty_query_has_no_fake_reference(self):
        self.assertEqual(self.kb.retrieve(""), [])
        self.assertEqual(self.kb.retrieve("zzzzunmatchedtoken"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
