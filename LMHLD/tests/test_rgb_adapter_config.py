"""Reject unverified, corrupted or out-of-root adapter components before inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgb_adapter import load_guarded_adapter


class AdapterConfig(unittest.TestCase):
    def setUp(self):
        directory=ROOT/"cache/tests"
        directory.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config={"schema_version":1,"regression_passed":True,"adapter_weight":.15,
            "components":{"current":{"path":"current.pth","sha256":"0"*64},
                          "adapter":{"path":"new.pth","sha256":"0"*64}}}

    def invoke(self):
        path=self.root/"adapter.json"
        path.write_text(json.dumps(self.config),encoding="utf-8")
        return load_guarded_adapter(path,self.root)

    def test_unverified_config_is_not_inferred(self):
        self.config["regression_passed"]=False
        with self.assertRaisesRegex(ValueError,"regression"):
            self.invoke()

    def test_component_path_cannot_escape_model_root(self):
        self.config["components"]["current"]["path"]="../outside.pth"
        with self.assertRaisesRegex(ValueError,"inside models_root"):
            self.invoke()

    def test_weight_hash_mismatch_is_not_silently_accepted(self):
        (self.root/"current.pth").write_bytes(b"changed-weight-bytes")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            self.invoke()

    def test_malformed_component_has_clear_error(self):
        self.config["components"]["current"]={"path":None}
        with self.assertRaisesRegex(ValueError,"Invalid RGB adapter component"):
            self.invoke()


if __name__=="__main__":
    unittest.main(verbosity=2)
