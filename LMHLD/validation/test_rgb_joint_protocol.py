"""Checks event separation, unknown regions, annotation guards and metric selection."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POS = ROOT/"experiments/positive_expansion_20261001"


class RGBJointProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.positive = json.loads((POS/"positive_cases.json").read_text(encoding="utf-8"))
        cls.protocol = json.loads((ROOT/"experiments/rgb_joint_20261001/protocol.json").read_text(encoding="utf-8"))

    def read(self,path):
        return cv2.imdecode(np.fromfile(POS/path,np.uint8),cv2.IMREAD_UNCHANGED)

    def test_event_split_and_weak_label_status(self):
        self.assertEqual({p["event_id"] for p in self.positive["collolar"]["patches"]},{"collolar_201102"})
        self.assertIn("training only",self.positive["collolar"]["role"])
        self.assertIn("held-out",self.positive["hpakant"]["role"])
        self.assertEqual(self.positive["full_expert_audited_test_sets"],0)
        self.assertTrue(self.protocol["no_external_selection"])
        self.assertEqual(self.protocol["replay_weight"],.25)

    def test_unknown_runout_is_never_background(self):
        h = self.positive["hpakant"]
        valid,target = self.read(h["valid"])>0,self.read(h["reference"])>0
        unknown = self.read("cases/hpakant_runout_unknown.png")>0
        self.assertFalse((unknown&valid).any())
        self.assertEqual(int((valid&target).sum()),h["scored_positive_pixels"])
        self.assertEqual(int((valid&~target).sum()),h["scored_background_pixels"])

    def test_scored_tiles_exclude_color_annotations_and_polygon_uncertainty(self):
        h = self.positive["hpakant"]
        image,target,valid = self.read(h["image"]),self.read(h["reference"])>0,self.read(h["valid"])>0
        b,g,r = [c.astype(np.int16) for c in cv2.split(image)]
        red = (r>110)&(r-g>50)&(r-b>50)
        self.assertFalse((red&valid).any())
        edge = cv2.morphologyEx(target.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((21,21),np.uint8))>0
        self.assertFalse((edge&valid).any())
        for y in range(0,len(valid),32):
            for x in range(0,valid.shape[1],32):
                part=valid[y:y+32,x:x+32]
                self.assertTrue(not part.any() or (part.shape==(32,32) and part.all()))

    def test_training_patches_have_no_strip_seams_and_keep_red_proxy(self):
        patches = np.load(POS/"collolar_positive_x.npy")
        np.testing.assert_array_equal(patches[:,3],patches[:,2])
        self.assertEqual(len(patches),self.positive["collolar"]["positive_patch_n"])
        for row in self.positive["collolar"]["patches"]:
            x0,y0,x1,y1=row["box_xyxy"]
            self.assertEqual((x1-x0,y1-y0),(32,32))
            for seam in self.positive["collolar"]["seam_rows_excluded"]:
                self.assertFalse(y0<seam+4 and y1>seam-3)

    def test_frozen_files_match_manifest(self):
        for name,digest in self.positive["frozen_files"].items():
            self.assertEqual(hashlib.sha256((POS/name).read_bytes()).hexdigest(),digest,name)

    def test_eligibility_requires_both_modalities_and_background(self):
        tree=ast.parse((ROOT/"validation/run_rgb_joint_study.py").read_text(encoding="utf-8"))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="eligible")
        ns={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"eligible","exec"),ns)
        def row(nir,rgb,neg):
            return {"source_NIR":{"F1":nir},"source_RGB_proxy":{"F1":rgb},"generic_background":{"block_activation":neg}}
        base=row(.7,.6,.9)
        self.assertTrue(ns["eligible"](row(.65,.58,.05),base))
        self.assertFalse(ns["eligible"](row(.65,.4,.01),base))
        self.assertFalse(ns["eligible"](row(.5,.65,.01),base))
        self.assertFalse(ns["eligible"](row(.65,.58,.3),base))


if __name__=="__main__":
    unittest.main(verbosity=2)
