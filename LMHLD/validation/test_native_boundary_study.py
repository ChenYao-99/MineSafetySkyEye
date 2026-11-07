# -*- coding: utf-8 -*-
"""Meaningful checks for unknown-label gradients, spatial separation and raw bands."""
import ast
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import rasterio

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/native_boundary_20261002"


class NativeBoundaryStudy(unittest.TestCase):
    def test_unknown_pixels_have_no_loss_gradient(self):
        tree=ast.parse((ROOT/"validation/run_native_boundary_study.py").read_text(encoding="utf-8"))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="partial_loss")
        ns={"torch":torch}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"partial_loss","exec"),ns)
        logits=torch.zeros((1,1,2,2),requires_grad=True)
        labels=torch.tensor([[[[1,0],[255,1]]]])
        confidence=torch.tensor([[[[.6,.25],[0,.6]]]])
        loss=ns["partial_loss"](logits,labels,confidence)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[0,0,1,0]),0)
        self.assertLess(float(logits.grad[0,0,0,0]),0)
        self.assertGreater(float(logits.grad[0,0,0,1]),0)
        empty=ns["partial_loss"](logits,torch.full_like(labels,255),torch.zeros_like(confidence))
        self.assertEqual(float(empty),0)

    def test_spatial_training_windows_have_a_32px_validation_halo(self):
        data=json.loads((DATA/"development_data.json").read_text(encoding="utf-8"))
        for train in data["train_windows"]:
            x,y,x1,y1=train["box"]
            for bx,by in data["validation_blocks"]:
                self.assertFalse(x<(bx+1)*128+32 and x1>bx*128-32 and y<(by+1)*128+32 and y1>by*128-32)

    def test_validation_windows_do_not_duplicate_pixels(self):
        plan=json.loads((DATA/"training_protocol.json").read_text(encoding="utf-8"))
        data=json.loads((DATA/"development_data.json").read_text(encoding="utf-8"))
        points=[data["val_windows"][i]["box"] for i in plan["native_validation_indices"]]
        for i,a in enumerate(points):
            for b in points[i+1:]:
                self.assertFalse(a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1])

    def test_partial_schema_and_confidence_remain_distinct(self):
        for split in ("train","val"):
            y=np.load(DATA/f"mine_{split}_y.npy")
            w=np.load(DATA/f"mine_{split}_weight.npy")
            self.assertTrue(set(np.unique(y)).issubset({0,1,255}))
            self.assertTrue((w[y==255]==0).all())
            self.assertTrue((w[y==1]>.5).all())
            self.assertTrue((w[y==0]<.3).all())

    def test_input_is_measured_spectral_data_and_new_tests_are_excluded(self):
        with rasterio.open(ROOT/"datasets/mine_benchmark_20261002/development/Amyntaio/post_BGRN_10m.tif") as src:
            self.assertEqual(src.descriptions,("B02","B03","B04","B08"))
        plan=json.loads((DATA/"training_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["training_mines"],["Amyntaio"])
        self.assertFalse(plan["new_reserved_mines_read_for_prediction"])
        self.assertFalse((ROOT/"datasets/mine_benchmark_20261002/final_evaluation_once.json").exists())


if __name__=="__main__":
    unittest.main(verbosity=2)
