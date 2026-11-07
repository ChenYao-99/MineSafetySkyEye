"""Scientific diagnosis guards: thresholds, geometry, context and seed units."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import miss_attribution as diag


class AttributionTests(unittest.TestCase):
    def test_response_categories_are_distinct(self):
        mask=np.ones((2,3),bool)
        p=np.array([[.9,.9,.9],[0,0,0]],np.float32);q=np.zeros_like(p);g=np.zeros_like(p)
        self.assertEqual(diag.response_category(p,q,g,mask,.7),'routing_loss')
        p=np.array([[.9,.9,0],[0,0,0]],np.float32);q=np.array([[0,0,.9],[.9,0,0]],np.float32)
        self.assertEqual(diag.response_category(p,q,g,mask,.7),'complementarity_loss')
        p[:]=.6;q[:]=.6
        self.assertEqual(diag.response_category(p,q,g,mask,.7),'threshold_gap')
        p[:]=.2;q[:]=.2
        self.assertEqual(diag.response_category(p,q,g,mask,.7),'weak_branch_response')
        g[:]=.9
        self.assertEqual(diag.response_category(p,q,g,mask,.7),'hit')

    def test_half_coverage_and_threshold_are_inclusive(self):
        mask=np.ones((2,2),bool);p=np.array([[.7,.7],[.1,.1]],np.float32)
        # Float32 .7 is below the exact decimal threshold, matching prior float64 comparison.
        self.assertLess(diag.coverage(p,mask,.7),.5)
        self.assertEqual(diag.coverage(p,mask,float(np.float32(.7))),.5)

    def test_label_oracle_is_upper_bound_for_convex_gate(self):
        rng=np.random.default_rng(2);p=rng.random((32,32));q=rng.random((32,32));a=rng.random((32,32));mask=np.ones((32,32),bool)
        self.assertLessEqual(diag.coverage((1-a)*p+a*q,mask,.7),diag.coverage(np.maximum(p,q),mask,.7))

    def test_tiling_is_in_bounds_and_covers_entire_parent(self):
        for side in (32,64,128,224):
            counts=np.zeros((side,side),np.int32)
            for x0,y0,x1,y1 in diag.boxes(side):
                self.assertGreaterEqual(min(x0,y0),0);self.assertLessEqual(max(x1,y1),side)
                self.assertEqual(x1-x0,min(side,64));counts[y0:y1,x0:x1]+=1
            self.assertTrue(np.all(counts>0))

    def test_quadrant_restore_order(self):
        full=np.zeros((64,64),np.float32);local=np.stack([np.full((64,64),v,np.float32) for v in [.1,.2,.3,.4]])
        p,q,t=diag.restore(full,local,full,128)
        self.assertAlmostEqual(float(q[10,10]),.1);self.assertAlmostEqual(float(q[10,100]),.2)
        self.assertAlmostEqual(float(q[100,10]),.3);self.assertAlmostEqual(float(q[100,100]),.4)

    def test_subpixel_is_not_silently_zero_label(self):
        mask=np.zeros((224,224),bool);mask[30,30]=True;r=diag.occupancy(mask)
        self.assertTrue(r['both_views_subpixel']);self.assertGreater(r['global_peak_occupancy'],0)
        self.assertGreater(r['local_peak_occupancy'],r['global_peak_occupancy'])

    def test_internal_seam_is_recorded(self):
        mask=np.zeros((128,128),bool);mask[60:68,60:68]=True;r=diag.occupancy(mask)
        self.assertEqual(r['quadrant_pieces'],4);self.assertAlmostEqual(r['largest_quadrant_fraction'],.25)

    def test_constant_image_paired_views_identical(self):
        image=np.ones((4,224,224),np.float32)*.3
        a,b=diag.paired_inputs(image);np.testing.assert_array_equal(a,b)

    def test_high_frequency_pair_loses_detail_at_same_fov(self):
        board=(np.indices((64,64)).sum(0)%2).astype(np.float32);image=np.stack([board]*4)
        a,b=diag.paired_inputs(image)
        self.assertEqual(a.shape,b.shape);self.assertGreater(float(a.var()),float(b.var())+.1)

    def test_contrast_excludes_other_positive_objects(self):
        image=np.zeros((4,32,32),np.float32);image[:,10:12,10:12]=1
        mask=np.zeros((32,32),bool);mask[10:12,10:12]=True
        r=diag.contrast_measure(image,image,image,mask,np.ones((32,32),np.uint8),3)
        self.assertEqual(r['ring_pixels'],0);self.assertIsNone(r['native_contrast'])


if __name__=='__main__':unittest.main()
