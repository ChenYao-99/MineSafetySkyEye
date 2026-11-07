# -*- coding: utf-8 -*-
import copy,json,unittest
from pathlib import Path
import numpy as np
from import_independent_boundary_review import validate,sha,DATA


class IndependentReviewImportTests(unittest.TestCase):
    def record(self):
        folder=DATA/'development/LasCruces';p=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))['images']['post'];t=p['transform'];points=[[20,20],[50,20],[50,50],[20,50]]
        return {'schema':'mine_independent_review_v1','reviewer':'UNIT_TEST_UNVERIFIED_REVIEWER','notes':'unit test only, never saved as real feedback',
            'expert_identity_verified':True,'model_outputs_shown_by_tool':False,'author_contours_shown_by_tool':False,
            'previous_prediction_or_reference_exposure':'unknown','cases':[{'case_id':'LasCruces','post_sha256':sha(folder/'post_BGRN_10m.tif'),
                'post_date':p['datetime'],'crs':p['crs'],'shape':p['shape'][1:],'transform':t,
                'polygons':[{'kind':'uncertain','vertices_pixel':points,'vertices_projected':[[t[0]*x+t[1]*y+t[2],t[3]*x+t[4]*y+t[5]] for x,y in points]}]}]}

    def test_coordinate_conversion_does_not_verify_identity_or_train(self):
        result=validate(self.record());self.assertFalse(result['properties']['expert_identity_verified'])
        self.assertEqual(result['properties']['new_verified_expert_reviews'],0)
        lon,lat=result['features'][0]['geometry']['coordinates'][0][0];self.assertTrue(-6.2<lon<-6.0 and 37.4<lat<37.6)
        self.assertFalse(result['properties']['frozen_training_labels_modified'])

    def test_reserved_case_or_changed_image_is_rejected(self):
        for changed in ('case_id','post_sha256'):
            record=self.record();record['cases'][0][changed]='CL0002' if changed=='case_id' else '0'*64
            with self.assertRaises(ValueError):validate(record)

    def test_bow_tie_and_inconsistent_projected_vertices_are_rejected(self):
        record=self.record();record['cases'][0]['polygons'][0]['vertices_pixel']=[[20,20],[50,50],[20,50],[50,20]]
        with self.assertRaisesRegex(ValueError,'Self-intersecting'):validate(record)
        record=self.record();record['cases'][0]['polygons'][0]['vertices_projected'][0][0]+=10
        with self.assertRaisesRegex(ValueError,'disagree'):validate(record)

    def test_automated_smoke_export_cannot_become_real_feedback(self):
        record=self.record();record['automated_smoke_not_real_review']=True
        with self.assertRaisesRegex(ValueError,'not real review'):validate(record)


if __name__=='__main__':unittest.main(verbosity=2)
