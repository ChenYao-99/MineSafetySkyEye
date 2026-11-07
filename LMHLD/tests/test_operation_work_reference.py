import sys,unittest
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from operation_work_reference import source_mine_clearing_geometries,derive_work_core

class OperationReferenceTests(unittest.TestCase):
    def test_source_class_filter_and_uncertainty_visibility_remain_unknown(self):
        native={'type':'Polygon','coordinates':[[[100,100],[400,100],[400,400],[100,400],[100,100]]]}
        geo=transform_geom('EPSG:3857','EPSG:4326',native)
        collection={'features':[{'properties':{'objectid':1,'class':'Clearing - Mine'},'geometry':geo},
                                {'properties':{'objectid':2,'class':'Clearing - Infrastructure'},'geometry':geo}]}
        selected=source_mine_clearing_geometries(collection);self.assertEqual([x[0] for x in selected],[1])
        pre=np.full((50,50,4),.2,np.float32);pre[:,:,3]=.8
        post=np.full_like(pre,.3);visible=np.ones((50,50),bool);visible[20:25,20:25]=False
        area,interior,core,counts=derive_work_core(selected,CRS.from_epsg(3857),from_origin(0,500,10,10),pre,post,visible)
        self.assertLess(interior.sum(),area.sum());self.assertFalse(core[~visible].any())
        self.assertFalse(core[~interior].any());self.assertGreater(core.sum(),0)
        self.assertFalse(counts['normal_accident_absence_or_stability_ground_truth'])
        self.assertFalse(counts['exact_per_polygon_clearing_dates_verified'])

    def test_no_actual_observed_change_cannot_become_work_change_target(self):
        image=np.full((20,20,4),.3,np.float32);visible=np.ones((20,20),bool)
        _,_,core,_=derive_work_core([],CRS.from_epsg(3857),from_origin(0,200,10,10),image,image,visible)
        self.assertFalse(core.any())

if __name__=='__main__':unittest.main(verbosity=2)
