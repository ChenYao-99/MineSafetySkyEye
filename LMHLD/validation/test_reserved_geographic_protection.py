import sys,unittest
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from protected_inputs import input_split
from raster_upload import decode_tiff
from validation.mine_data_protocol import registry,DATA

class GeographicProtectionTests(unittest.TestCase):
    def test_actual_reencoded_crops_without_tags_remain_protected(self):
        v=registry()
        for case_id in ('CL0002','KG0001','US_BC2021','TR_COP2024'):
            source=v['cases'][case_id]['protected_geographic_extent_source']
            with rasterio.open(source) as f:
                win=Window(f.width//2,f.height//2,24,24);raw=f.read(window=win);profile=f.profile.copy()
                profile.update(height=24,width=24,transform=f.window_transform(win),compress='lzw')
                with MemoryFile() as memory:
                    with memory.open(**profile) as out:
                        out.write(raw);out.descriptions=('B02','B03','B04','B08')
                    content=memory.read()
            _,meta=decode_tiff(content,'AUTO',10000)
            self.assertIsNone(meta['declared_case_id'])
            split=input_split(meta);self.assertTrue(split['reserved']);self.assertIn(case_id,split['case_ids'])
            self.assertEqual(split['verified_by'],'registered_reserved_geographic_overlap')
            self.assertFalse(split['identity_verified'])
    def test_fake_development_declaration_cannot_override_geography(self):
        case=registry()['cases']['CL0002'];a,b,c,d=case['protected_geographic_extent_wgs84']
        meta={'declared_case_id':'CN_XJ2023','georeference':{'crs':'EPSG:4326','width':100,'height':100,
               'transform':[(c-a)/100,0,a,0,-(d-b)/100,d]}}
        self.assertTrue(input_split(meta)['reserved'])
    def test_known_nonreserved_location_and_locationless_image_not_misidentified(self):
        meta={'georeference':{'crs':'EPSG:4326','width':100,'height':100,'transform':[.0001,0,115,0,-.0001,30]}}
        self.assertFalse(input_split(meta)['reserved']);self.assertFalse(input_split({})['reserved'])
        self.assertFalse((DATA/'final_evaluation_once.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
