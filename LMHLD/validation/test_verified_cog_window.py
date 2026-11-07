import io,sys,unittest,tempfile
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,tifffile
from verified_cog_window import decode_window
class VerifiedCOGWindowTests(unittest.TestCase):
    def test_compressed_predictor_tiles_match_original_across_boundaries(self):
        original=np.random.default_rng(42).integers(0,12000,(117,139),dtype=np.uint16);stream=io.BytesIO()
        tifffile.imwrite(stream,original,tile=(32,32),compression='deflate',predictor=2);stream.seek(0)
        with tifffile.TiffFile(stream) as tf:
            def read_at(offset,n):stream.seek(offset);return stream.read(n)
            actual=decode_window(tf.pages[0],read_at,27,29,75,83)
        self.assertTrue(np.array_equal(actual,original[27:102,29:112]))
    def test_truncated_tile_is_rejected(self):
        stream=io.BytesIO();tifffile.imwrite(stream,np.ones((64,64),np.uint16),tile=(32,32),compression='deflate');stream.seek(0)
        with tifffile.TiffFile(stream) as tf:
            def bad(offset,n):stream.seek(offset);return stream.read(max(0,n-1))
            with self.assertRaisesRegex(ValueError,'truncated'):decode_window(tf.pages[0],bad,0,0,16,16)
    def test_window_cannot_escape_original_native_raster(self):
        stream=io.BytesIO();tifffile.imwrite(stream,np.ones((64,64),np.uint16),tile=(32,32));stream.seek(0)
        with tifffile.TiffFile(stream) as tf:
            with self.assertRaisesRegex(ValueError,'exceeds'):decode_window(tf.pages[0],lambda a,b:b'',50,0,20,20)
if __name__=='__main__':unittest.main(verbosity=2)
