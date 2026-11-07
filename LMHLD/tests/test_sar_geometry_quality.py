from pathlib import Path
import sys,unittest
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
from sar_geometry_quality import equivalent_ground_shift,historical_local_geometry

def test_source_index_jacobian_inverts_ground_coordinates_with_north_up_sign():
    y,x=np.indices((7,9));gx=x*30.;gy=-y*30.
    line=.02*gx+.1*gy;pixel=.12*gx-.01*gy
    dx,dy,good,_=equivalent_ground_shift(line,pixel,np.full(line.shape,.02*40+.1*(-25)),np.full(line.shape,.12*40-.01*(-25)))
    assert good[1:-1,1:-1].all()
    np.testing.assert_allclose(dx[good],40);np.testing.assert_allclose(dy[good],-25)

def test_singular_geometry_and_missing_observations_stay_unknown():
    y,x=np.indices((7,9));line=x.astype(float);pixel=2*line
    dx,dy,good,_=equivalent_ground_shift(line,pixel,np.ones(line.shape),np.ones(line.shape))
    assert not good.any() and np.isnan(dx).all() and np.isnan(dy).all()
    line=.1*y+.05*x;pixel=.03*y+.1*x;line[3,3]=np.nan
    _,_,good,_=equivalent_ground_shift(line,pixel,np.ones(line.shape),np.ones(line.shape))
    assert not good[3,2] and not good[3,4]

def test_historic_slope_screen_detects_active_layover_and_local_backface():
    y,x=np.indices((7,9));theta=np.radians(35);le=np.full(x.shape,np.sin(theta));ln=np.zeros(x.shape);lu=np.full(x.shape,np.cos(theta))
    flat=historical_local_geometry(np.zeros(x.shape),le,ln,lu)
    assert not flat['active_layover_risk'].any() and not flat['local_backface_risk'].any()
    np.testing.assert_allclose(flat['local_incidence_deg'],35)
    toward=historical_local_geometry(-x*30.*np.tan(np.radians(45)),le,ln,lu)
    assert toward['active_layover_risk'].all()
    away=historical_local_geometry(x*30.*np.tan(np.radians(65)),le,ln,lu)
    assert away['local_backface_risk'].all()

class GeometryQualityTests(unittest.TestCase):
    def test_analytic_ground_coordinates(self):
        test_source_index_jacobian_inverts_ground_coordinates_with_north_up_sign()
    def test_unknown_singular_or_missing_geometry(self):
        test_singular_geometry_and_missing_observations_stay_unknown()
    def test_historic_slope_geometry(self):
        test_historic_slope_screen_detects_active_layover_and_local_backface()

if __name__=='__main__':unittest.main()
