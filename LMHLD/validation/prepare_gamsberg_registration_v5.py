"""Create a new registration version after independently reading original printed axes.

v2-v4 failures remain intact. Tick texts come from magnified source labels, never from
the fitted affine. No fit threshold, validation allocation, detector, or polygon changes.
"""
from pathlib import Path
import hashlib,json,sys
from PIL import Image
ROOT=Path('D:/LMHLD');BASE=ROOT/'validation/register_gamsberg_published_boundary_v4.py'
OUT=ROOT/'validation/register_gamsberg_published_boundary_v5.py'
SRC=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image13.png'
QA=SRC.parents[1]
for number,box in ((1,(0,100,26,260)),(2,(0,320,26,480))):
 with Image.open(SRC) as im:
  im.convert('RGB').crop(box).rotate(270,expand=True).resize((960,156),Image.Resampling.NEAREST).save(QA/f'gamsberg_latitude_tick{number}_original_upright.png')
text=BASE.read_text(encoding='utf-8')
old='from register_gamsberg_published_boundary_v2 import SOURCE,CASE,POLYGON,AXES,printed_prior,write'
new='''from register_gamsberg_published_boundary_v2 import SOURCE,CASE,POLYGON,write
from rasterio.warp import transform as coordinates
AXES={'longitude':{'source_x':[224.,454.,684.],'degrees':[18.95,18.975,19.]},
      'latitude':{'source_y':[198.,400.],'degrees':[-(29+14/60+17/3600),-(29+15/60+36/3600)]}}
def printed_prior(points,frame):
    lon=np.polyval(np.polyfit(AXES['longitude']['source_x'],AXES['longitude']['degrees'],1),points[:,0])
    lat=np.polyval(np.polyfit(AXES['latitude']['source_y'],AXES['latitude']['degrees'],1),points[:,1])
    x,y=coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist());inverse=~frame.transform
    return np.array([inverse*(a,b) for a,b in zip(x,y)])-.5
'''
if text.count(old)!=1:raise ValueError('Unexpected source version')
text=text.replace(old,new).replace('reference_registration_v4_wang2026_ground_support','reference_registration_v5_wang2026_verified_axes')
if OUT.exists() and OUT.read_text(encoding='utf-8')!=text:raise ValueError('Existing v5 code changed')
OUT.write_text(text,encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
value={'source_sha256':sha(SRC),'source_tick_pixel_positions':{'longitude':[224,454,684],'latitude':[198,400]},
       'verified_original_printed_latitudes':['29deg14min17secS','29deg15min36secS'],
       'earlier_wrong_readings_not_used':['29deg14min7secS','29deg15min30secS'],
       'basis':'magnified intact source label pixels; not inferred geographic values from the fitted matrix',
       'source_label_previews':{str(QA/f'gamsberg_latitude_tick{i}_original_upright.png'):sha(QA/f'gamsberg_latitude_tick{i}_original_upright.png') for i in (1,2)},
       'v4_code_sha256':sha(BASE),'v5_code_sha256':sha(OUT),'acceptance_thresholds_unchanged':True,'accident_detector_predictions':0}
(QA/'gamsberg_original_graticule_verification.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print('New verified-axis version saved; historical failed registrations preserved')
