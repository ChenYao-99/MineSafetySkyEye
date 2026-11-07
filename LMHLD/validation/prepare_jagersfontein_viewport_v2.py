"""Correct the source viewport bottom to the actual imagery edge, preserving failedv1."""
from pathlib import Path
ROOT=Path('D:/LMHLD');BASE=ROOT/'validation/register_jagersfontein_official_partial_map.py';OUT=ROOT/'validation/register_jagersfontein_official_partial_map_v2.py'
text=BASE.read_text(encoding='utf-8')
if text.count('CROP=(48,48,3025,2430)')!=1:raise ValueError('Unexpected original viewport')
text=text.replace('CROP=(48,48,3025,2430)','CROP=(48,48,3025,2420)').replace('official_reference_registration_v1','official_reference_registration_v2_actual_viewport')
if OUT.exists() and OUT.read_text(encoding='utf-8')!=text:raise ValueError('V2code changed')
OUT.write_text(text,encoding='utf-8');print('Newsourceviewport stops at actualmapbottom,notwhitefooter;croppededge remains unknown')
