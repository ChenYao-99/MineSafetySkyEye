"""Resolve which source dates are verified and keep unknown boundary dates explicit."""
from pathlib import Path
from datetime import datetime
import json,sys,urllib.request,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
SOURCE=ROOT/'datasets/operational_boundary_sources_20261004/brumadinho'
DATA=ROOT/'datasets/mine_benchmark_20261002'
quality=DATA/'development/Brumadinho/exact_date_quality_20261004'
q=json.loads((quality/'manifest.json').read_text(encoding='utf-8'))
for period,date in (('pre','2019-01-07'),('post','2019-02-01')):
    row=q['images'][period]
    if row['datetime'][:10]!=date:raise ValueError('Exact original image date mismatch')
    for name,digest in row['assets'].items():
        if file_sha256(quality/name)!=digest:raise ValueError('Exact-date quality asset changed')
xml=ET.parse(SOURCE/'official_impact_metadata.xml').getroot()
dates=[]
for node in xml.iter():
    key=node.tag.rsplit('}',1)[-1]
    if key in ('Date','DateTime','beginPosition','endPosition') and (node.text or '').strip():dates.append({'field':key,'value':node.text.strip()})
attempts=json.loads((SOURCE/'ibge_original_attempts/attempts.json').read_text(encoding='utf-8'))
thumb_url='https://idesisema.meioambiente.mg.gov.br/geonetwork/srv/api/records/202cf333-3337-4f7d-91a8-68ee9f1b93ad/attachments/ide_250102_mg_impactos_ambientais_img.png'
thumbnail={'url':thumb_url}
try:
    with urllib.request.urlopen(thumb_url,timeout=25) as r:raw=r.read(4*1024*1024+1)
    if len(raw)>4*1024*1024 or not raw.startswith(b'\x89PNG'):raise ValueError('Invalid official preview')
    path=SOURCE/'official_metadata_map_preview.png';path.write_bytes(raw);thumbnail.update(path=str(path),sha256=file_sha256(path))
except Exception as exc:thumbnail['error']=str(exc)
p={'reviewed_at':datetime.now().isoformat(timespec='seconds'),'event_id':'BR0001','scope':'tailings-dam auxiliary; no strict pit-wall final acceptance',
   'official_SEMAD_Pleiades_metadata_sha256':file_sha256(SOURCE/'official_impact_metadata.xml'),'official_SEMAD_metadata_dates':dates,
   'SEMAD_boundary_image_capture_date_verified':False,'SEMAD_source_dates_are_publication_metadata_not_capture':True,
   'IBGE_notes_sha256':file_sha256(SOURCE/'ibge_notes.pdf'),'IBGE_named_boundary_image_dates':attempts['notes_image_dates'],
   'IBGE_actual_dated_boundary_obtained':attempts['actual_dated_boundary_obtained'],
   'IBGE_date_cannot_be_assigned_to_SEMAD_vector':True,'IBGE_original_access_attempts_sha256':file_sha256(SOURCE/'ibge_original_attempts/attempts.json'),
   'original_satellite_pre_date_verified':'2019-01-07','original_satellite_post_date_verified':'2019-02-01',
   'exact_date_quality_manifest_sha256':file_sha256(quality/'manifest.json'),
   'quality_metadata_is_not_manual_visibility_ground_truth':True,'official_map_preview':thumbnail,
   'remaining_gap':'actual SEMAD Pleiades acquisition date or accessible original dated IBGE geometry; publication2020 and metadata2023 do not identify image date',
   'automatic_training_or_scoring_acceptance':False,'new_model_predictions':0,'original_images_or_prior_labels_changed':False}
out=DATA/'development/Brumadinho/official_boundary_review_v1_20261004/boundary_date_verification.json'
out.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps({'source_dates':dates,'date_verified':False,'actual_IBGE_geometry':attempts['actual_dated_boundary_obtained'],'preview':thumbnail},ensure_ascii=False))
