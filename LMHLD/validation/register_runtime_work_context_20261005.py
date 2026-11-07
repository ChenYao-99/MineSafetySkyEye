from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/slats_operation_development_20261005';CASE=DATA/'AU_AMRUN_SLATS2324'

def run():
    acceptance=json.loads((DATA/'work_reference_acceptance.json').read_text(encoding='utf-8'))
    accepted=acceptance['accepted']['AU_AMRUN_SLATS2324'];reference=Path(accepted['reference'])
    assert file_sha256(reference)==accepted['reference_sha256']
    row=json.loads(reference.read_text(encoding='utf-8'));mask=reference.parent/'expected_work_mask.tif'
    assert file_sha256(mask)==row['frozen_assets'][mask.name]
    package=ROOT/'experiments/government_work_runtime_validation_20261005/post_original_BGRN_with_original_SCL.tif'
    value={'scope':'exact-input observed development reference;no arbitrary unseen image interpretation',
           'entries':{file_sha256(p):{'reference':str(reference),'reference_sha256':accepted['reference_sha256'],
                    'mask':str(mask),'mask_sha256':file_sha256(mask),
                    'source_url':'https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Environment/SLATS/MapServer/255'}
                    for p in (CASE/'post_BGRN_10m.tif',package)},
           'source_is_not_geotechnical_safety_or_daily_cause_truth':True,'prediction_suppression':False}
    (DATA/'runtime_work_context_registry.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('2 verified encodings of same observed source registered;no threshold/mask suppression or arbitrary-image source assumption')

if __name__=='__main__':run()
