"""Check auxiliary mining-change data; keep held-out pixels unopened."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import hashlib,json,io
from PIL import Image
import numpy as np

DATA=ROOT/'datasets/data_conditions_20261007'


def run():
    roles=json.loads((DATA/'MineNetCD_site_roles.json').read_text(encoding='utf-8'))
    original=json.loads((DATA/'MineNetCD_subset_acquisition.json').read_text(encoding='utf-8'))
    by_name={row['member']:row for row in original['files']}
    old=json.loads((ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json').read_text(encoding='utf-8'))
    protected=[case['protected_geographic_extent_wgs84'] for case in old['cases'].values() if case.get('role')=='reserved_final_test']
    result=[];derived=[]
    for site,role in roles['roles'].items():
        meta=roles['metadata'][site];box=[float(meta[k]) for k in ['left','bottom','right','top']]
        assert box[0]<box[2] and box[1]<box[3]
        assert not any(min(box[2],b[2])>max(box[0],b[0]) and min(box[3],b[3])>max(box[1],b[1]) for b in protected)
        row={'site':site,'role':role,'author_metadata':meta,'original_file_hashes':{},
             'accident_truth_or_safe_background_admitted':False,'overlap_existing_reserved_extents':False}
        for name in ['im1.jpg','im2.jpg','ref.png']:
            p=DATA/'MineNetCD/MCD'/site/name
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            assert sha==by_name[f'MCD/{site}/{name}']['sha256'];row['original_file_hashes'][name]=sha
        if role.startswith('heldout'):
            row['pixel_inspection_or_model_use']=False
        else:
            sizes=[]
            for name in ['im1.jpg','im2.jpg','ref.png']:
                with Image.open(DATA/'MineNetCD/MCD'/site/name) as im:
                    sizes.append(im.size)
                    if name=='ref.png':
                        values=np.array(im.convert('L'));row['reference_values']=np.unique(values).tolist()
                        assert set(row['reference_values']) <= {0,255}
                        row['author_change_pixels']=int((values==255).sum())
                        row['author_change_fraction']=float((values==255).mean())
            assert sizes[0]==sizes[1]==sizes[2],(site,sizes)
            row['native_dimensions']=sizes[0];row['pixel_inspection_or_model_use']='input QA only;no model prediction'
            # The centre crop is fixed by geometry, never selected by masks/model response.
            width,height=sizes[0];side=min(512,width,height);box_px=((width-side)//2,(height-side)//2,(width+side)//2,(height+side)//2)
            for epoch in ['im1','im2']:
                with Image.open(DATA/'MineNetCD/MCD'/site/(epoch+'.jpg')) as im:
                    cropped=im.crop(box_px).convert('RGB');path=DATA/'engineering_inputs'/site/(epoch+'.png');path.parent.mkdir(parents=True,exist_ok=True);cropped.save(path)
                derived.append({'site':site,'epoch':epoch,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source_file_sha256':row['original_file_hashes'][epoch+'.jpg'],'native_crop_box_px':box_px,
                    'acquisition_date_from_author_catalogue':meta['T1' if epoch=='im1' else 'T2'],
                    'embedded_physical_GSD_NIR_or_georeference':False,'role':'development engineering input;not independent accident test'})
        result.append(row)
    receipt={'passed':True,'source_DOI':'10.14278/rodare.3251','site_count':len(result),
        'development_groups':6,'heldout_auxiliary_change_groups':2,'heldout_pixels_inspected_or_predicted':0,
        'new_accident_or_verified_safety_labels':0,'new_classifier_predictions_or_fits':0,'rows':result,'engineering_inputs':derived,
        'scope':'published high-resolution RGB mining-change data;cause and safety not certified;catalogue extents/dates not independently ground-verified'}
    (DATA/'MineNetCD_qualification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('QUALIFIED',len(result),'source groups;6development;2unopened auxiliary holdouts;no accident truth promotion')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');run()
