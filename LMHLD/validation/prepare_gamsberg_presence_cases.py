"""Use confirmed incident presence without fabricating georeferenced pixel ground truth."""
from pathlib import Path
from datetime import datetime
import json,sys
import numpy as np
import rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from prepare_winter_constructed_normals import display,alignment
OUT=ROOT/'datasets/arid_accident_operation_20261004'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    source=json.loads((OUT/'source_review/dated_satellite_reference.json').read_text(encoding='utf-8'))
    for row in source['sources'].values():
        if row.get('status')!='downloaded' or file_sha256(row['path'])!=row['sha256']:raise ValueError('Original dated reference unavailable/changed')
    if source['capture_date_verified_from_archived_article']!='2020-12-03':raise ValueError('Reference image date missing')
    registration=json.loads((OUT/'ZA_GAMS2020S/reference_registration_v1/registration.json').read_text(encoding='utf-8'))
    rows={}
    for eid,roi in (('ZA_GAMS2020S',[248,438,338,538]),('ZA_GAMS2023W',[130,265,345,460])):
        folder=OUT/eid;dest=folder/'presence_reference_v1';dest.mkdir(exist_ok=True)
        path=dest/'presence_reference.json'
        if path.exists():rows[eid]=json.loads(path.read_text(encoding='utf-8'));continue
        acq=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'));images=[];clear=[]
        for period in ('pre','post'):
            for name,digest in acq['images'][period]['assets_sha256'].items():
                if file_sha256(folder/name)!=digest:raise ValueError('Acquired original source changed')
            image,valid=display(folder/f'{period}_BGRN_10m.tif');images.append(image)
            with rasterio.open(folder/f'{period}_SCL_10m.tif') as f:scl=f.read(1)
            clear.append(valid&~np.isin(scl,[0,1,3,8,9,10,11]))
        shape=images[0].shape[:2];support=np.zeros(shape,np.uint8);x0,y0,x1,y1=roi;support[y0:y1,x0:x1]=1
        # Registration of raw acquisition grids still requires scene ground outside the coarse incident support.
        try:pair=alignment(images[0],images[1],support.astype(bool)|~(clear[0]&clear[1]))
        except ValueError as exc:pair={'raw_pair_gate_pass':False,'reason':str(exc)}
        labels=np.full(shape,255,np.uint8)
        with rasterio.open(folder/'post_BGRN_10m.tif') as f:profile=f.profile.copy()
        with rasterio.open(dest/'pixel_labels_unknown.tif','w',**{**profile,'count':1,'dtype':'uint8','nodata':None}) as f:
            f.write(labels,1);f.update_tags(label_granularity='incident_presence_only',pixel_ground_truth_available='false',all_pixels_unknown='true')
        canvas=Image.new('RGB',(shape[1]*2,shape[0]+28),'white');draw=ImageDraw.Draw(canvas)
        for i,im in enumerate(images):
            frame=Image.fromarray(im);ImageDraw.Draw(frame).rectangle(roi,outline=(255,70,30),width=2)
            canvas.paste(frame,(i*shape[1],28));draw.text((i*shape[1]+5,5),eid+' '+('pre' if i==0 else 'post')+' | box=coarse presence support, not failure boundary',fill='black')
        canvas.save(dest/'coarse_presence_support_review.png')
        record={'reviewed_at':datetime.now().isoformat(timespec='seconds'),'case_id':eid,'mine_group':'ZA_GAMSBERG','role':'development',
          'event_date':'2020-11-17' if eid.endswith('2020S') else '2023-08-03','label_granularity':'case_level_incident_presence',
          'incident_presence':1,'coarse_roi_pixel_edges':roi,'coarse_support_is_not_ground_truth_polygon':True,
          'event_source':'https://www.vedantalimited.com/Vedanta2021/zinc-international.html' if eid.endswith('2020S') else 'https://papers.acg.uwa.edu.au/d/2535_52_Pretorius/52_Pretorius.pdf',
          'source_GIS_transferred':False,'source_registration_failed_does_not_become_pixel_truth':True,
          'reference_image_capture_date':'2020-12-03' if eid.endswith('2020S') else None,
          'target_dates':{k:v['datetime'] for k,v in acq['images'].items()},'post_support_weather_clear_fraction':float(clear[1][support>0].mean()),
          'pair_alignment':pair,'RGB_incident_presence_visually_supported':eid.endswith('2020S'),
          'pre_existing_failure_possible':not eid.endswith('2020S'),'pre_epoch_slide_free_asserted':False,
          '2022_black_polygon_not_2023_truth':not eid.endswith('2020S'),
          'weak_presence_training_eligible':bool(eid.endswith('2020S') and pair['raw_pair_gate_pass'] and clear[1][support>0].mean()>=.95),
          'pixel_segmentation_training_or_IoU_scoring_eligible':False,'weak_presence_confidence':.25,
          'professional_project_expert_feedback':0,'model_predictions_used_for_reference_selection':False,
          'original_acquisition_sha256':file_sha256(folder/'acquisition.json'),
          'derived_assets_sha256':{a.name:file_sha256(a) for a in (dest/'pixel_labels_unknown.tif',dest/'coarse_presence_support_review.png')}}
        write(path,record);rows[eid]=record
    write(OUT/'incident_presence_catalogue.json',{'created_at':datetime.now().isoformat(timespec='seconds'),'new_mine_groups':1,
      'source_confirmed_incident_dates':2,'cases':rows,'all_pixels_remain_unknown_without_accepted_boundary':True,'new_reserved_predictions':0})
    print(json.dumps({e:{'weak_presence_eligible':r['weak_presence_training_eligible'],'post_clear':r['post_support_weather_clear_fraction'],'pair':r['pair_alignment']} for e,r in rows.items()}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
