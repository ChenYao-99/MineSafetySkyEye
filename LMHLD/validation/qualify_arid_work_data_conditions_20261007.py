"""Independent annual mining class plus actual optical observations; no safety truth."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation'),str(ROOT/'cache/research_dependencies')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import json,hashlib
import numpy as np,rasterio,cv2
from PIL import Image,ImageDraw
from operation_work_reference import derive_work_core,source_mine_clearing_geometries
from prepare_winter_constructed_normals import alignment

DATA=ROOT/'datasets/data_conditions_20261007/SLATS_arid_pairs'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    plan=json.loads((DATA/'site_plan.json').read_text(encoding='utf-8'));rows=[]
    for site in plan['sites']:
        folder=DATA/site;source=DATA/(site+'_official_mining_polygons.geojson')
        geoms=source_mine_clearing_geometries(json.loads(source.read_text(encoding='utf-8')))
        acquisition=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))
        values=[];visible=[];rgb=[];grids=[]
        for epoch in ['pre','post']:
            record=acquisition['images'][epoch]
            assert sha(folder/(epoch+'_BGRN_10m.tif'))==record['sha256']
            assert sha(folder/(epoch+'_SCL_10m.tif'))==record['scl_sha256']
            with rasterio.open(folder/(epoch+'_BGRN_10m.tif')) as image:
                raw=image.read().transpose(1,2,0).astype(np.float32)
                values.append(raw*np.array(image.scales,np.float32)+np.array(image.offsets,np.float32))
                valid=np.all(image.read_masks()>0,axis=0)&np.all(raw>0,axis=2)
                grids.append((image.crs,image.transform,image.width,image.height));profile=image.profile.copy()
            with rasterio.open(folder/(epoch+'_SCL_10m.tif')) as scl:valid&=~np.isin(scl.read(1),[0,1,3,6,7,8,9,10,11])
            visible.append(valid)
            display=np.uint8(255*np.power(np.clip(values[-1][:,:,[2,1,0]]/.45,0,1),.65));display[~valid]=[80,0,100];rgb.append(display)
        assert grids[0]==grids[1]
        common=visible[0]&visible[1]
        area,interior,core,counts=derive_work_core(geoms,grids[0][0],grids[0][1],values[0],values[1],common)
        try:quality=alignment(rgb[0],rgb[1],~common|area)
        except ValueError as error:quality={'raw_pair_gate_pass':False,'reason':str(error),'not_a_displacement_measurement':True}
        passed=quality['raw_pair_gate_pass'] and common.mean()>=.6 and core.sum()>=100
        output=folder/'source_work_reference';output.mkdir(exist_ok=True)
        mask=output/'expected_work_mask.tif'
        with rasterio.open(mask,'w',**{**profile,'count':1,'dtype':'uint8','nodata':None}) as dst:dst.write(core.astype(np.uint8),1)
        h,w=common.shape;canvas=Image.new('RGB',(w*2,h+40),'white');draw=ImageDraw.Draw(canvas)
        for index,epoch in enumerate(['pre','post']):
            display=rgb[index].copy();display[core]=np.uint8(.65*display[core]+.35*np.array([0,255,100]))
            contours,_=cv2.findContours(area.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(display,contours,-1,(255,210,0),1)
            canvas.paste(Image.fromarray(display),(index*w,40));draw.text((index*w+4,4),site+' '+epoch+' '+acquisition['images'][epoch]['datetime'][:10],fill='black')
        canvas.save(output/'source_observation_QA.png')
        row={'site_id':site,'mine_group':site,'status':'qualified_pending_visual_review' if passed else 'not_admitted',
            'source_nominal_period':plan['source_nominal_period'],'actual_source_dates':{e:acquisition['images'][e]['datetime'] for e in ['pre','post']},
            'actual_dates':{e:acquisition['images'][e]['datetime'] for e in ['pre','post']},
            'confidence':.25,'source_url':plan['source_url'],'source_geojson_sha256':sha(source),
            'official_mining_clearing_object_ids':plan['sites'][site]['official_mining_clearing_object_ids'],
            'common_visible_fraction':float(common.mean()),'pair_optical_registration_QA':quality,'counts':counts,
            'expected_work_pixels':int(core.sum()),'same_native_grid':True,'raw_images_warped':False,
            'exact_daily_cause_or_slope_safety_or_accident_absence_verified':False,
            'accident_negative_labels_generated':False,'new_classifier_predictions_or_training':0,
            'allowed_use':'source-backed lowconfidence expected-work context;not complete mine safety or field false-alarm truth',
            'mask_path':str(mask),'mask_sha256':sha(mask),'QA_image':str(output/'source_observation_QA.png')}
        (output/'reference.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');rows.append(row)
        print(site,row['status'],row['expected_work_pixels'],round(row['common_visible_fraction'],3),quality,flush=True)
    (DATA/'qualification.json').write_text(json.dumps({'rows':rows,'visual_review_pending':True,'training_or_prediction_calls':0},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
