"""Check native imagery against original mapped outlines before pixel admission."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation'),str(ROOT/'cache/research_dependencies')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio,cv2
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape
from PIL import Image,ImageDraw
from prepare_winter_constructed_normals import alignment
OUT=ROOT/'datasets/accident_boundary_pursuit_20261007/Kerala2018'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan');parser.add_argument('--input');args=parser.parse_args()
    plan_path=Path(args.plan).resolve() if args.plan else OUT/'native_pair_plan.json'
    inputs=Path(args.input).resolve() if args.input else OUT
    assert plan_path.is_relative_to(OUT) and inputs.is_relative_to(OUT)
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    rows=json.loads((OUT/'original_vector_correspondence.json').read_text(encoding='utf-8'))['rows'];results=[]
    for key,case in plan['cases'].items():
        folder=inputs/key;acq=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'));values=[];visible=[];grids=[];rgb=[]
        for epoch in ['pre','post']:
            meta=acq['images'][epoch]
            for name,digest in meta['assets_sha256'].items():assert sha(folder/name)==digest
            with rasterio.open(folder/(epoch+'_BGRN_native_10m.tif')) as src:
                raw=src.read().astype(np.float32).transpose(1,2,0);values.append(raw*np.array(src.scales,np.float32)+np.array(src.offsets,np.float32))
                mask=np.all(src.read_masks()>0,axis=0)&np.all(raw>0,axis=2);grids.append((src.crs,src.transform,src.shape));profile=src.profile.copy()
            with rasterio.open(folder/(epoch+'_SCL_on_10m_grid.tif')) as scl:mask&=~np.isin(scl.read(1),[0,1,3,6,7,8,9,10,11])
            visible.append(mask);display=np.uint8(255*np.power(np.clip(values[-1][:,:,[2,1,0]]/.45,0,1),.65));display[~mask]=[90,0,110];rgb.append(display)
        assert grids[0]==grids[1]
        selected=[r for r in rows if r['mine_group']==key];geoms=[]
        for row in selected:
            path=Path(row['geometry_file']);assert sha(path)==row['geometry_file_sha256']
            data=json.loads(path.read_text(encoding='utf-8'))
            for feature in data['features']:geoms.append(transform_geom('EPSG:4326',profile['crs'],feature['geometry']))
        polygon=rasterize([(geom,1) for geom in geoms],out_shape=grids[0][2],transform=grids[0][1],dtype='uint8')>0
        assert polygon.any()
        common=visible[0]&visible[1]
        try:registration=alignment(rgb[0],rgb[1],polygon|~common)
        except ValueError as error:registration={'raw_pair_gate_pass':False,'reason':str(error),'not_a_displacement_measurement':True}
        qa=folder/'geometry_review';qa.mkdir(exist_ok=True)
        reference=qa/'original_mapped_outline_on_native_grid.tif'
        with rasterio.open(reference,'w',**{**profile,'count':1,'dtype':'uint8','nodata':None}) as dst:
            dst.write(polygon.astype(np.uint8),1);dst.update_tags(role='original_NRSC_inventory_geometry_review_only',pixel_accident_labels_accepted='false',classifier_predictions='0')
        h,w=polygon.shape;scale=4;canvas=Image.new('RGB',(w*2*scale,h*scale+60),'white');draw=ImageDraw.Draw(canvas)
        contours,_=cv2.findContours(polygon.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for i,epoch in enumerate(['pre','post']):
            display=rgb[i].copy();cv2.drawContours(display,contours,-1,(255,230,0),1)
            canvas.paste(Image.fromarray(display).resize((w*scale,h*scale),Image.Resampling.NEAREST),(i*w*scale,60))
            draw.text((i*w*scale+5,5),key+' '+epoch+' '+acq['images'][epoch]['datetime'][:10],fill='black')
            draw.text((i*w*scale+5,24),'Yellow: original mapped outline;purple: obscured/invalid',fill='black')
        canvas.save(qa/'original_boundary_native_pair_QA.png')
        fractions={'pre':float(visible[0][polygon].mean()),'post':float(visible[1][polygon].mean()),'common':float(common[polygon].mean())}
        row={'case_id':key,'original_source_polygon_count':len(selected),'original_full_ring_geometries_received':True,
            'native_dates':{e:acq['images'][e]['datetime'] for e in ['pre','post']},'event_date_granularity':'August2018regional episode;individual day not present',
            'original_boundary_sensor_exact_acquisition_datetime_available':False,'native_grid_equal_and_multispectral_not_resampled':True,
            'mapped_outline_pixels':int(polygon.sum()),'visibility_fraction_on_original_outline':fractions,
            'whole_window_common_visible_fraction':float(common.mean()),'independent_stable_control_registration':registration,
            'original_georeferenced_published_inventory_received':True,'original_professional_ground_survey_vector_received':False,
            'quarry_in_use_at_source_point_author_code':'QUU','strict_active_highwall_failure_not_assumed':True,
            'mask_path':str(reference),'mask_sha256':sha(reference),'QA_image':str(qa/'original_boundary_native_pair_QA.png'),
            'pixel_reference_admitted':False,'visual_review_pending':True,'new_CNN_predictions_or_training':0,
            'scope':'source geometry/observation check,not model performance or independent professional review'}
        write(qa/'qualification.json',row);results.append(row)
        print(key,'original polygons',len(selected),'nativepixels',row['mapped_outline_pixels'],'visibility',fractions,'registration',registration,flush=True)
    write(inputs/'native_geometry_qualification.json',{'cases':results,'visual_review_pending':True,'model_predictions':0,'new_admitted_full_accident_pixel_truth':0})


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
