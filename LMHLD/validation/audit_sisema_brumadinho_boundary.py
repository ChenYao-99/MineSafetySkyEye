"""Native-grid public-boundary comparison; no model, label overwrite or safety inference."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape,mapping,box
from shapely.ops import unary_union
from rasterio.warp import transform_geom
from PIL import Image,ImageDraw
sys.path.insert(0,str(ROOT))
from mine_data_protocol import assert_development_cases,verify_assets
from model_registry import file_sha256
DATA=ROOT/'datasets/mine_benchmark_20261002'
SOURCE=ROOT/'datasets/operational_boundary_sources_20261004/brumadinho'
OUT=DATA/'development/Brumadinho/official_boundary_review_v1_20261004'
RAW=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes/BR0001'


def run():
    assert_development_cases(['BR0001']);verify_assets();OUT.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((SOURCE/'official_impact_manifest.json').read_text(encoding='utf-8'))
    for row in source_manifest['sources']:
        if row.get('error'):raise ValueError('Official source download incomplete')
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Official source changed')
    value=json.loads((SOURCE/'official_impact_polygons.geojson').read_text(encoding='utf-8'))
    if value['crs']['properties']['name']!='urn:ogc:def:crs:EPSG::4674':raise ValueError('Unexpected original government CRS')
    geo=[shape(f['geometry']) for f in value['features']]
    if any(not g.is_valid or g.is_empty or g.geom_type not in ('Polygon','MultiPolygon') for g in geo):raise ValueError('Invalid official polygon')
    xml=ET.parse(SOURCE/'official_impact_metadata.xml').getroot()
    meta=[' '.join((n.text or '').split()) for n in xml.iter() if (n.text or '').strip()]
    lineage=[s for s in meta if 'Pleiades' in s]
    if not lineage:raise ValueError('Official optical-interpretation lineage absent')
    maskpath=next((p for p in RAW.glob('*.tif') if 'mask' in p.name.lower()),None)
    if maskpath is None:
        masks=[p for p in DATA.rglob('*.tif') if 'BR0001' in p.name and ('mask' in p.name.lower() or 'label' in p.name.lower())]
        if len(masks)!=1:raise ValueError(f'Explicit original mask needed: {masks}')
        maskpath=masks[0]
    with rasterio.open(maskpath) as f:
        published=f.read(1)>0;profile=f.profile.copy();native_crs=f.crs;grid=f.transform;bounds=f.bounds
    native=[shape(transform_geom('EPSG:4674',native_crs,mapping(g))) for g in geo];union=unary_union(native)
    if not union.intersects(box(*bounds)):raise ValueError('Boundary does not intersect original image')
    official=rasterize([(mapping(g),1) for g in native],out_shape=published.shape,transform=grid,dtype='uint8')
    prof={**profile,'count':1,'dtype':'uint8','nodata':None,'compress':'deflate'}
    maskdest=OUT/'government_impact_outline_on_original_grid.tif'
    with rasterio.open(maskdest,'w',**prof) as f:
        f.write(official,1);f.update_tags(role='public_auxiliary_boundary_for_review_only',labels_accepted='false',no_model_predictions='true',source_CRS='EPSG:4674')
    inter=int((published&(official>0)).sum());uni=int((published|(official>0)).sum())
    minx,miny,maxx,maxy=union.bounds
    inv=~grid;x0,y0=inv*(minx-300,maxy+300);x1,y1=inv*(maxx+300,miny-300)
    x0,y0=max(0,int(np.floor(x0))),max(0,int(np.floor(y0)));x1,y1=min(published.shape[1],int(np.ceil(x1))),min(published.shape[0],int(np.ceil(y1)))
    window=rasterio.windows.Window(x0,y0,x1-x0,y1-y0)
    originals={str(maskpath.relative_to(DATA)):file_sha256(maskpath)};images=[]
    for period in ('pre','post'):
        date={'pre':'2019-01-07','post':'2019-02-01'}[period]
        arrays=[]
        for band in ('B04','B03','B02'):
            paths=[p for p in RAW.glob('*.tif') if date in p.name and band in p.name]
            if len(paths)!=1:raise ValueError(f'Original {period}/{band} ambiguous: {paths}')
            path=paths[0]
            with rasterio.open(path) as f:
                if f.crs!=native_crs or f.transform!=grid or f.shape!=published.shape:raise ValueError('Original grids differ')
                arrays.append(f.read(1,window=window))
            originals[str(path.relative_to(DATA))]=file_sha256(path)
        rgb=np.stack(arrays,2).astype(np.float32)
        lo=np.percentile(rgb,2,axis=(0,1));hi=np.percentile(rgb,98,axis=(0,1))
        rgb=np.uint8(np.clip((rgb-lo)/np.maximum(hi-lo,1)*255,0,255))
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        for geom,colour in ((union,(255,80,40)),):
            polys=[geom] if geom.geom_type=='Polygon' else list(geom.geoms)
            for poly in polys:
                for ring in (poly.exterior,*poly.interiors):
                    xy=[inv*(coord[0],coord[1]) for coord in ring.coords]
                    draw.line([(xx-x0,yy-y0) for xx,yy in xy],fill=colour,width=2)
        # Research inventory contour is an independent publisher comparison, never model output.
        import cv2
        local=published[y0:y1,x0:x1].astype(np.uint8)
        contours,_=cv2.findContours(local,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points=[tuple(map(int,p)) for p in contour[:,0]]
            if len(points)>1:draw.line(points+[points[0]],fill=(0,220,255),width=1)
        dest=OUT/(period+'_government_vs_inventory.png');image.save(dest);images.append(str(dest))
    result={'reviewed_at':datetime.now().isoformat(timespec='seconds'),'event_id':'BR0001','role':'auxiliary_development',
            'scope':'tailings dam waste plume; not strict open-pit wall failure','source_manifest_sha256':file_sha256(SOURCE/'official_impact_manifest.json'),
            'official_metadata_url':source_manifest['official_layer']['metadataurl'],'official_polygon_count':len(geo),
            'original_official_crs':'EPSG:4674','target_original_crs':str(native_crs),'geometries_valid':True,
            'horizontal_boundary_comparison_only':True,'original_vector_Z_coordinates_not_used_as_elevation_evidence':True,
            'vector_area_m2':float(union.area),'clipped_native_grid_pixels':int(official.sum()),
            'published_inventory_pixels':int(published.sum()),'inventory_government_mask_IoU':inter/max(uni,1),
            'inventory_positive_overlap_fraction':inter/max(int(published.sum()),1),
            'government_positive_overlap_fraction':inter/max(int(official.sum()),1),'comparison_is_not_model_accuracy':True,
            'official_lineage':lineage,'metadata_citation_date':'2020-10-26','metadata_created':'2023-07-31',
            'exact_original_Pleiades_acquisition_date_not_given':True,'target_satellite_dates':{'pre':'2019-01-07','post':'2019-02-01'},
            'timing_limit':'official ADA product includes entire2019 plume; metadata citation/creation dates do not identify image date or prove Feb1 boundary',
            'independence_limit':'government publication independent of project labels; underlying independence from DLR inventory imagery and earlier IBGE vectors not established',
            'source_type':'official georeferenced remote-sensing interpretation; no independent field survey received',
            'model_inference_run':0,'accepted_for_training_or_scoring':False,'unknown_pixels_are_not_normal_operation_labels':True,
            'original_assets':originals,'overlay_legend':{'orange':'government ADA outline','cyan':'published DLR research mask'},
            'review_images':images,'derived_asset_sha256':{str(p.relative_to(OUT)):file_sha256(p) for p in [maskdest,*map(Path,images)]}}
    (OUT/'boundary_audit.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:result[k] for k in ('vector_area_m2','inventory_government_mask_IoU','inventory_positive_overlap_fraction','government_positive_overlap_fraction','review_images')}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
