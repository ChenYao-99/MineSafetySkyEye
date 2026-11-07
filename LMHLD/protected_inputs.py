"""Keep registered reserved mine inputs out of routine upload inference."""
import json
import math
from pathlib import Path

DATA=Path(__file__).resolve().parent/'datasets/mine_benchmark_20261002'


def geographic_reserved_overlap(meta,cases):
    """Protect reencoded/cropped GeoTIFFs even when event tags and byte hashes change."""
    geo=meta.get('georeference')
    if not geo:return None
    from affine import Affine
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.errors import RasterioError
    try:
        width,height=int(geo['width']),int(geo['height'])
        if min(width,height)<=0:raise ValueError('Invalid raster grid')
        matrix=Affine(*geo['transform'])
        native=array_bounds(height,width,matrix)
        box=transform_bounds(geo['crs'],'EPSG:4326',*native,densify_pts=21)
        if not all(math.isfinite(v) for v in box):raise ValueError('Invalid raster footprint')
    except (KeyError,TypeError,ValueError,RasterioError):
        return None
    overlaps=[]
    for case_id,case in cases.items():
        if case.get('role')!='reserved_final_test':continue
        protected=case.get('protected_geographic_extent_wgs84')
        if not protected:continue
        if min(box[2],protected[2])>max(box[0],protected[0]) and min(box[3],protected[3])>max(box[1],protected[1]):
            overlaps.append(case_id)
    if not overlaps:return None
    return {'reserved':True,'case_id':overlaps[0],'case_ids':overlaps,
            'verified_by':'registered_reserved_geographic_overlap','identity_verified':False,
            'scope':'conservative source-window isolation, not verified accident footprint'}


def input_split(meta):
    auxiliary=DATA.parent/'input_reserve_registry.json'
    if auxiliary.exists():
        value=json.loads(auxiliary.read_text(encoding='utf-8'))
        if value.get('schema_version')!=1 or not isinstance(value.get('cases'),dict):
            raise ValueError('Invalid auxiliary reserve registry;routine inference refused')
        for case_id,case in value['cases'].items():
            if meta.get('input_sha256') in case['input_sha256_values'] or meta.get('declared_case_id')==case_id:
                return {'reserved':True,'case_id':case_id,'verified_by':'registered_auxiliary_input_identity',
                        'scope':case['scope'],'identity_verified':meta.get('input_sha256') in case['input_sha256_values']}
        spatial=geographic_reserved_overlap(meta,value['cases'])
        if spatial:
            spatial['scope']='conservative auxiliary mining-change holdout;not an accident benchmark'
            return spatial
    path=DATA/'mine_case_registry.json'
    declared=meta.get('declared_case_id')
    if not path.exists():
        return {'reserved':str(meta.get('declared_data_role','')).startswith('reserved'),'case_id':declared}
    value=json.loads(path.read_text(encoding='utf-8'));cases=value['cases']
    digest=meta.get('input_sha256')
    assets={**value['frozen_assets'],**value.get('additional_frozen_assets',{})}
    for name,expected in assets.items():
        if digest!=expected:continue
        for case_id,case in cases.items():
            acquisition=case.get('acquisition')
            folder=Path(acquisition).parent.as_posix()+'/' if acquisition else None
            if case_id in name or (folder and folder!='./' and name.startswith(folder)):
                return {'reserved':case['role']=='reserved_final_test','case_id':case_id,'verified_by':'frozen_raw_sha256'}
    spatial=geographic_reserved_overlap(meta,cases)
    if spatial:return spatial
    if declared in cases:
        # Registry role wins over immutable historical Xinjing tags.
        flagged=str(meta.get('declared_data_role','')).startswith('reserved')
        return {'reserved':cases[declared]['role']=='reserved_final_test' or flagged,
                'case_id':declared,'verified_by':'registered_declared_case','identity_verified':False}
    return {'reserved':str(meta.get('declared_data_role','')).startswith('reserved'),'case_id':declared,'verified_by':'header_declaration_only'}
