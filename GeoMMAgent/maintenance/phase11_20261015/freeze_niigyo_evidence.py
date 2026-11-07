"""Freeze source grades and make the remaining qualification task explicit."""
import sys
from screen_noto import DATA, HERE, now, read, write, sha
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from rasterio.warp import transform_geom
from rasterio.features import rasterize
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from overlay_source_references import rings


def main():
    path = HERE / 'source_grade_contract.json'
    contract = {'event_id':'JP_NOTO20240101', 'site_id':'JP_NOTO_NIIGYO_QUARRY_SOURCE_CANDIDATE',
          'same_site_same_event_across_dates':True,
          'mine_identity_grade':'visually verified benched quarry; locality supported by geological report; legal permit footprint not acquired',
          'reference_grade':'official optical and DEM interpretation, no field-certified complete accident footprint',
          'current_role':'qualification_only', 'training_allowed':False,'model_predictions_allowed':False,
          'current_batch_model_outputs_not_viewed':True,
          'historical_pretraining_exposure_audit_complete':False,
          'eligible_as_new_blind_acceptance_set':False,
          'positive_pixel_scope':'not admitted; source agreements are geometric support only',
          'negative_pixel_scope':'none; unannotated areas and pre-event bare benches are not safe negatives',
          'reference_priority_thresholds_unchanged':read(HERE/'protocol.json')['source_priority_rules'],
          'components_count_as_independent_mines':False,
          'april_role':'locating/context only, not January truth',
          'cloud_and_snow_policy':'alpha validity is not cloud/snow quality; January 11 has extensive snow',
          'generic_geocoder_datum_hypothesis_correction':'documented locality geocoder is a WGS84 search hint; old Tokyo-datum alternative is irrelevant and ignored; original protocols retained',
          'required_next_decision':'freeze supervised RGB target ontology and source uncertainty margin before any split or experiment'}
    if path.exists():
        if read(path) != contract:
            raise ValueError('Frozen source grade changed')
    else:
        write(path,contract)
    features = read(DATA / 'qualification/niigyo_source_candidates.geojson')['features']
    selected = [r for r in features if r['properties']['family']=='optical']
    # Search relation remains a relation, not a new union/complete accident mask.
    rows = []
    dem = [(r,shape(transform_geom('EPSG:4326','EPSG:6675',r['geometry']))) for r in features if r['properties']['family']=='dem']
    for row in selected:
        original = shape(transform_geom('EPSG:4326','EPSG:6675',row['geometry']))
        counterparts = [(r,g) for r,g in dem if g.intersects(original)]
        agreement = original.intersection(unary_union([g for _,g in counterparts]))
        core = agreement.buffer(-10)
        rows.append({'source_feature_index':row['properties']['feature_index'],
              'official_optical_area_m2':original.area,
              'overlapping_dem_fids':[r['properties']['original_fid'] for r,_ in counterparts],
              'optical_dem_agreement_area_m2':agreement.area,
              'agreement_eroded_10m_area_m2':core.area,
              'official_optical_20m_core_area_m2':original.buffer(-20).area,
              'large_source_priority':original.area>=10000 and original.buffer(-20).area>=2500,
              'training_label_admitted':False})
    items = [
        ('FY2020 (2020-04 to 2021-03)',DATA/'imagery_overviews/NIIGYO_PRE_FY2020/released_aerial_RGB_3857.tif',False),
        ('2023-12-06 Sentinel-2 (10 m)',DATA/'recent_pre_satellite/red_provider_native.tif',True),
        ('2024-01-05 aerial (snow/haze)',DATA/'imagery_overviews/NIIGYO_R2_JAN05/released_aerial_RGB_3857.tif',False),
        ('2024-01-11 aerial (extensive snow)',DATA/'imagery_overviews/NIIGYO_R2_JAN11/released_aerial_RGB_3857.tif',False),
        ('2024-04-05 to 26 (context only)',DATA/'imagery_overviews/NIIGYO_R2_APR_CONTEXT/released_aerial_RGB_3857.tif',False)]
    with rasterio.open(items[0][1]) as src:
        a,b = src.transform * (200,1080)
        c,d = src.transform * (820,400)
    window = box(a,b,c,d)
    fig,axes = plt.subplots(1,len(items),figsize=(18,7),constrained_layout=True)
    for axis,(label,image_path,satellite) in zip(axes,items):
        with rasterio.open(image_path) as src:
            bounds,crs = src.bounds,src.crs
            if satellite:
                data = []
                for name in ['red','green','blue']:
                    with rasterio.open(image_path.parent/(name+'_provider_native.tif')) as band:
                        data.append(band.read(1))
                rgb = np.stack(data,axis=2).astype(float)
                valid = np.all(rgb>0,axis=2)
                lo,hi = np.percentile(rgb[valid],[2,98])
                rgb = np.clip((rgb-lo)/max(hi-lo,1),0,1)
            else:
                rgb = src.read().transpose(1,2,0)
        view = shape(transform_geom('EPSG:3857',crs.to_string(),mapping(window)))
        xmin,ymin,xmax,ymax = view.bounds
        axis.imshow(rgb,extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],interpolation='nearest')
        for feature in selected:
            geom = shape(transform_geom('EPSG:4326',crs.to_string(),feature['geometry']))
            for ring in rings(geom):
                axis.plot(*ring.xy,color='yellow',linewidth=.8)
        axis.set_xlim(xmin,xmax)
        axis.set_ylim(ymin,ymax)
        axis.set_title(label,fontsize=10)
        axis.set_axis_off()
    fig.suptitle('Same quarry, original optical reference outlines | source QA only, no model output\nDates remain distinct; no complete accident mask or verified-safe background asserted',fontsize=13)
    out = DATA/'qualification/niigyo_temporal_source_review.png'
    fig.savefig(out,dpi=180)
    plt.close(fig)
    result = {'completed_at':now(),'contract':str(path),'contract_sha256':sha(path),
              'temporal_source_review':str(out),'source_object_support':rows,
              'new_visually_verified_quarry_sites':1,'new_independent_events':1,
              'official_optical_objects_in_quarry_search_window':len(selected),
              'large_source_priority_objects':sum(r['large_source_priority'] for r in rows),
              'new_complete_field_certified_mine_accident_boundaries':0,
              'new_admitted_supervised_positive_pixels':0,'new_verified_negative_pixels':0,
              'new_predictions':0,'new_fits':0,'new_external_generation_calls':0,
              'decision':'retain one quarry-associated source candidate with before/after context; do not force small, snow-covered or vegetated components into exposed-earth training labels',
              'manual_review_notes':[
                  '2020 and December 2023 show an established benched quarry; quarry identity is visually supported.',
                  'NE quarry-associated official optical objects 206/207 have DEM deep-landslide body/scarp counterparts; complete quarry legal extent and field event certification unavailable.',
                  'Optical 204 lies on an excavation bench and lacks matching DEM reference; do not promote as reliable positive.',
                  'Optical 205 is quarry-peripheral slope failure, not evidence of failure across the complete working pit.',
                  'January 11 has snow; January 5 body includes dark/vegetated surfaces. April exposed soil cannot be backdated to January.',
                  'December 6 satellite pixels resolve coarse land cover only; they cannot certify 1 m boundary placement or exclude an intervening December change.'
              ]}
    write(DATA/'qualification/niigyo_evidence_freeze.json',result)
    print(__import__('json').dumps({k:result[k] for k in ['new_visually_verified_quarry_sites','official_optical_objects_in_quarry_search_window','large_source_priority_objects','new_admitted_supervised_positive_pixels','new_fits']}),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
