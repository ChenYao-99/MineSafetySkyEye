"""A transparent data-QA revision: actual working footprints and clear-land display."""
from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD')
sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
PARENT=ROOT/'validation/register_carmen_published_outline_20261005.py'
DEST=ROOT/'validation/register_carmen_published_outline_v2_20261005.py'
WORKINGS=[[[345,165],[417,150],[500,160],[533,250],[554,330],[538,437],[520,535],[406,570],[367,482],[300,410],[288,294],[314,244]],
          [[109,210],[197,203],[268,224],[323,266],[318,339],[271,358],[208,399],[154,373],[94,358],[83,293]],
          [[128,365],[230,347],[335,377],[374,417],[394,542],[346,580],[155,565],[111,470]],
          [[499,151],[584,136],[704,119],[743,161],[750,282],[720,365],[651,406],[551,355],[524,290]]]
def run():
    source=PARENT.read_text(encoding='utf-8')
    helper='''WORKINGS=np.array([]) # replaced below with source-only digitised working polygons
def observed_display(path,valid):
    with rasterio.open(path) as f:x=f.read()[:3].transpose(1,2,0).astype(np.float32)
    for band in range(3):
        low,high=np.percentile(x[:,:,band][valid],[2,98]);x[:,:,band]=np.clip((x[:,:,band]-float(low))/max(float(high-low),1e-6),0,1)
    x[~valid]=0
    return (x[:,:,::-1]*255).astype(np.uint8)
'''.replace('WORKINGS=np.array([]) # replaced below with source-only digitised working polygons','WORKINGS='+repr(WORKINGS))
    replacements={"reference_registration_v1_wang2026":"reference_registration_v2_actual_workings_clear_land",
        'def write(p,v):':helper+'\ndef write(p,v):',
        "'source_mask':'maprect31:837,31:590;all main workingsx110:790,y155:580 excluded;white annotations excluded'":
        "'source_mask':'maprect;four visible working footprints+55m buffer excluded;intervening unchanged vegetation remains;white annotations excluded','source_working_polygons':WORKINGS,'display_normalisation':'target observed land 2/98,not cloud-contaminated fullimage statistics','previous_v1_failure_preserved':True",
        "'target_mask':'same printed-prior working rectangle excluded;strict observed land only'":
        "'target_mask':'same printed-prior four working footprints+50m buffer excluded;strict observed land only'",
        'mask[155:580,110:790]=False;mask[source.min(2)>230]=False':
        "working=np.zeros(mask.shape,np.uint8);cv2.fillPoly(working,[np.array(poly,np.int32) for poly in WORKINGS],1);mask &= ~cv2.dilate(working,np.ones((11,11),np.uint8)).astype(bool);mask[source.min(2)>230]=False",
        "with rasterio.open(CASE/(period+'_SCL_10m.tif')) as f:valid &= ~np.isin(f.read(1),p['SCL_excluded'])":
        "with rasterio.open(CASE/(period+'_SCL_10m.tif')) as f:valid &= ~np.isin(f.read(1),p['SCL_excluded'])\n        target=observed_display(CASE/(period+'_BGRN_10m.tif'),valid)",
        "workings=prior(np.array([[110,155],[790,155],[790,580],[110,580]]),frame)\n            tmask=valid.astype(np.uint8);cv2.fillPoly(tmask,[np.rint(workings).astype(np.int32)],0)":
        "workings=[prior(np.array(poly),frame) for poly in WORKINGS]\n            excluded=np.zeros(valid.shape,np.uint8);cv2.fillPoly(excluded,[np.rint(poly).astype(np.int32) for poly in workings],1)\n            tmask=(valid & ~cv2.dilate(excluded,np.ones((11,11),np.uint8)).astype(bool)).astype(np.uint8)"}
    for before,after in replacements.items():
        if before not in source:raise ValueError('Acquisition/registration parent changed')
        source=source.replace(before,after)
    if DEST.exists() and DEST.read_text(encoding='utf-8')!=source:raise ValueError('Frozen second registration code changed')
    if not DEST.exists():DEST.write_text(source,encoding='utf-8')
    provenance={'parent':str(PARENT),'parent_sha256':file_sha256(PARENT),'derived':str(DEST),'derived_sha256':file_sha256(DEST),
                'reason':'Original rectangular working mask removed unrelated forest;fullimage2/98 contrast was dominated by clouds.New source-only footprints and strict observed-land display stats',
                'all_numeric_acceptance_gates_unchanged':True,'raw_pixels_labels_detector_or_blind_test_outputs_used_to_change_masks':False,
                'source_working_polygons':WORKINGS,'existing_failed_registration_retained':True}
    (ROOT/'datasets/carmen_development_20261005/registration_v2_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':run()
