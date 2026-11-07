"""Independent local descriptor matcher; same source mask and prefit holdout gates."""
from pathlib import Path
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
def run():
    parent=ROOT/'validation/register_kef_author_outline_20261005.py';text=parent.read_text(encoding='utf-8')
    text=text.replace('reference_registration_v1_wang2026','reference_registration_v4_mutual_sift')
    text=text.replace("'matcher':'fixedLoFTR outdoor;not mine detector'","'matcher':'SIFT localdescriptors;mutualnearest ratio.8;not mine detector','LoFTR_prior_failures_preserved':True")
    start=text.index('    torch.set_num_threads(4);model=LoFTR');end=text.index("    for period in ('pre','post'):",start)
    text=text[:start]+"    sift=cv2.SIFT_create(nfeatures=30000,contrastThreshold=.008);clahe=cv2.createCLAHE(2,(8,8));source_gray=clahe.apply(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY));source_keypoints,source_descriptors=sift.detectAndCompute(source_gray,mask*255);rows=[]\n"+text[end:]
    start=text.index('tmask[cv2.dilate(excluded');end=text.index('        with rasterio.open(path) as frame:expected=prior(aa,frame)',start)
    replacement="""tmask[cv2.dilate(excluded,np.ones((11,11),np.uint8))>0]=0
        target_gray=clahe.apply(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY));target_keypoints,target_descriptors=sift.detectAndCompute(target_gray,tmask*255)
        pairs=[]
        if source_descriptors is not None and target_descriptors is not None:
            matcher=cv2.BFMatcher();forward=matcher.knnMatch(source_descriptors,target_descriptors,k=2);reverse=matcher.knnMatch(target_descriptors,source_descriptors,k=2)
            backward={a.queryIdx:a.trainIdx for match in reverse if len(match)==2 for a,b in [match] if a.distance<.8*b.distance}
            pairs=[(a,b) for match in forward if len(match)==2 for a,b in [match] if a.distance<.8*b.distance and backward.get(a.trainIdx)==a.queryIdx]
        aa=np.array([source_keypoints[a.queryIdx].pt for a,b in pairs],np.float32).reshape(-1,2);bb=np.array([target_keypoints[a.trainIdx].pt for a,b in pairs],np.float32).reshape(-1,2)
        cc=np.array([1-a.distance/max(b.distance,1e-6) for a,b in pairs],np.float32)
"""
    text=text[:start]+replacement+text[end:]
    text=text.replace('if cc[i]<.5 or not(','if cc[i]<.2 or not(').replace("'matcher_confidence_min':.5","'descriptor_quality_min':.2,'mutual_ratio_max':.8")
    text=text.replace('        del output,y,n;torch.cuda.empty_cache()','        del target_gray,target_descriptors')
    dst=ROOT/'validation/register_kef_author_outline_v4.py'
    if dst.exists() and dst.read_text(encoding='utf-8')!=text:raise ValueError('Frozen independent matching changed')
    if not dst.exists():dst.write_text(text,encoding='utf-8')
    (ROOT/'datasets/kef_essnoun_development_20261005/sift_registration_v4_provenance.json').write_text(json.dumps({
       'parent':str(parent),'parent_sha256':file_sha256(parent),'derived':str(dst),'derived_sha256':file_sha256(dst),
       'change':'independent localdescriptor matcher;mutualratio.8;originalimages masks geography and90unique30prefitheldgates unchanged',
       'detector_predictions_or_label_answers_entered':False},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':run()
