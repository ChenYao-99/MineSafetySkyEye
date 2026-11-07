"""Freeze source-only digitisation and a held-ground registration protocol for the new mine."""
from pathlib import Path
import json,hashlib
ROOT=Path('D:/LMHLD');BASE=ROOT/'validation/register_gamsberg_published_boundary_v5.py'
DEST=ROOT/'validation/register_copler_field_boundary_v1.py'
text=BASE.read_text(encoding='utf-8')
old="from register_gamsberg_published_boundary_v2 import SOURCE,CASE,POLYGON,write\nfrom rasterio.warp import transform as coordinates\nAXES={'longitude':{'source_x':[224.,454.,684.],'degrees':[18.95,18.975,19.]},\n      'latitude':{'source_y':[198.,400.],'degrees':[-(29+14/60+17/3600),-(29+15/60+36/3600)]}}"
new="""from register_gamsberg_published_boundary_v2 import write
from rasterio.warp import transform as coordinates
from mine_data_protocol import registry,assert_mine_group_role
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/copler_author_figure1.png'
CASE=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'
AXES={'longitude':{'source_x':[314.5,607.,899.5],'degrees':[38.5,38+32/60,38+34/60]},
      'latitude':{'source_y':[70.5,449.],'degrees':[39+28/60,39+26/60]}}
POLYGON=np.array([[719,414],[727,419],[732,432],[737,445],[744,448],[744,455],
 [752,460],[764,463],[771,470],[774,484],[771,496],[773,513],[778,523],
 [776,528],[761,534],[752,544],[750,559],[746,571],[744,585],[737,602],
 [729,609],[720,613],[706,615],[697,614],[690,609],[686,602],[686,589],
 [691,578],[694,568],[681,569],[672,567],[661,566],[655,556],[647,550],
 [644,539],[638,533],[638,526],[647,532],[659,535],[665,531],[667,523],
 [677,518],[686,516],[694,519],[703,522],[713,529],[721,530],[726,523],
 [731,510],[734,507],[743,505],[747,501],[748,491],[745,478],[738,467],
 [729,460],[724,449],[721,433]],np.float64)
"""
if text.count(old)!=1:raise ValueError('Unexpected source constants')
text=text.replace(old,new)
replacements={
 "reference_registration_v5_wang2026_verified_axes":"reference_registration_v1_field2025",
 "https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-mmc1.docx":"https://gfzpublic.gfz.de/rest/items/item_5035234_2/component/file_5035235/content",
 "Wang etal2026 FigureS13; original author outline, source event explicitly17November2020 inS14":"Buyukakpinar etal2025 Figure1 white2024failure outline; p5 documents14Feb2024 two-day field campaign and source optical dates",
 "'source_google_earth_capture_date':None,'event_date':'2020-11-17'":"'source_google_earth_capture_date':'2021-11-19','event_date':'2024-02-13'",
 "'earlier_failed_v1_and_v2_preserved':True":"'prior_Gams_registration_failures_not_reused_as_Copler_truth':True",
 "'ground_only_mask':'source workings/accident/annotation excluded; target workings[y170:455,x150:450] and Southpit[y435:550,x248:338] excluded; western unchanged terrain retained'":"'ground_only_mask':'source inset/river/mine workings/failure/text excluded; target weather exclusion plus source-geographic mine rectangles with100m buffer; terrain only'",
 "    OUT.mkdir(exist_ok=True)":"    value=registry();assert_mine_group_role('TR_COP2024','reserved_final_test',value)\n    if value['cases']['TR_COP2024']['model_predictions_run']!=0:raise ValueError('Mine detector already observed this candidate')\n    OUT.mkdir(exist_ok=True)",
 "    ok=np.zeros(source.shape[:2],bool);ok[45:533,45:821]=True\n    ok[50:347,105:421]=False;ok[:110,:85]=False;ok[370:,:180]=False;ok[485:,680:]=False":"    ok=np.zeros(source.shape[:2],bool);ok[55:696,60:1010]=True\n    ok[50:285,45:510]=False;ok[50:365,430:810]=False;ok[265:345,50:660]=False\n    for x0,y0,x1,y1 in [(330,325,655,420),(385,407,655,535),(425,525,708,672),(625,395,795,632),(795,395,925,545),(120,310,420,365),(165,405,390,455),(45,490,355,559),(725,260,1010,310),(630,135,875,185),(735,600,1000,696)]:ok[y0:y1,x0:x1]=False\n    ok[620:696,60:310]=False",
 "        target,valid=display(CASE/f'{period}_BGRN_10m.tif');tmask=valid.copy();tmask[170:455,150:450]=False; tmask[435:550,248:338]=False":"        target,valid=display(CASE/f'{period}_BGRN_10m.tif')\n        with rasterio.open(CASE/f'{period}_SCL_10m.tif') as q:weather=~np.isin(q.read(1),[0,1,3,8,9,10,11])\n        tmask=valid & weather\n        with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as frame:\n            for x0,y0,x1,y1 in [(330,325,655,420),(385,407,655,535),(425,525,708,672),(625,395,795,632),(795,395,925,545)]:\n                corners=printed_prior(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float),frame)\n                excluded=np.zeros(valid.shape,np.uint8);cv2.fillPoly(excluded,[np.rint(corners).astype(np.int32)],1)\n                excluded=cv2.dilate(excluded,np.ones((21,21),np.uint8));tmask[excluded>0]=False"
}
for old,new in replacements.items():
 if text.count(old)!=1:raise ValueError('Unexpected reused structure:'+old)
 text=text.replace(old,new)
if DEST.exists() and DEST.read_text(encoding='utf-8')!=text:raise ValueError('Candidate registration code changed')
DEST.write_text(text,encoding='utf-8')
OUT=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'
(OUT/'source_digitisation_provenance.json').write_text(json.dumps({'case_id':'TR_COP2024','role':'reserved_final_test','digitisation_basis':'original2025author Figure1 white stroke centreline; target image/model outputs not used to trace boundary','source_image_sha256':hashlib.sha256((ROOT/'datasets/pixel_boundary_sources_20261004/copler_author_figure1.png').read_bytes()).hexdigest(),'source_figure_base_date':'2021-11-19; not a postevent picture','original_failure_annotation_event_date':'2024-02-13','source_field_campaign':'14Feb2024, two days, described on paperp5;notprojecttrial','code_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'same_60_unique_30_fit_30_heldout_geometry_gate':True,'detector_predictions':0},indent=2)+'\n',encoding='utf-8')
print('New-mine field-informed boundary digitisation frozen; no detector outputs or tuning')
