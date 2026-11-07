"""Acquire different arid mining contexts after the blind audit; not accident labels."""
from datetime import datetime
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests
import acquire_minenet_data_conditions_20261007 as source
DATA=ROOT/'datasets/post_audit_development_20261008'
OLD=ROOT/'datasets/data_conditions_20261007'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    DATA.mkdir(exist_ok=True)
    catalogue=OLD/'MineNetCD/MCD/summary20240127.csv'
    rows={row['Task name']:row for row in csv.DictReader(catalogue.open(encoding='utf-8-sig'))}
    directory=json.loads((OLD/'MineNetCD_zip_directory.json').read_text(encoding='utf-8'))
    protocol_path=DATA/'protocol.json'
    selected=['Chile1','Chile2']
    metadata={site:{key:rows[site][key] for key in ['Task name','T1','T2','left','bottom','right','top']} for site in selected}
    protocol={'registered_at':datetime.now().astimezone().isoformat(),'source_DOI':'10.14278/rodare.3251',
        'reason':'blind audit showed poor separation of failure footprint from dry terraced/bare mining contexts',
        'site_ids':selected,'roles':{site:'development_auxiliary_change_context_only' for site in selected},
        'metadata':metadata,'author_catalogue_sha256':sha(catalogue),'source_directory_sha256':sha(OLD/'MineNetCD_zip_directory.json'),
        'transfer_budget_bytes':150*1024**2,'source_url':directory['source'],
        'accident_or_safe_negative_labels_admitted':False,'holdout_mine_or_neighbouring_images_used':False,
        'native_pair_labels_may_be_transferred_from_longer_RGB_period':False,'new_model_predictions_or_fits':0,
        'operator_sources':{'Chile1':{'possible_identity':'Spence;identity still needs independent spatial confirmation',
            'URL':'https://www.bhp.com/news/media-centre/releases/2017/08/bhp-approves-investment-to-extend-life-of-spence-copper-mine'},
            'Chile2':{'possible_identity':'Mantoverde;identity still needs independent spatial confirmation',
            'URL':'https://capstonecopper.com/news/capstone-copper-reports-fourth-quarter-2023-results/'}}}
    if protocol_path.exists():protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    else:write(protocol_path,protocol)
    assert protocol['site_ids']==selected and protocol['new_model_predictions_or_fits']==0
    protected=json.loads((ROOT/'datasets/input_reserve_registry.json').read_text(encoding='utf-8'))['cases']
    primary=json.loads((ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json').read_text(encoding='utf-8'))['cases']
    for site,meta in metadata.items():
        box=[float(meta[key]) for key in ['left','bottom','right','top']]
        for entry in list(protected.values())+list(primary.values()):
            other=entry.get('protected_geographic_extent_wgs84')
            assert not other or not (min(box[2],other[2])>max(box[0],other[0]) and min(box[3],other[3])>max(box[1],other[1]))
    source.ROOT=DATA;used=0;files=[]
    with requests.Session() as session:
        for site in selected:
            entries=[entry for entry in directory['entries'] if entry['name'].startswith('MCD/'+site+'/') and entry['uncompressed']]
            assert len(entries)==3
            for entry in entries:
                path,digest,count=source.acquire(entry,directory,session,protocol['transfer_budget_bytes']-used)
                used+=count;files.append({'path':str(path),'sha256':digest,'CRC32':entry['crc32'],'bytes':entry['uncompressed']})
                print('VERIFIED different-mine source',entry['name'],flush=True)
        snapshots=[]
        for site,info in protocol['operator_sources'].items():
            path=DATA/(site+'_operator_original.html')
            if not path.exists():
                try:
                    response=session.get(info['URL'],timeout=(10,25));response.raise_for_status()
                    assert len(response.content)<4*1024**2;path.write_bytes(response.content)
                except requests.RequestException as error:
                    snapshots.append({'site':site,'URL':info['URL'],'original_page_downloaded':False,
                        'error_type':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None),
                        'web_primary_source_accessed_separately':True,'pixelwise_normal_work_or_safety_claim':False})
                    continue
            snapshots.append({'site':site,'URL':info['URL'],'path':str(path),'sha256':sha(path),
                'original_page_downloaded':True,'claim':'operator confirms site activity;not pixelwise proof of safe normal mining'})
    write(DATA/'acquisition.json',{'passed':True,'new_author_RGB_pairs':2,'new_original_change_masks':2,
        'bytes_transferred_this_resume':used,'files':files,'operator_sources':snapshots,
        'first_attempt_transfer_count_not_persisted':True,'new_accident_or_verified_safe_labels':0,'classifier_predictions_or_fits':0})
    cases={}
    for site,pre,post in [('Chile1','2020-02-21','2021-02-21'),('Chile2','2022-04-06','2023-04-06')]:
        meta=metadata[site]
        cases['AUX_ARID_'+site.upper()]={'role':'development_auxiliary_change_context_only','source_event_ids':[site],
            'bbox_wgs84':[float(meta[key]) for key in ['left','bottom','right','top']],
            'query_interval':('2020-02-07T00:00:00Z/2021-03-08T23:59:59Z' if site=='Chile1' else '2022-03-23T00:00:00Z/2023-04-21T23:59:59Z'),
            'pre_through':pre,'post_from':('2021-02-07' if site=='Chile1' else '2023-03-23'),'post_target':post}
    plan={'cases':cases,'max_pixels_per_side':1536,
        'selection_rule':'latest clear full-cover scene through frozen predate;post nearest frozen target date then scene ID;no model use',
        'normal_operation_cause_or_safe_background_not_certified':True,'author_RGB_change_mask_not_transferred_to_different_native_dates':True}
    native=DATA/'native_pair_plan.json'
    if native.exists():assert json.loads(native.read_text(encoding='utf-8'))==plan
    else:write(native,plan)
    print('Acquired2 different author mining-change groups;native-year context plan frozen;accident/safety labels0',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
