"""Verified mining-area archive, metadata only at first; never treat mine masks as failures."""
from pathlib import Path
from datetime import datetime
import json,sys,hashlib,zipfile
import requests
ROOT=Path(__file__).resolve().parents[1]
META=ROOT/'datasets/mine_benchmark_20261002/boundary_catalogue_leads_20261004/zenodo_14195737.json'
OUT=ROOT/'datasets/mine_geometry_auxiliary_20261004'


def run():
    metadata=json.loads(META.read_text(encoding='utf-8'));file=next(f for f in metadata['files'] if f['key']=='mining_area_data.zip')
    OUT.mkdir(exist_ok=True);target=OUT/'mining_area_data.zip';partial=OUT/'mining_area_data.zip.part'
    if not target.exists():
        digest=hashlib.md5();size=0
        with requests.get(file['links']['self'],stream=True,timeout=(10,40)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk);digest.update(chunk);size+=len(chunk)
                    if size>int(file['size']):raise ValueError('Archive exceeded publisher size')
        if size!=file['size'] or 'md5:'+digest.hexdigest()!=file['checksum']:raise ValueError('Publisher checksum/size mismatch')
        if not target.resolve().is_relative_to(OUT.resolve()):raise ValueError('Archive target escaped dataset root')
        partial.replace(target)
    if 'md5:'+hashlib.md5(target.read_bytes()).hexdigest()!=file['checksum']:raise ValueError('Saved archive checksum mismatch')
    inventory=[];extracted=[]
    with zipfile.ZipFile(target) as z:
        for item in z.infolist():
            inventory.append({'name':item.filename,'bytes':item.file_size,'compressed':item.compress_size})
            if item.is_dir() or not item.filename.lower().endswith(('.csv','.json','.md','.txt','.geojson')) or item.file_size>40*1024*1024:continue
            path=(OUT/'metadata'/item.filename).resolve()
            if not path.is_relative_to((OUT/'metadata').resolve()):raise ValueError('Archive metadata path escaped root')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(item))
            extracted.append(str(path.relative_to(OUT)))
    record={'checked_at':datetime.now().isoformat(timespec='seconds'),'source':'https://zenodo.org/records/14195737',
            'archive_bytes':target.stat().st_size,'publisher_checksum':file['checksum'],'publisher_checksum_verified':True,
            'inventory':inventory,'metadata_extracted':extracted,'label_role':'auxiliary mining-area geometry only',
            'accident_or_stability_labels':False,'satellite_original_images_downloaded':0,'model_training_started':False,
            'reserved_mine_exclusion_required_before_any_training':True}
    (OUT/'acquisition.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'archive_bytes':record['archive_bytes'],'entries':len(inventory),'metadata_files':len(extracted),'first_entries':inventory[:16]}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
