"""Retrieve only published government-file candidates; missing/blocked is not obtained truth."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,hashlib,io,json,sys,urllib.request,zipfile
OUT=Path('D:/LMHLD/datasets/operational_boundary_sources_20261004/brumadinho/ibge_original_attempts')
PREFIX='https://www.ibge.gov.br/np_download/novoportal/extras/'
SOURCES={
 'published_package_zip':PREFIX+'Municipios_localidades_Afetados_MG.zip',
 'published_named_kml_root':PREFIX+'limites_rejeitos_dia29_v2.kml',
 'published_named_kml_folder':PREFIX+'Municipios_localidades_Afetados_MG/limites_rejeitos_dia29_v2.kml',
 'published_named_shape_zip':PREFIX+'limites_rejeitos_dia29_v2.zip'}


def fetch(item):
    key,url=item;row={'candidate':key,'url':url,'request_basis':'official notes list archive directory and target filename; exact working attachment path still to verify'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r:data=r.read(40*1024*1024+1)
        if len(data)>40*1024*1024:raise ValueError('Public file exceeds40MB retrieval budget')
        if data.startswith(b'PK'):
            archive=zipfile.ZipFile(io.BytesIO(data));row['archive_members']=archive.namelist();files=[]
            for member in archive.infolist():
                name=Path(member.filename).name
                if 'limites_rejeitos_dia29' not in name.lower():continue
                if member.file_size>8*1024*1024 or Path(name).suffix.lower() not in ('.kml','.shp','.shx','.dbf','.prj','.cpg'):raise ValueError('Unexpected target member')
                dest=OUT/(key+'_'+name);dest.write_bytes(archive.read(member));files.append({'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
            row.update(actual_target_files=files,target_boundary_obtained=any(Path(x['path']).suffix.lower() in ('.kml','.shp') for x in files))
        elif b'<kml' in data[:3000]:
            dest=OUT/(key+'.kml');dest.write_bytes(data);row.update(path=str(dest),sha256=hashlib.sha256(data).hexdigest(),target_boundary_obtained=True)
        else:raise ValueError('Response is neither archive nor actual KML')
    except Exception as exc:row.update(error=str(exc),target_boundary_obtained=False)
    return row


OUT.mkdir(exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(fetch,SOURCES.items()))
manifest={'checked_at':datetime.now().isoformat(timespec='seconds'),'official_notes':'../ibge_notes.pdf',
          'notes_image_dates':{'post_GeoEye1':'2019-01-29','pre_WorldView3':'2018-07-22'},
          'date_scope':'dates apply to the named IBGE target vector only, never transfer to SEMAD Pleiades vector',
          'rows':rows,'actual_dated_boundary_obtained':any(r['target_boundary_obtained'] for r in rows),'model_predictions_run':0}
(OUT/'attempts.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
