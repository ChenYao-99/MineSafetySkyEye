"""Read-only discovery of published Brazilian government boundary sources."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,hashlib,json,re,sys,urllib.request,urllib.error
OUT=Path('D:/LMHLD/datasets/operational_boundary_sources_20261004/brumadinho')
PREFIX='https://www.ibge.gov.br/np_download/novoportal/extras/'
SOURCES={
 'ibge_published_kml':PREFIX+'Municipios_localidades_Afetados_MG/limites_rejeitos_dia29_v2.kml',
 'ibge_published_directory':PREFIX+'Municipios_localidades_Afetados_MG/',
 'ibge_notes':PREFIX+'Informacoes_complementares_Brumadinho_18fev2019.pdf',
 'ibge_capabilities':'https://geoservicos.ibge.gov.br/geoserver/wfs?service=WFS&request=GetCapabilities',
 'sisema_viewer':'https://visualizador.idesisema.meioambiente.mg.gov.br/'}


def read_source(item):
    key,url=item;row={'key':key,'url':url,'checked_at':datetime.now().isoformat(timespec='seconds')}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=25) as response:
            data=response.read(15*1024*1024+1)
            if len(data)>15*1024*1024:raise ValueError('Discovery response exceeds budget')
            row.update(status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'))
        path=OUT/(key+('.pdf' if data.startswith(b'%PDF') else '.raw'))
        path.write_bytes(data);row.update(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        text=data.decode('utf-8',errors='replace')
        if key=='sisema_viewer':
            row['script_sources']=re.findall(r'<script[^>]+src=["\']([^"\']+)',text)
        if key=='ibge_capabilities':
            row['relevant_names']=re.findall(r'<[^>]*Name>([^<]*(?:[Bb]rumadinho|rejeito|[Ff]eijao)[^<]*)</',text)
    except Exception as exc:row['error']=str(exc)
    return row


def run():
    OUT.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(read_source,SOURCES.items()))
    result={'checked_at':datetime.now().isoformat(timespec='seconds'),'event_id':'BR0001','role':'auxiliary development',
            'scope':'tailings dam mudflow; not pit-wall collapse','new_model_predictions':0,'sources':rows}
    (OUT/'discovery.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
