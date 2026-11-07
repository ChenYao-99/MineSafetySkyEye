"""Archive actual government impact polygons; acceptance requires time/lineage review."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,hashlib,json,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/operational_boundary_sources_20261004/brumadinho')
rows=json.loads((OUT/'sisema_relevant_layers.json').read_text(encoding='utf-8'))['layers']
layer=next(r for r in rows if r['id']==1187)
uid=layer['metadataurl'].rsplit('/',1)[1]
sources={'official_impact_polygons.geojson':next(d['url'] for d in layer['downloads'] if d['type']=='GeoJSON'),
         'official_impact_metadata.xml':'https://idesisema.meioambiente.mg.gov.br/geonetwork/srv/api/records/'+uid+'/formatters/xml',
         'official_impact_metadata_legacy.xml':'https://idesisema.meioambiente.mg.gov.br/geonetwork/srv/por/xml.metadata.get?uuid='+uid}


def read(item):
    name,url=item;row={'url':url,'filename':name}
    try:
        with urllib.request.urlopen(url,timeout=30) as r:data=r.read(12*1024*1024+1)
        if len(data)>12*1024*1024:raise ValueError('Boundary response exceeds12MB')
        path=OUT/name;path.write_bytes(data);row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path=str(path))
        if name.endswith('geojson'):
            value=json.loads(data)
            if value.get('type')!='FeatureCollection' or not value.get('features'):raise ValueError('No actual impact polygon data')
            row.update(feature_count=len(value['features']),crs=value.get('crs'),properties=[f.get('properties') for f in value['features']][:8])
        else:
            import xml.etree.ElementTree as ET
            root=ET.fromstring(data);values=[]
            for node in root.iter():
                t=(node.text or '').strip()
                if len(t)>20 or any(w in node.tag.lower() for w in ('date','time','scale')):
                    if t:values.append(t)
            row['metadata_text']=values
    except Exception as exc:row['error']=str(exc)
    return row


with concurrent.futures.ThreadPoolExecutor(3) as pool:result=list(pool.map(read,sources.items()))
manifest={'downloaded_at':datetime.now().isoformat(timespec='seconds'),'official_layer':layer,
          'role':'BR0001 auxiliary development; no new blind event','model_inference_run':0,
          'acceptance':'pending temporal/lineage/geometry review; government hosting is not proof of independent survey',
          'sources':result}
(OUT/'official_impact_manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
