# -*- coding: utf-8 -*-
"""Retrieve actual productXML calibration, preserve frozen originals, save derivatives."""
from pathlib import Path
import json,hashlib,sys,urllib.parse
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import rasterio
from mine_data_protocol import DATA,verify_assets

OUT=DATA/'radiometry_audit_20261004'
CASES={'Amyntaio':'Amyntaio','LasCruces':'LasCruces','Hpakant':'Hpakant','CN_XJ2023':'Xinjing'}
BAND_IDS={'B02':1,'B03':2,'B04':3,'B08':7}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radiometry import parse_xml


def run():
    verify_assets();OUT.mkdir(exist_ok=True);session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.8,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))))
    ledger_path=OUT/'audit.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'cases':{},'frozen_originals_modified':False,'model_predictions_run':0}
    for eid,foldername in CASES.items():
        folder=DATA/f'development/{foldername}';acq=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))
        catpath=folder/'catalogue.json' if foldername=='Xinjing' else DATA/f'scene_catalogues/{eid}_planetarycomputer.json'
        catalog=json.loads(catpath.read_text(encoding='utf-8-sig'));out=OUT/foldername;out.mkdir(exist_ok=True)
        ledger['cases'].setdefault(eid,{})
        for period,old in acq['images'].items():
            if period in ledger['cases'][eid]:continue
            scene=next(f for f in catalog['features'] if f['id']==old['scene_id']);href=scene['assets']['product-metadata']['href'];xmlpath=out/(period+'_product.xml')
            if not xmlpath.exists():
                r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(15,35));r.raise_for_status()
                r=session.get(r.json()['href'],timeout=(15,35));r.raise_for_status()
                if len(r.content)>5*1024*1024:raise ValueError('Unexpected XML size')
                ET.fromstring(r.content);xmlpath.write_bytes(r.content)
            calibration=parse_xml(xmlpath.read_bytes());original=folder/(period+'_BGRN_10m.tif');derivative=out/(period+'_BGRN_calibrated_tags.tif')
            with rasterio.open(original) as src:
                raw=src.read();profile=src.profile.copy();descs=src.descriptions;tags=src.tags()
            with rasterio.open(derivative,'w',**profile) as dst:
                dst.write(raw);dst.scales=tuple(calibration['scales']);dst.offsets=tuple(calibration['reflectance_offsets']);dst.descriptions=descs
                dst.update_tags(**{**tags,'radiometry_source_xml_sha':sha(xmlpath),'original_sha':sha(original),'role':'development_calibration_derivative'})
            record={**calibration,'source_unsigned_XML':href,'source_XML_sha':sha(xmlpath),'original_sha':sha(original),'derivative':str(derivative),'derivative_sha':sha(derivative),
                'raw_DN_samples_changed':False,'independent_ground_reflectance_validation':False,'scene_id':scene['id']}
            ledger['cases'][eid][period]=record;write(ledger_path,ledger)
            print(eid,period,'baseline',calibration['processing_baseline'],'DN offset',calibration['DN_offsets'],flush=True)
    verify_assets();print('Calibration metadata verified, all frozen original samples unchanged',flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
