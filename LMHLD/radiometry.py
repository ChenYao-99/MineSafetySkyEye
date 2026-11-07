"""Actual Sentinel-2 XML coefficients; no network/research-runner dependency."""
import xml.etree.ElementTree as ET
import numpy as np

BAND_IDS={'B02':1,'B03':2,'B04':3,'B08':7}


def parse_xml(data):
    root=ET.fromstring(data)
    def find(name):return [e for e in root.iter() if e.tag.split('}')[-1]==name]
    values=find('BOA_QUANTIFICATION_VALUE')
    if len(values)!=1:raise ValueError('Actual BOA quantification metadata required')
    quant=float(values[0].text)
    baselines=find('PROCESSING_BASELINE')
    if len(baselines)!=1:raise ValueError('Unique actual processing baseline required')
    baseline=baselines[0].text;number=float(baseline)
    if not np.isfinite(number):raise ValueError('Invalid processing baseline')
    rows=find('BOA_ADD_OFFSET');offsets={int(e.attrib['band_id']):float(e.text) for e in rows}
    if len(offsets)!=len(rows):raise ValueError('Duplicate BOA offset band IDs')
    if number>=4 and not set(BAND_IDS.values()).issubset(offsets):raise ValueError('Modern baseline must provide all selected BOA offsets')
    if offsets and not set(BAND_IDS.values()).issubset(offsets):raise ValueError('Incomplete selected BOA offsets')
    if not np.isfinite(list(offsets.values())).all():raise ValueError('Nonfinite BOA offsets')
    if not np.isfinite(quant) or quant<=0:raise ValueError('Invalid BOA quantification')
    return {'processing_baseline':baseline,'BOA_quantification':quant,'scales':[1/quant]*4,
            'DN_offsets':[offsets.get(BAND_IDS[b],0.) for b in BAND_IDS],
            'reflectance_offsets':[offsets.get(BAND_IDS[b],0.)/quant for b in BAND_IDS],
            'offset_semantics':'reflectance=(DN+BOA_ADD_OFFSET)/BOA_QUANTIFICATION; DN0 remains no-observation'}
