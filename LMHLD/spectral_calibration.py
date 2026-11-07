"""Recover actual XML coefficients only for exact-hash known raw rasters; never guess offsets."""
import hashlib,json
from pathlib import Path

DATA=Path(__file__).resolve().parent/'datasets/mine_benchmark_20261002'


def known_calibration(raw):
    audit=DATA/'radiometry_audit_20261004/audit.json'
    if not audit.is_file():return None
    digest=hashlib.sha256(raw).hexdigest()
    records=json.loads(audit.read_text(encoding='utf-8'))
    for case,periods in records['cases'].items():
        for period,record in periods.items():
            if record.get('original_sha')!=digest:continue
            folder='Xinjing' if case=='CN_XJ2023' else case
            xml=DATA/f'radiometry_audit_20261004/{folder}/{period}_product.xml'
            if not xml.is_file() or hashlib.sha256(xml.read_bytes()).hexdigest()!=record['source_XML_sha']:
                raise ValueError('原始影像匹配，但已审核产品XML发生变化，不能自动恢复定标')
            # Read coefficients from actual XML, rather than trusting editable cached values.
            from radiometry import parse_xml
            coefficients=parse_xml(xml.read_bytes())
            return {'spectral_scales':coefficients['scales'],'spectral_offsets':coefficients['reflectance_offsets'],
                    'radiometry_status':'verified_local_product_XML','radiometry_source_XML_sha256':record['source_XML_sha'],
                    'radiometry_raw_sha256':digest,'radiometry_independent_ground_validation':False}
    return None
