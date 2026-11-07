"""Acquire public evidence only; never infer safety labels from a compliance report."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, urllib.request
import fitz

OUT = Path('D:/LMHLD/datasets/actual_operation_evidence_20261005/Amrun')
SOURCES = {
    'annual_compliance_2024': 'https://www.riotinto.com/-/media/content/documents/operations/weipa/2024-epbc-compliance-report.pdf?rev=76b88f14a55d4ed79cc0cefcc8075e2e',
    'independent_audit_2025': 'https://www.riotinto.com/-/media/content/documents/operations/weipa/2025-weipa-independent-compliance-audit-report.pdf?rev=e4096e9b2fca44c68b73a1ad32b48dce',
}

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for key, url in SOURCES.items():
        p = OUT / (key + '.pdf')
        if not p.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'GeoMMAgent-public-evidence-review/1.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(25 * 1024 * 1024 + 1)
            if len(data) > 25 * 1024 * 1024 or not data.startswith(b'%PDF'):
                raise ValueError('Expected bounded public PDF')
            p.write_bytes(data)
        doc = fitz.open(p)
        hits = []
        for index, page in enumerate(doc):
            txt = page.get_text()
            terms = ('clearing', 'shape file', 'shapefile', 'end of month', '12 May 2023', 'inspection', 'rehabilitation')
            matched = [term for term in terms if term in txt.lower()]
            if matched:
                (OUT / f'{key}_page_{index+1:03d}.txt').write_text(txt, encoding='utf-8')
                hits.append({'PDF_page_1based': index+1, 'matched_terms': matched})
        records.append({'id': key, 'URL': url, 'local_file': str(p), 'bytes': p.stat().st_size,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'pages': len(doc), 'review_pages': hits})
        print(key, 'pages', len(doc), 'relevant pages', [x['PDF_page_1based'] for x in hits], flush=True)
    receipt = {'acquired_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
               'records': records, 'role': 'public temporal and execution evidence, not spatial safety ground truth',
               'our_project_field_trial_or_expert_endorsement': False,
               'pixel_labels_or_classifier_predictions_created': False}
    (OUT / 'acquisition.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    run()
