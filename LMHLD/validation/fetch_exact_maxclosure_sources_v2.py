"""Retrieve exact publisher-linked attachments, preserving earlier failed attempts."""
import sys,json,concurrent.futures
from pathlib import Path
sys.path.insert(0,'D:/LMHLD/validation')
from fetch_maxclosure_sources_20261005 import fetch,OUT
SOURCES={
 'cl_before_reference_exact.jpg':'https://blogs.agu.org/landslideblog/files/2021/11/21_11-Ministro-Hales-1.jpg',
 'cl_after_reference_exact.jpg':'https://blogs.agu.org/landslideblog/files/2021/11/21_11-Ministro-Hales-2.jpg',
 'dlr_dataset_description_api.pdf':'https://zenodo.org/api/records/17007637/files/S1S2_dataset.pdf/content'}
if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8')
 with concurrent.futures.ThreadPoolExecutor(3) as pool:rows=list(pool.map(fetch,SOURCES.items()))
 (OUT/'exact_source_acquisition_v2.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
