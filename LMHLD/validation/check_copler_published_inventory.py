"""Read-only spatial check for this named mining accident, not model selection."""
from pathlib import Path
import sqlite3,json,sys
ROOT=Path('D:/LMHLD');DATA=ROOT/'datasets/mine_benchmark_20261002'
FILE=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/vector_data/landslide_inventory.gpkg'
OUT=ROOT/'datasets/pixel_boundary_sources_20261004'
def quote(n):return '"'+n.replace('"','""')+'"'
con=sqlite3.connect(FILE.as_uri()+'?mode=ro',uri=True);rows=[]
for table,column,srs in con.execute('SELECT table_name,column_name,srs_id FROM gpkg_geometry_columns'):
 if srs!=4326:raise ValueError('Unexpected inventory CRS; no unverified geographic query')
 rt='rtree_'+table+'_'+column
 hits=con.execute('SELECT id FROM '+quote(rt)+' WHERE minx<? AND maxx>? AND miny<? AND maxy>?',(38.57,38.48,39.46,39.39)).fetchall()
 names=[p[1] for p in con.execute('PRAGMA table_info('+quote(table)+')') if p[1]!=column]
 for fid, in hits:
  cur=con.execute('SELECT '+','.join(quote(n) for n in names)+' FROM '+quote(table)+' WHERE fid=?',(fid,))
  rows.extend(dict(zip(names,r)) for r in cur.fetchall())
result={'source':'https://zenodo.org/records/17007637','bbox':[38.48,39.39,38.57,39.46],'matched_rows':rows,'model_predictions':0,'zero_rows_means_only_this_inventory_not_all_available_sources':True}
(OUT/'copler_published_inventory_query.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
