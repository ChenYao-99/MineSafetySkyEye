"""Audit mine-geometry reference rows and exclude entire reserved surroundings before use."""
from pathlib import Path
import sqlite3,json,sys
from datetime import datetime
from mine_data_protocol import DATA,registry,verify_assets
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/mine_geometry_auxiliary_20261004'


def quoted(name):return '"'+name.replace('"','""')+'"'


def reserved_envelopes():
    raw=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/vector_data/landslide_inventory.gpkg'
    source=sqlite3.connect(raw.as_uri()+'?mode=ro',uri=True)
    layer=source.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
    geom=source.execute('select column_name from gpkg_geometry_columns where table_name=?',(layer,)).fetchone()[0]
    rtree='rtree_'+layer+'_'+geom
    result={}
    for case in ('CL0002','KG0001'):
        row=source.execute('select min(r.minx),min(r.miny),max(r.maxx),max(r.maxy) from '+quoted(layer)+' f join '+quoted(rtree)+' r on f.fid=r.id where f.EID=?',(case,)).fetchone()
        if row[0] is None:raise ValueError('Missing reserved event envelope')
        result[case]=[row[0]-.05,row[1]-.05,row[2]+.05,row[3]+.05]
    context=json.loads((DATA/'reserved_candidates/US_BC2021/context.json').read_text(encoding='utf-8'))
    b=context['bbox_wgs84'];result['US_BC2021']=[b[0]-.05,b[1]-.05,b[2]+.05,b[3]+.05]
    return result


def run():
    verify_assets();envelopes=reserved_envelopes()
    connection=sqlite3.connect((OUT/'mining_area_data.gpkg').as_uri()+'?mode=ro',uri=True)
    columns=[r[1] for r in connection.execute('pragma table_info(tiles)') if r[1]!='geom']
    select=','.join('t.'+quoted(c) for c in columns)
    rows=[]
    for row in connection.execute('select '+select+',r.minx,r.miny,r.maxx,r.maxy from tiles t join rtree_tiles_geom r on t.fid=r.id'):
        fields=dict(zip(columns,row[:-4]));b=row[-4:];protected=[]
        for case,limit in envelopes.items():
            if b[0]<=limit[2] and b[2]>=limit[0] and b[1]<=limit[3] and b[3]>=limit[1]:protected.append(case)
        fields.update(envelope_wgs84=list(b),reserved_overlap=protected,eligible_for_development_geometry=not protected)
        rows.append(fields)
    def counts(key):
        result={}
        for row in rows:result[str(row.get(key))]=result.get(str(row.get(key)),0)+1
        return result
    record={'checked_at':datetime.now().isoformat(timespec='seconds'),'raw_rows':len(rows),
            'published_metadata_claimed_sites':1210,'actual_rows_are_not_yet_verified_unique_mines':True,
            'distribution':{k:counts(k) for k in ('minetype1','minetype2','preferred_dataset','split')},
            'reserved_envelopes_with_conservative_buffer':envelopes,
            'excluded_rows':[r for r in rows if r['reserved_overlap']],
            'all_rows':rows,'image_date_source':'acquisition token in s2_tile_id; timestamp is annotation metadata, not image acquisition',
            'failure_or_stability_labels':False,'no_model_training_or_prediction':True,
            'official_crop_split_not_independent_mine_split':True}
    (OUT/'split_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'raw_rows':len(rows),'distributions':record['distribution'],'reserved_rows_excluded':len(record['excluded_rows']),
                      'excluded_tile_ids':[r['tile_id'] for r in record['excluded_rows']]}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
