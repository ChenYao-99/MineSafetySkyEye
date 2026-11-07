# -*- coding: utf-8 -*-
"""Read published vector attributes before deciding which events are mine failures."""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002"
path=ROOT/"raw_sources/dlr/s1s2_landslide_reference_data/vector_data/landslide_inventory.gpkg"


def quoted(name):
    return '"'+name.replace('"','""')+'"'


def audit():
    connection=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
    layers=connection.execute("select table_name,data_type,srs_id from gpkg_contents").fetchall()
    records=[]
    summaries=[]
    for table,kind,srs in layers:
        if kind!="features":
            continue
        cols=connection.execute("pragma table_info("+quoted(table)+")").fetchall()
        names=[c[1] for c in cols if c[2].upper() not in ("BLOB","GEOMETRY","MULTIPOLYGON","POLYGON") and c[1].lower() not in ("geom","geometry")]
        cursor=connection.execute("select "+",".join(quoted(c) for c in names)+" from "+quoted(table))
        rows=[dict(zip(names,row)) for row in cursor.fetchall()]
        records.extend(rows)
        summaries.append({"table":table,"srs_id":srs,"columns":[c[1:3] for c in cols],"feature_n":len(rows)})
    events=defaultdict(list)
    for row in records:
        events[row.get("EID",row.get("eid","unknown"))].append(row)
    event_rows=[]
    for eid,rows in sorted(events.items()):
        example=rows[0]
        event_rows.append({"event_id":eid,"polygon_n":len(rows),"example":example})
    result={"source":"https://zenodo.org/records/17007637","layers":summaries,"events":event_rows,
            "mine_label_policy":"Confirm mine identity and failure semantics from original event sources; never convert all natural events into mining positives"}
    (ROOT/"inventory_attribute_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    audit()
