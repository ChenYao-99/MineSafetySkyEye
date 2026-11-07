"""Inspect the small CC BY Century Mine data package; no predictive fitting.

The generated replay contains only explicitly labelled quantities. It does not
turn author retrospective precursor observations into a project warning score.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

ROOT = Path(r'D:\LMHLD\datasets\phase1_20261013\sources\century2014')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(data, delimiter=','):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter=delimiter))


def numeric(text):
    return None if text is None or not text.strip() else float(text)


def clean(row):
    return {str(k).strip(): v for k, v in row.items() if k is not None}


def main():
    meta = json.loads((ROOT / 'record_metadata.json').read_text(encoding='utf-8'))
    source = next(f for f in meta['files'] if f['key'] == 'Data.zip')
    raw = (ROOT / 'Data.zip').read_bytes()
    assert len(raw) == source['size']
    assert hashlib.md5(raw).hexdigest() == source['checksum'].split(':')[1]
    assert meta['metadata']['license']['id'] == 'cc-by-4.0'
    with ZipFile(io.BytesIO(raw)) as z:
        members = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    source_files = [dict(member=n, bytes=len(b), sha256=digest(b)) for n, b in members.items()]
    velocity_member = 'velocity-change-measurements/mean-velocity-changes.txt'
    velocity = []
    for index, source_row in enumerate(rows(members[velocity_member], '\t'), 2):
        r = clean(source_row)
        center = datetime.fromisoformat(r['Date'])
        value = float(r['Velocity Change (%)'])
        upper = float(r['Velocity Change plus Standard Deviation (%)'])
        lower = float(r['Velocity Change minus Standard Deviation (%)'])
        assert lower <= value <= upper
        assert abs((upper - value) - (value - lower)) < 1e-9
        velocity.append(dict(source_row=index, center_time_utc_assumed_from_paper=center.isoformat() + 'Z',
            value_percent=value, standard_deviation_percent=(upper-lower)/2,
            approximate_window_end_proxy_utc=(center + timedelta(days=1)).isoformat() + 'Z',
            availability_verified=False, source_member=velocity_member))
    intervals = Counter()
    for a, b in zip(velocity, velocity[1:]):
        intervals[str((datetime.fromisoformat(b['center_time_utc_assumed_from_paper'].replace('Z', '+00:00')) -
                       datetime.fromisoformat(a['center_time_utc_assumed_from_paper'].replace('Z', '+00:00'))).total_seconds())] += 1
    daily_rain = []
    for name in ['weather-data/IDCJAC0009_029167_2013_Data.csv', 'weather-data/IDCJAC0009_029167_2014_Data.csv']:
        for i, rr in enumerate(rows(members[name]), 2):
            r = clean(rr)
            day = f"{int(r['Year']):04}-{int(r['Month']):02}-{int(r['Day']):02}"
            daily_rain.append(dict(date_local=day, rainfall_mm=numeric(r['Rainfall amount (millimetres)']),
                accumulation_days=numeric(r['Period over which rainfall was measured (days)']),
                quality_flag=r['Quality'], source_row=i, source_member=name,
                observation_end_hour_verified=False))
    event_member = 'seismic-event-data/events-and-blasts.csv'
    events = rows(members[event_member])
    event_daily = Counter()
    all_event_times = []
    for r in events:
        d = datetime.fromtimestamp(float(r['EventTimeEpochSeconds']), timezone.utc)
        event_daily[d.date().isoformat()] += 1
        all_event_times.append(d)
    prism_member = 'radar-deformation-data/West Wall Prism Movements - 3 months to 20-Feb-14.csv'
    prism_rows = rows(members[prism_member])
    prism_bins = defaultdict(list)
    prism_times = []
    duplicate_keys = Counter()
    for r in prism_rows:
        d = datetime.strptime(r['Date'] + ' ' + r['Time'], '%d-%b-%y %H:%M:%S')
        sensor = r['Prism']
        v = float(r['Cumulative 3D Movement (mm) Since 20th Nov'])
        prism_bins[(sensor, d.date().isoformat())].append(v)
        prism_times.append(d)
        duplicate_keys[(sensor, d.isoformat())] += 1
    prism_daily = [dict(prism=sensor, date_unverified_local_time=day, samples=len(values),
                        median_cumulative_3d_mm=median(values), minimum_cumulative_3d_mm=min(values),
                        maximum_cumulative_3d_mm=max(values), source_member=prism_member,
                        availability_verified=False)
                   for (sensor, day), values in sorted(prism_bins.items())]
    replay = dict(schema_version=1, role='new_retrospective_multisensor_research_development_case',
        mine='Century Mine, Queensland, Australia', event_date_reported='2014-02-23',
        event_time_precision='date_only; timezone of failure statement not explicit',
        event_date_basis=dict(doi='10.26443/seismica.v5i1.1902', pdf_page=3),
        licence='CC BY 4.0', original_source_doi='10.5281/zenodo.14969229',
        velocity_changes=velocity,
        rain_daily=[r for r in daily_rain if '2013-12-25' <= r['date_local'] <= '2014-03-31'],
        seismic_events_daily_utc=[dict(date=d, recorded_events=n, blast_or_microseismic_labels_available=False)
                                  for d,n in sorted(event_daily.items())],
        prism_daily=prism_daily,
        future_data_protection='retrospective display only; no forecast target, threshold tuning, forward filling or future normalisation',
        actual_early_warning_or_geological_accuracy_validated=False)
    (ROOT / 'replay_inputs.json').write_text(json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    warnings = [
        'One known event already described in a paper; not an untouched blind evaluation.',
        'Mean velocity timestamps differ by 48h50m, although paper describes nonoverlapping two-day stacks; confirm temporal indexing before prospective scoring.',
        'Paper figure description extends to March31, but provided mean series ends March22; preserve actual file coverage.',
        'Mean measurements plotted at stack centres are not available at those centre timestamps; proxy window ends are assumptions, not proven production availability.',
        'Event CSV does not label individual rows as blasts or microseismic events; no invented event-type classifier labels.',
        'The two parseable deformation CSV files identify prisms. Do not relabel them as raw ground-based radar observations because of the enclosing directory name.',
        'slopeMovement.dat splits the prism ID into two CSV columns and has a header/data width mismatch; its matching Quikslope CSV should be used instead.',
        'Prism dates have no timezone, quality or operational latency metadata in CSV; weather accumulation endpoint needs confirmation.',
        'Local engineering Easting/Northing coordinates lack a supplied CRS; do not place them on optical maps without independent registration.',
        'No dated georeferenced optical imagery or accident polygon is supplied. This source does not close the optical pixel-boundary gap.',
        'The paper reports a retrospective precursor; its lead time is not this project performance.',
        'Early2014 event predates Sentinel-2 operations; any optical pairing needs an appropriate original sensor, not fabricated Sentinel-2 data.',
    ]
    report = dict(schema_version=1, audited_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        source=dict(landing='https://zenodo.org/records/14969229', licence=meta['metadata']['license'],
                    archive_sha256=digest(raw), official_md5=source['checksum'], official_size=source['size'],
                    checksum_passed=True, files=source_files,
                    paper='https://seismica.library.mcgill.ca/article/view/1902',
                    paper_sha256=digest((ROOT / 'de_wit_snieder_2026.pdf').read_bytes())),
        counts=dict(velocity_samples=len(velocity), rain_days_in_two_year_files=len(daily_rain),
                    seismic_events=len(events), prism_measurements=len(prism_rows),
                    prism_sensors=len({r['Prism'] for r in prism_rows}), prism_daily_records=len(prism_daily),
                    duplicate_prism_timestamps=sum(x-1 for x in duplicate_keys.values() if x>1)),
        coverage=dict(velocity_first=velocity[0]['center_time_utc_assumed_from_paper'],
                      velocity_last=velocity[-1]['center_time_utc_assumed_from_paper'],
                      velocity_interval_seconds=dict(intervals),
                      event_first=min(all_event_times).isoformat(), event_last=max(all_event_times).isoformat(),
                      prism_first=min(prism_times).isoformat(), prism_last=max(prism_times).isoformat()),
        admitted_for_retrospective_evidence_replay=True,
        admitted_for_prospective_warning_evaluation=False,
        admitted_for_accident_pixel_supervision=False,
        complete_accident_boundaries_added=0,
        new_model_predictions_or_fits=0, raw_seismic_correlations_downloaded=False,
        raw_archive_extracted_to_duplicate_files=False,
        replay=dict(path=str(ROOT / 'replay_inputs.json'), sha256=digest((ROOT / 'replay_inputs.json').read_bytes())),
        unresolved=warnings)
    (ROOT / 'qualification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['counts','coverage','admitted_for_retrospective_evidence_replay',
                                          'admitted_for_prospective_warning_evaluation','complete_accident_boundaries_added']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
