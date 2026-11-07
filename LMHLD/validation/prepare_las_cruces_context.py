# -*- coding: utf-8 -*-
"""Locate an already observed development mine from primary published coordinates."""
import hashlib
import json
from pathlib import Path
from mine_data_protocol import DATA, REGISTRY, registry, verify_assets


def prepare():
    pdf = DATA.parents[1] / 'experiments/mining_validation_20261001/sources/las_cruces_remote_analysis.pdf'
    context = {
        'case_id': 'LasCruces', 'role': 'development', 'event_date': '2019-01-23',
        'event_source': 'https://zenodo.org/records/4704955',
        'event_date_note': 'Use the research authors early morning January 23 account; some media use January 22.',
        'paper_path': str(pdf), 'paper_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'figure': 'Figure 3b, helicopter SfM orthoimage from 2019-01-25; external red boundary',
        'location_source': 'Figure 3 geographic axes: 6deg6minW, 6deg5minW, 37deg30minN',
        'coordinate_crosscheck_source': 'https://info.igme.es/sarai-movesar/site/Train_IGME_CRUCES_Mining%20Landslide',
        'coordinate_crosscheck': {
            'original_crs': 'EPSG:3035', 'points': [[2898498.992,1752857.582],[2898859.077,1753571.312]],
            'wgs84_points': [[-6.0954571658736825,37.49956295475415],[-6.092976208940952,37.50656670422075]],
            'meaning': 'published InSAR measurement locations; not accident boundary or our measured deformation'},
        'bbox_wgs84': [-6.115,37.485,-6.073,37.523],
        'pre_date': '2019-01-07', 'post_date': '2019-01-27',
        'pair_date_source': 'https://www.sentinelvision.eu/gallery/pdf/a4a9127053f2484d8fba09ab6c4a5c66',
        'pair_date_note': 'Same published Sentinel dates sought, not the authors 2016 pre or January25 helicopter image.',
        'label_quality': 'published figure reference requires geographic registration and uncertainty buffer',
        'independent_expert_or_field_audit': False, 'model_predictions_during_labeling': 0,
        'previous_figure_evaluation': True, 'never_claim_as_new_blind_mine': True,
    }
    path = DATA / 'downloads/las_cruces_context.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != context:
            raise ValueError('Existing context changed; review explicitly rather than overwrite')
    else:
        path.write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    value = registry()
    if 'LasCruces' not in value['cases']:
        value['cases']['LasCruces'] = {
            'mine_id':'LasCruces','event_date':'2019-01-23','scope':'open_pit_wall_failure',
            'role':'development','quality_review_status':'pending','use_all_times_with_same_mine_role':True,
            'raw_source':'Copernicus Sentinel-2 L2A; anonymous public Planetary Computer',
            'label_source':context['event_source'],'label_quality':context['label_quality'],
            'independent_expert_or_field_audit':False,'context':'downloads/las_cruces_context.json',
            'native_model_predictions_run':0,'previous_figure_evaluation':True,
            'unresolved_quality':['original pair acquisition','boundary georeferencing','visibility and normal-pit negative review']}
        verify_assets(value)
        REGISTRY.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Las Cruces registered only for development; reserved final events untouched')


if __name__ == '__main__':
    prepare()
