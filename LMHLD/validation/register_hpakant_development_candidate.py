# -*- coding: utf-8 -*-
"""Register an observed mine for acquisition only; preserve a running study registry snapshot."""
import hashlib
import json
from mine_data_protocol import DATA, REGISTRY, registry, verify_assets


def run():
    context=json.loads((DATA/'downloads/hpakant_context.json').read_text(encoding='utf-8'))
    picture=DATA/'development/Hpakant/source_figure1.jpg'
    if not picture.read_bytes().startswith(b'\xff\xd8'): raise ValueError('Primary location source is not a JPEG')
    # This snapshot exactly preserves the registration provenance of the active
    # two-mine study when the global acquisition registry grows with another mine.
    study=DATA.parents[1]/'experiments/native_mine_transfer_20261003'
    planpath=study/'training_protocol.json'; snapshot=study/'training_registry_snapshot.json'
    if planpath.exists() and not snapshot.exists():
        plan=json.loads(planpath.read_text(encoding='utf-8'))
        if hashlib.sha256(REGISTRY.read_bytes()).hexdigest()!=plan['registry_sha256']:
            raise ValueError('Cannot manufacture the historical study registry snapshot')
        snapshot.write_bytes(REGISTRY.read_bytes())
    value=registry()
    if 'Hpakant' not in value['cases']:
        value['cases']['Hpakant']={'mine_id':'Hpakant','event_date':'2020-07-02','scope':'open_pit_wall_failure',
            'role':'development','quality_review_status':'pending','use_all_times_with_same_mine_role':True,
            'context':'downloads/hpakant_context.json','raw_source':'Copernicus S2L2A / anonymous public Planetary Computer',
            'label_source':context['source'],'previous_figure_evaluation':True,'native_model_predictions_run':0,
            'independent_expert_or_field_audit':False,'source_figure1_sha256':hashlib.sha256(picture.read_bytes()).hexdigest(),
            'unresolved_quality':['native visibility/SCL review','source-zone georeferencing','date and source-zone uncertainty']}
        verify_assets(value); REGISTRY.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Hpakant historical development candidate; no native label/training acceptance')


if __name__=='__main__': run()
