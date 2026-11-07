"""Separate strict formatting errors from substantive values; preserve frozen raw scores."""
from pathlib import Path
import json
OUT=Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005')

def run():
    data=json.loads((OUT/'reporting_results.json').read_text(encoding='utf-8'))
    rows=[]
    for row in data['models_and_fixed_outputs']:
        categories={'scope_alias_from_ambiguous_schema':[],'omitted_all_null_change_container':[],
                    'missing_conflict_counter_default_not_numeric_geology_error':[],'substantive_field_errors':[]}
        for error in row['score']['errors']:
            field=error['field']
            if field=='facts.analysis_scope' and error['expected']=='whole_image' and error['actual']=='whole':
                key='scope_alias_from_ambiguous_schema'
            elif field.startswith('facts.change.') and error['expected'] is None and error['actual']=='__missing__':
                key='omitted_all_null_change_container'
            elif field=='facts.unverified_pixels' and error['expected']==0 and error['actual'] is None:
                key='missing_conflict_counter_default_not_numeric_geology_error'
            else:key='substantive_field_errors'
            categories[key].append(error)
        rows.append({'task_id':row['task_id'],'arm':row['arm'],'strict_errors':row['score']['wrong_fields'],
                     'categories':categories,'mechanical_summary_errors':row['score'].get('summary_mechanical_contract_errors',[])})
    summary={}
    for arm in data['summary']:
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]={k:sum(len(r['categories'][k]) for r in rr) for k in rr[0]['categories']}
        summary[arm]['mechanical_summary_error_cases']=sum(bool(r['mechanical_summary_errors']) for r in rr)
        summary[arm]['logical_tokens']=data['summary'][arm]['logical_token_total']
    result={'raw_scores_and_provider_receipts_preserved':True,
            'scope_schema_ambiguity_found':'schema offered whole while input carried whole_image;cannot call this geological factual error',
            'only_reporting_stage_prototype_not_full_GeoMMAgent_or_segmentation_accuracy':True,
            'fixed_fields_are_exact_by_construction_not_independent_AI_accuracy_evidence':True,
            'data':rows,'summary':summary,
            'live_study_calls':data['actual_paid_calls_initialwriter_shared_not_triplecounted'],
            'provider_tokens':data['actual_recorded_provider_tokens'],
            'next_engineering_action':'fill authoritative facts from detector;preserve generated interpretations separately;canonicalize future schema before any repeat study',
            'extra_cloud_calls':0}
    (OUT/'reporting_score_interpretation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':run()
