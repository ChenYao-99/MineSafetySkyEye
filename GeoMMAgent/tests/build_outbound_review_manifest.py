"""Prepare the exact minimum data types for user review; never make a cloud request."""
from pathlib import Path
import json,sys
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT));sys.path.insert(0,'D:/LMHLD')
from project_runtime import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from evidence_reporting import bounded_input
from knowledge_base import KnowledgeBase
OUT=Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005')
def run():
    bank=json.loads((OUT/'detector_bank.json').read_text(encoding='utf-8'));kb=KnowledgeBase();chunks=kb.retrieve('光学筛查 监测范围 人工复核 掩膜变化 量测',k=4)
    rows=[{'local_task_id':t['id'],'task_description':t['task'],'fixture_kind':t['fixture_kind'],
           'tool_fields_to_send':bounded_input(t['detection'])} for t in bank['tasks']]
    manifest={'status':'awaiting explicit approval after automatic approval rejection;no study cloud request performed',
              'proposed_destination':'existing project Qwen API at Alibaba Cloud DashScope;modelqwen-max',
              'purpose':'8 fixed objective reportingtasks;singlemodel/fixedflow/roleprototype comparison',
              'max_calls_per_task':3,'max_output_tokens_per_call':1400,'maximum_study_calls':24,
              'raw_images_exact_coordinates_filepaths_model_hashes_credentials_or_personal_data_in_proposed_payload':False,
              'own_derived_detection_stats_and_acquisition_dates_remain_outbound_data_requiring_approval':True,
              'detector_bank_sha256':file_sha256(OUT/'detector_bank.json'),
              'public_knowledge_summary':kb.format_context(chunks),'provided_reference_IDs':[c['id'] for c in chunks],
              'proposed_tasks':rows,'internal_task_IDs_kept_forlocalreceipt_only':True}
    text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n';(OUT/'outbound_review_manifest.json').write_text(text,encoding='utf-8')
    print('8 minimumfield packages prepared forreview;zero external requests;rawimages/coordinates/paths/keys excluded')
if __name__=='__main__':run()
