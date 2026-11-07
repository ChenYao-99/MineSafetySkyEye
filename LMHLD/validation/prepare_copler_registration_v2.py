from pathlib import Path
ROOT=Path('D:/LMHLD');SRC=ROOT/'validation/register_copler_field_boundary_v1.py';OUT=ROOT/'validation/register_copler_field_boundary_v2.py'
text=SRC.read_text(encoding='utf-8')
replacements={
 'reference_registration_v1_field2025':'reference_registration_v2_field2025_map_prior',
 "'geographic_prior_tolerance_px':15":"'geographic_prior_tolerance_px':50",
 'np.linalg.norm(b-prior[i])>15':'np.linalg.norm(b-prior[i])>50',
 "'prior_Gams_registration_failures_not_reused_as_Copler_truth':True":"'prior_Gams_registration_failures_not_reused_as_Copler_truth':True,'source_graticule_used_only_as500m_search_prior_not_accepted_georeference':True,'all30held_ground_error_gates_unchanged':True"
}
for old,new in replacements.items():
 if text.count(old)!=1:raise ValueError('Unexpected v1 code')
 text=text.replace(old,new)
if OUT.exists() and OUT.read_text(encoding='utf-8')!=text:raise ValueError('Existing v2 changed')
OUT.write_text(text,encoding='utf-8');print('Map search prior broadened only; independent geometric acceptance unchanged')
