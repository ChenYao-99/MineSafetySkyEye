const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('D:/GeoMMAgent/矿安天眼_前端演示(后端版).html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={};
function el(id){return elements[id] ||= {id,style:{},value:'',files:[],src:'',textContent:'',innerHTML:'',
  classList:{add(){},remove(){},toggle(){}},addEventListener(){},scrollIntoView(){},click(){},
  getContext(){return {clearRect(){},drawImage(){},fillRect(){},strokeRect(){}}},querySelectorAll(){return []}};}
const sb={console,document:{getElementById:el,querySelectorAll(){return []},createElement(tag){
  if(tag==='a')return el('download');
  return {innerHTML:'',set textContent(value){this.innerHTML=String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}};
}},
 window:{addEventListener(){}},Image:class{},URL:{createObjectURL(){},revokeObjectURL(){}},FileReader:class{},
 setInterval(){},setTimeout(){},clearInterval(){},requestAnimationFrame(){},alert(){},Blob:class{},atob:x=>x,btoa:x=>x,
 fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sb);vm.runInContext(script,sb);
async function run(){
 const data={report_id:'UPLOAD_CURRENT',agents:[],final_answer:'facts',knowledge:[],
  objective_facts:{analysis_scope:'whole_image',kind:'single_epoch_mask',reported_regions:2,area_percent:null,largest_area_px:10,largest_area_m2:null},
  detection:{work_context:{status:'matched_public_expected_work_reference',note:'<script>fake</script>',predicted_pixels_overlapping_work_reference:10,source_period:'nominalannual'}},
  review_handoff:{task_contract:{purpose:'影像辅助核查'},pending_checks:[{task:'<script>bad</script>'}],review_items_truncated:true}};
 sb.data=data;vm.runInContext("renderPipelineResult(data,'upload')",sb);
 assert.ok(el('rp-upload-steps').innerHTML.includes('&lt;script&gt;'));
 assert.ok(!el('rp-upload-steps').innerHTML.includes('<script>bad'));
 assert.ok(el('rp-upload-actions').innerHTML.includes('exportReviewRecord'));
 assert.ok(el('rp-upload-steps').innerHTML.includes('本次检测事实'));
 assert.ok(el('rp-upload-steps').innerHTML.includes('掩膜占比：不可计算'));
 assert.ok(el('rp-upload-steps').innerHTML.includes('不可得 平方米'));
 assert.ok(el('rp-upload-steps').innerHTML.includes('公开采矿作业参考'));
 assert.ok(!el('rp-upload-steps').innerHTML.includes('<script>fake'));
 let sent;
 sb.fetch=async(url,opts)=>{sent={url,body:JSON.parse(opts.body)};return {ok:true,json:async()=>({content_b64:'JSON',filename:'review.json'})}};
 await vm.runInContext("exportReviewRecord('upload')",sb);
 assert.deepEqual(sent.body,{report_id:'UPLOAD_CURRENT',format:'json'});assert.ok(sent.url.endsWith('/review_record'));
 sb.data={...data,report_id:'COMPARE_CURRENT'};vm.runInContext("renderPipelineResult(data,'compare')",sb);
 await vm.runInContext("exportReviewRecord('compare')",sb);assert.equal(sent.body.report_id,'COMPARE_CURRENT');
 await vm.runInContext("exportReviewRecord('upload')",sb);assert.equal(sent.body.report_id,'UPLOAD_CURRENT');
 assert.equal(el('download').download,'review.json');
 sb.fetch=async(url,opts)=>{sent={url,body:JSON.parse(opts.body)};return {ok:true,json:async()=>({content_b64:'HTML',filename:'review.html',mime_type:'text/html'})}};
 await vm.runInContext("exportReviewRecord('upload','html')",sb);
 assert.deepEqual(sent.body,{report_id:'UPLOAD_CURRENT',format:'html'});
 assert.equal(el('download').download,'review.html');
 sb.data={...data,report_id:'ARCHIVE_CURRENT'};vm.runInContext("renderPipelineResult(data,'archive')",sb);
 await vm.runInContext("exportReviewRecord('archive','html')",sb);
 assert.deepEqual(sent.body,{report_id:'ARCHIVE_CURRENT',format:'html'});
 assert.equal(vm.runInContext('reportParams.upload.report_id',sb),'UPLOAD_CURRENT');
 await vm.runInContext("exportReport('upload')",sb);
 assert.ok(sent.url.endsWith('/report_upload'));
 assert.deepEqual(sent.body,{report_id:'UPLOAD_CURRENT'});
 assert.ok(el('rp-compare-actions').innerHTML.includes("exportCompareReport('html')"));
 assert.ok(el('rp-compare-actions').innerHTML.includes("exportCompareReport('json')"));
 await vm.runInContext("exportCompareReport('html')",sb);
 assert.ok(sent.url.endsWith('/report_compare'));
 assert.deepEqual(sent.body,{report_id:'COMPARE_CURRENT',format:'html'});
 await vm.runInContext("exportCompareReport('json')",sb);
 assert.deepEqual(sent.body,{report_id:'COMPARE_CURRENT',format:'json'});
 vm.runInContext("hideReport('compare')",sb);
 assert.equal(vm.runInContext('reportParams.compare',sb),null);
 assert.equal(vm.runInContext('reportParams.upload.report_id',sb),'UPLOAD_CURRENT');
 sb.data={...data};delete sb.data.report_id;vm.runInContext("renderPipelineResult(data,'upload')",sb);
 assert.equal(vm.runInContext('reportParams.upload',sb),null);
 let exportCalls=0;
 sb.fetch=async()=>{exportCalls++;throw new Error('Missing report ID must not regenerate analysis');};
 vm.runInContext("reportParams.upload={image:'UNVERIFIED_RAW_INPUT',model_tag:'mine'}",sb);
 await vm.runInContext("exportReport('upload')",sb);
 assert.equal(exportCalls,0);
 console.log('PASS cached record identity, independent upload/compare/archive slots, escaped pending tasks and JSON/HTML download');
}
run().catch(e=>{console.error(e);process.exitCode=1});
