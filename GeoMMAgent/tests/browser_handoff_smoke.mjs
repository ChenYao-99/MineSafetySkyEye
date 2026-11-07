import fs from 'node:fs/promises';
const out=process.env.KAT_HANDOFF_SMOKE_DIR || 'D:/GeoMMAgent/cache/handoff_browser_smoke_20261005';
const base=process.env.KAT_HANDOFF_SMOKE_URL || 'http://127.0.0.1:5105';
const requireKnown=process.argv.includes('--expect-known-work');
await fs.mkdir(out,{recursive:true});
let target;
for(let i=0;i<25;i++){
 try{target=(await(await fetch('http://127.0.0.1:9222/json')).json()).find(p=>p.type==='page');if(target)break;}catch{}
 await new Promise(r=>setTimeout(r,500));
}
if(!target)throw new Error('Acceptance browser unavailable');
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true})});
let id=0;const waiting=new Map();
const downloadEvents=[];
ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.method==='Browser.downloadWillBegin')downloadEvents.push({event:v.method,guid:v.params.guid,filename:v.params.suggestedFilename});if(v.method==='Browser.downloadProgress')downloadEvents.push({event:v.method,guid:v.params.guid,state:v.params.state,receivedBytes:v.params.receivedBytes})});
ws.addEventListener('message',e=>{const v=JSON.parse(e.data),p=waiting.get(v.id);if(p){waiting.delete(v.id);v.error?p.reject(new Error(JSON.stringify(v.error))):p.resolve(v.result)}});
function cmd(method,params={}){return new Promise((resolve,reject)=>{const k=++id;waiting.set(k,{resolve,reject});ws.send(JSON.stringify({id:k,method,params}))})}
async function value(expression){const r=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await cmd('Page.enable');
await cmd('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:out.replaceAll('/','\\'),eventsEnabled:true});
await cmd('Page.navigate',{url:base+'/'});
for(let i=0;i<50;i++){if(await value("typeof renderPipelineResult==='function'"))break;await new Promise(r=>setTimeout(r,100));}
await value("window.testAlerts=[];window.alert=x=>testAlerts.push(String(x));");
const fixture=await value("fetch('/api/local_handoff_smoke_fixture').then(r=>r.json())");
if(fixture.fixture_version!=='authoritative_facts_and_public_work_context_v2')throw new Error('Stale test server fixture');
if(requireKnown&&(!fixture.known_work||fixture.rows.find(x=>x.slot==='upload').result.detection.work_context?.status!=='matched_public_expected_work_reference'))throw new Error('Expected real work-context fixture missing');
for(const row of fixture.rows){if(!row.result.objective_facts)throw new Error('Current authoritative fact fields missing');}
await value(`window.fixture=${JSON.stringify(fixture)};fixture.rows.forEach(row=>renderPipelineResult(row.result,row.slot));`);
const rendered=await value("fixture.rows.map(row=>({slot:row.slot,id:reportParams[row.slot].report_id,steps:rp(row.slot,'steps').textContent,actions:rp(row.slot,'actions').textContent}))");
for(const row of rendered){
 if(!row.steps.includes('人工复核交接')||!row.actions.includes('JSON'))throw new Error('Review handoff invisible');
 if(row.id!==fixture.rows.find(x=>x.slot===row.slot).result.report_id)throw new Error('Cross-slot identity mismatch');
}
const exports=[];
for(const row of fixture.rows){
 const expected=await value(`fetch('/api/review_record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:${JSON.stringify(row.result.report_id)}})}).then(r=>r.json())`);
 await value(`exportReviewRecord(${JSON.stringify(row.slot)})`);
 const path=out+'/'+expected.filename;let record;
 for(let i=0;i<50;i++){try{record=JSON.parse(await fs.readFile(path,'utf8'));break;}catch{}await new Promise(r=>setTimeout(r,100));}
 if(!record||record.trace.request_id!==row.result.detection.request_id||record.trace.report_id!==row.result.report_id)throw new Error('Downloaded JSON identity mismatch: '+JSON.stringify({path,files:await fs.readdir(out),downloadEvents,alerts:await value('testAlerts'),actual_request_id:record?.trace?.request_id,expected_request_id:row.result.detection.request_id}));
 if(record.task_contract.purpose!=='光学影像疑似异常筛查、量测与人工复核辅助')throw new Error('UTF8 Chinese download corrupted');
 if(record.accepted_human_review_decisions.length||record.project_field_trial_or_expert_endorsement)throw new Error('False human trial claim');
 if(row.result.detection.work_context?.status==='matched_public_expected_work_reference'){
   if(record.work_context.predicted_pixels_overlapping_work_reference!==row.result.detection.work_context.predicted_pixels_overlapping_work_reference)throw new Error('Work context changed on export');
   if(record.objective_facts.site_stability_confirmed)throw new Error('Work context turned into safety');
   if(!await value("rp('upload','steps').textContent.includes('公开采矿作业参考')"))throw new Error('Work evidence hidden');
 }
 exports.push({slot:row.slot,path,request_identity_correct:true,UTF8_Chinese_correct:true});
}
const alerts=await value('testAlerts');if(alerts.length)throw new Error(JSON.stringify(alerts));
await value("document.querySelectorAll('.scene').forEach(el=>el.style.display='none');document.getElementById('scene-mine').style.display='block';rp('upload','steps').scrollIntoView({block:'center',behavior:'instant'});");
await new Promise(r=>setTimeout(r,200));
const shot=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
await fs.writeFile(out+'/review_handoff.png',Buffer.from(shot.data,'base64'));
await fs.writeFile(out+'/result.json',JSON.stringify({passed:true,real_browser:'Microsoft Edge headless',exports,
 authoritative_facts_required:true,known_work_required:requireKnown,
 scope:'actual browser UI and cached-record API download;not human trial or accuracy evaluation',
 inference_and_cloud_disabled:true,new_classifier_predictions:0},null,2));
console.log('PASS real browser upload+compare JSON downloads, same-request identity, UTF8 Chinese and pending review display');
await fetch(base+'/api/local_handoff_smoke_shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
ws.close();
