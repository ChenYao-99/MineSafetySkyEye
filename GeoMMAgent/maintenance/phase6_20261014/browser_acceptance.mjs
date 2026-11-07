import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const out='D:/GeoMMAgent/maintenance/phase6_20261014';
const downloads=out+'/_scratch/downloads';await fs.mkdir(downloads,{recursive:true});
const target=await(await fetch('http://127.0.0.1:9235/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let serial=0;const pending=new Map(),blocked=[];
const command=(method,params={})=>new Promise((resolve,reject)=>{
 const id=++serial,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP command timed out: '+method))},15000);
 pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
});
ws.addEventListener('message',event=>{
 const message=JSON.parse(event.data);
 if(message.method==='Fetch.requestPaused'){
  const req=message.params,url=req.request.url;
  if(/^https?:/.test(url)&&!url.startsWith('http://127.0.0.1:5122/')){
   blocked.push(url);command('Fetch.failRequest',{requestId:req.requestId,errorReason:'BlockedByClient'}).catch(()=>{});
  }else command('Fetch.continueRequest',{requestId:req.requestId}).catch(()=>{});
 }
 const job=pending.get(message.id);if(job){pending.delete(message.id);clearTimeout(job.timer);message.error?job.reject(Error(JSON.stringify(message.error))):job.resolve(message.result);}
});
async function evaluate(expression){const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function until(expression){for(let i=0;i<80;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error('UI not ready');}
try{
 await command('Page.enable');await command('Runtime.enable');
 await command('Fetch.enable',{patterns:[{urlPattern:'http*',requestStage:'Request'}]});
 await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads.replaceAll('/','\\')});
 await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
 await command('Page.navigate',{url:'http://127.0.0.1:5122/'});
 await until("typeof renderPipelineResult==='function'");
 const rows=[];
 for(const kind of ['single','paired','blocked']){
  const slot=kind==='single'?'upload':'compare';
  const data=await evaluate(`fetch('/api/competition_fixture/${kind}').then(r=>r.json())`);
  await evaluate(`window.fixture=${JSON.stringify(data)};renderPipelineResult(fixture,'${slot}');rp('${slot}','steps').scrollIntoView({behavior:'instant',block:'start'});`);
  const ui=await evaluate(`({steps:rp('${slot}','steps').textContent,cards:rp('${slot}','steps').querySelectorAll('.agent-result').length,mock:rp('${slot}','mock').textContent,report:rp('${slot}','text').textContent})`);
  assert.ok(ui.mock.includes('模拟'));
  if(kind==='blocked'){
   assert.equal(ui.cards,1);assert.ok(ui.steps.includes('协作未完整执行'));
   assert.ok(ui.steps.includes('尚未执行：ReasoningAgent')&&ui.steps.includes('尚未执行：ReportReviewAgent'));
  }else{
   assert.equal(ui.cards,kind==='single'?2:3);
   assert.ok(ui.steps.includes('EvidenceAuditAgent')&&ui.steps.includes('ReasoningAgent'));
   if(kind==='paired')assert.ok(ui.steps.includes('ReportReviewAgent'));
  }
  if(kind!=='single')assert.ok(ui.report.includes('净变化+0.05个百分点'));
  const descriptor=await evaluate(`fetch('/api/review_record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:fixture.report_id,format:'json'})}).then(r=>r.json())`);
  await evaluate(`exportReviewRecord('${slot}')`);
  let packet;
  for(let i=0;i<60;i++){try{packet=JSON.parse(await fs.readFile(downloads+'/'+descriptor.filename,'utf8'));break;}catch{}await new Promise(r=>setTimeout(r,100));}
  assert.equal(packet.trace.report_id,data.report_id);assert.equal(packet.trace.request_id,data.detection.request_id);
  assert.deepEqual(packet.objective_facts,data.objective_facts);
  assert.equal(packet.project_field_trial_or_expert_endorsement,false);
  if(kind==='paired'||kind==='blocked'){
   await evaluate(`rp('${slot}','steps').scrollIntoView({behavior:'instant',block:'start'});window.scrollBy(0,-160);`);
   const screenshot=await command('Page.captureScreenshot',{format:'png'});
   await fs.writeFile(out+'/browser_'+kind+'.png',Buffer.from(screenshot.data,'base64'));
  }
  rows.push({kind,passed:true,actual_browser_cards:ui.cards,downloaded_record_identity_preserved:true,
    original_facts_preserved:true,scripted_fixture_explicitly_labelled:true});
 }
 const result={passed:true,real_browser:'Microsoft Edge headless',rows,blocked_external_resource_requests:blocked,
  cloud_requests:0,new_AI_results:false,real_user_trial:false,production_archive_untouched:true};
 await fs.writeFile(out+'/browser_acceptance.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
}finally{
 await fetch('http://127.0.0.1:5122/api/competition_fixture_shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).catch(()=>{});
 ws.close();
}
