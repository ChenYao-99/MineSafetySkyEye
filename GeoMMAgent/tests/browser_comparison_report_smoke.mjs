import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const out=process.env.KAT_COMPARE_SMOKE_DIR || 'D:/GeoMMAgent/cache/comparison_report_acceptance_20261006';
const base=process.env.KAT_COMPARE_SMOKE_URL || 'http://127.0.0.1:5105';
const debug=process.env.KAT_COMPARE_DEBUG_URL || 'http://127.0.0.1:9223';
await fs.mkdir(out,{recursive:true});
let target;
for(let i=0;i<30;i++){
  try{
    // Downloads-hub popup targets cannot reliably emulate viewport or print.
    target=await(await fetch(debug+'/json/new?about:blank',{method:'PUT'})).json();
    if(target.type==='page'&&target.webSocketDebuggerUrl)break;
    target=null;
  }catch{}
  await new Promise(resolve=>setTimeout(resolve,200));
}
if(!target)throw new Error('Owned acceptance browser unavailable');
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;const pending=new Map();
ws.addEventListener('message',event=>{const row=JSON.parse(event.data),item=pending.get(row.id);if(item){pending.delete(row.id);row.error?item.reject(new Error(JSON.stringify(row.error))):item.resolve(row.result)}});
function command(method,params={}){return new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}))})}
async function value(expression){const row=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(row.exceptionDetails)throw new Error(JSON.stringify(row.exceptionDetails));return row.result.value}
async function awaitFile(path){for(let i=0;i<50;i++){try{return await fs.readFile(path);}catch{}await new Promise(resolve=>setTimeout(resolve,100));}throw new Error('Download missing: '+path)}
try{
  await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:out.replaceAll('/','\\'),eventsEnabled:true});
  await command('Page.navigate',{url:base+'/'});
  for(let i=0;i<50;i++){if(await value("typeof renderPipelineResult==='function'"))break;await new Promise(resolve=>setTimeout(resolve,100));}
  await value("window.testAlerts=[];window.alert=message=>testAlerts.push(String(message));");
  const fixture=await value("fetch('/api/local_handoff_smoke_fixture').then(response=>response.json())");
  const compare=fixture.rows.find(row=>row.slot==='compare');if(!compare)throw new Error('Paired fixture missing');
  await value(`(async()=>{window.originalTestFetch=window.fetch;window.pendingTestRequests=[];
    window.fetch=(url,options)=>String(url).endsWith('/detect_upload')?new Promise(resolve=>pendingTestRequests.push({resolve,options})):originalTestFetch(url,options);
    const canvas=document.createElement('canvas');canvas.width=canvas.height=32;canvas.getContext('2d').fillRect(0,0,32,32);
    const bytes=Uint8Array.from(atob(canvas.toDataURL('image/png').split(',')[1]),letter=>letter.charCodeAt(0));
    window.fileA=new File([bytes],'old.png',{type:'image/png'});window.fileB=new File([bytes],'new.png',{type:'image/png'});
    await handleMineFile(fileA);})()`);
  for(let i=0;i<50;i++){if(await value('roiImg !== null'))break;await new Promise(resolve=>setTimeout(resolve,20));}
  await value('window.oldUploadTask=runUploadDetect();void 0;');
  for(let i=0;i<50;i++){if(await value('pendingTestRequests.length===1'))break;await new Promise(resolve=>setTimeout(resolve,20));}
  if(!await value('pendingTestRequests.length===1'))throw new Error('Native file-read request did not start');
  await value(`(async()=>{await handleMineFile(fileB);pendingTestRequests[0].resolve(new Response(JSON.stringify(${JSON.stringify(fixture.rows.find(row=>row.slot==='upload').result.detection)}),{status:200,headers:{'Content-Type':'application/json'}}));await oldUploadTask;})()`);
  const race=await value('({oldAborted:pendingTestRequests[0].options.signal.aborted,correctFile:mineFileObj===fileB,noStaleGeometry:lastGeomStats===null,noStaleDetection:lastDetectedFile===null})');
  if(!Object.values(race).every(Boolean))throw new Error('Native browser old-input race failed: '+JSON.stringify(race));
  await value('window.fetch=originalTestFetch;');
  await value(`window.fixture=${JSON.stringify(fixture)};fixture.rows.forEach(row=>renderPipelineResult(row.result,row.slot));document.querySelectorAll('.scene').forEach(element=>element.style.display='none');document.getElementById('scene-compare').style.display='block';rp('compare','actions').scrollIntoView({block:'center',behavior:'instant'});`);
  const buttons=await value("[...rp('compare','actions').querySelectorAll('button')].map(element=>element.textContent)");
  if(!buttons[0].includes('HTML')||!buttons[1].includes('双期报告(JSON)'))throw new Error('Dedicated export actions missing');
  const exports=[];
  for(const [index,format] of ['html','json'].entries()){
    const expected=await value(`fetch('/api/report_compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:${JSON.stringify(compare.result.report_id)},format:${JSON.stringify(format)}})}).then(response=>response.json())`);
    if(expected.error)throw new Error(JSON.stringify(expected));
    await value(`rp('compare','actions').querySelectorAll('button')[${index}].click()`);
    const path=out+'/'+expected.filename;const bytes=await awaitFile(path);
    if(!bytes.equals(Buffer.from(expected.content_b64,'base64')))throw new Error('Browser download differs from the cached export');
    if(format==='json'){
      const packet=JSON.parse(bytes.toString('utf8'));
      if(packet.report_id!==compare.result.report_id||packet.review_handoff.trace.request_id!==compare.result.detection.request_id)throw new Error('Request identity changed');
      assert.deepEqual(packet.objective_facts,compare.result.objective_facts,'Measurement facts changed');
      if(packet.review_handoff.project_field_trial_or_expert_endorsement||packet.review_handoff.accepted_human_review_decisions.length)throw new Error('False review claim');
      if(!packet.review_handoff.review_items.some(row=>row.source_epoch==='t1')||!packet.review_handoff.review_items.some(row=>row.source_epoch==='t2'))throw new Error('Both epoch handoff regions required');
    }else{
      if(!bytes.toString('utf8').includes('矿安天眼 · 双期影像复核报告'))throw new Error('Chinese HTML corrupted');
    }
    exports.push({format,path,exact_cached_bytes:true,report_id:compare.result.report_id});
  }
  if((await value('testAlerts')).length)throw new Error(JSON.stringify(await value('testAlerts')));
  const uiShot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+'/comparison_actions.png',Buffer.from(uiShot.data,'base64'));
  await command('Page.navigate',{url:pathToFileURL(exports.find(row=>row.format==='html').path).href});
  for(let i=0;i<50;i++){if(await value("document.title==='矿安天眼 · 双期影像复核报告'"))break;await new Promise(resolve=>setTimeout(resolve,100));}
  const presentation=await value("({title:document.title,scrollWidth:document.documentElement.scrollWidth,innerWidth:window.innerWidth,body:document.body.textContent})");
  if(presentation.scrollWidth>presentation.innerWidth||!presentation.body.includes('同次变化量测')||!presentation.body.includes('双期影像与变化'))throw new Error('Standalone report layout failed');
  const reportShot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+'/comparison_report.png',Buffer.from(reportShot.data,'base64'));
  await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  const mobileLayout=await value("({scrollWidth:document.documentElement.scrollWidth,innerWidth:window.innerWidth})");
  if(mobileLayout.scrollWidth>mobileLayout.innerWidth)throw new Error('Mobile report overflow');
  await command('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await command('Emulation.setEmulatedMedia',{media:'print'});
  const printLayout=await value("({scrollWidth:document.documentElement.scrollWidth,innerWidth:window.innerWidth,appendixDisplay:getComputedStyle(document.querySelector('details')).display})");
  if(printLayout.scrollWidth>printLayout.innerWidth||printLayout.appendixDisplay!=='none')throw new Error('Print CSS layout failed');
  const printShot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+'/comparison_print_layout.png',Buffer.from(printShot.data,'base64'));
  let nativePDF=false,pdfLimit=null;
  try{
    const pdf=await command('Page.printToPDF',{printBackground:true,paperWidth:8.27,paperHeight:11.69});
    await fs.writeFile(out+'/print_smoke.pdf',Buffer.from(pdf.data,'base64'));nativePDF=true;
  }catch(error){pdfLimit=error.message;}
  await fs.writeFile(out+'/result.json',JSON.stringify({passed:true,real_browser:'Microsoft Edge headless',exports,
    print_CSS_layout_verified:true,native_PDF_export_verified:nativePDF,native_PDF_limit:pdfLimit,
    native_file_read_and_old_response_race_verified:race,
    no_horizontal_overflow:true,mobile_layout_verified:true,input_fixture:'previous observed LasCruces development detector result; images absent in archived fixture',
    scope:'engineering export acceptance, not new model evaluation or user/field trial',new_model_predictions:0,new_cloud_calls:0},null,2));
  console.log('PASS actual UI HTML/JSON downloads, exact cached bytes, both epoch handoff, desktop/mobile layout and print CSS; native PDF '+(nativePDF?'verified':'unverified'));
}finally{
  try{await fetch(base+'/api/local_handoff_smoke_shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch{}
  ws.close();
}
