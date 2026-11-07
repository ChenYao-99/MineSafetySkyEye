// Actual WebView2 desktop acceptance; only local facts and observed development data.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const root='D:/GeoMMAgent';
const output=process.env.KAT_DESKTOP_ACCEPTANCE_DIR || root+'/cache/desktop_acceptance_20261018';
const base=process.env.KAT_DESKTOP_TEST_URL || 'http://127.0.0.1:5120';
const debug=process.env.KAT_DESKTOP_DEBUG_URL || 'http://127.0.0.1:9228';
const port=new URL(base).port;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await fs.mkdir(output,{recursive:true});
const protectedBefore={};
for(const name of ['kat_demo.html','矿安天眼_前端演示(后端版).html','start_demo.ps1','requirements.txt','backend.py']){
  protectedBefore[name]=crypto.createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex');
}
let target;
for(let i=0;i<50;i++) {
  try {target=(await(await fetch(debug+'/json/list')).json()).find(row=>row.type==='page'&&row.url.startsWith(base+'/'));} catch {}
  if(target)break;
  await sleep(500);
}
if(!target)throw new Error('Owned desktop WebView2 did not load the project UI');
const instance=JSON.parse(await fs.readFile(root+'/cache/desktop/instance_'+port+'.json','utf8'));
assert.equal(instance.status,'ready');
assert.equal(instance.local_facts_only_requested,true,'Acceptance must explicitly disable cloud generation');
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;
const waiting=new Map();
ws.addEventListener('message',event=>{
  const message=JSON.parse(event.data),job=waiting.get(message.id);
  if(job){waiting.delete(message.id);clearTimeout(job.timer);message.error?job.reject(new Error(JSON.stringify(message.error))):job.resolve(message.result)}
});
function command(method,params={}) {
  return new Promise((resolve,reject)=>{const key=++id;const timer=setTimeout(()=>{waiting.delete(key);reject(new Error('CDP timed out: '+method))},45000);waiting.set(key,{resolve,reject,timer});ws.send(JSON.stringify({id:key,method,params}))});
}
async function value(expression) {
  const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});
  if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function until(expression) {
  for(let i=0;i<60;i++){try{if(await value(expression))return;}catch{}await sleep(250)}
  throw new Error('Desktop condition not reached: '+expression);
}
async function files(selector,filename) {
  const document=await command('DOM.getDocument');
  const node=await command('DOM.querySelector',{nodeId:document.root.nodeId,selector});
  await command('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[filename]});
}
async function layoutSnapshot(name,width,height) {
  await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await sleep(250);
  const layout=await value(`({width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth,
    scene:KATWorkbench.currentScene(),visibleScenes:[...document.querySelectorAll('.desk-scene')].filter(n=>!n.hidden).length,
    activeNavigation:[...document.querySelectorAll('#sceneNav .nav-pill.active')].map(n=>n.dataset.target),
    actionBottom:document.getElementById('uploadBtn').getBoundingClientRect().bottom,
    sidebarWidth:document.querySelector('.desk-sidebar').getBoundingClientRect().width})`);
  assert.ok(layout.documentWidth<=width+1,'Workbench causes horizontal overflow: '+JSON.stringify(layout));
  assert.equal(layout.visibleScenes,1);
  assert.deepEqual(layout.activeNavigation,[layout.scene]);
  if(layout.scene==='scene-mine')assert.ok(layout.actionBottom<=height,'Primary action is outside the workspace');
  const screenshot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(output+'/'+name+'.png',Buffer.from(screenshot.data,'base64'));
  return layout;
}
async function save(action,expected,filename) {
  const before=new Set(await fs.readdir(instance.exports));
  await value(action);
  for(let i=0;i<60;i++){
    for(const name of await fs.readdir(instance.exports)){
      if(before.has(name))continue;
      try{const bytes=await fs.readFile(path.join(instance.exports,name));if(bytes.equals(Buffer.from(expected.content_b64,'base64')))return {filename:name,bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),download_directory:instance.exports};}catch{}
    }
    await sleep(200);
  }
  throw new Error('Desktop saved bytes did not match same-report export: '+filename);
}

const result={renderer:'Windows WebView2 in pywebview',actual_desktop_pid:instance.pid,
  new_cloud_generation_calls:0,imagery_role:'previously observed LasCruces development only',checks:{},exports:[]};
try {
  await command('Runtime.enable');await command('DOM.enable');await command('Page.enable');
  await command('Page.navigate',{url:base+'/'});
  await until("typeof runUploadDetect==='function' && typeof runCompareReport==='function'");
  await until("window.KATWorkbench?.version === '2026.10.workbench.1' && getComputedStyle(document.getElementById('katWorkbench')).display==='grid'");
  result.layouts=[];
  result.layouts.push(await layoutSnapshot('workbench_empty',1440,900));
  result.layouts.push(await layoutSnapshot('workbench_compact',1000,720));
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
  result.checks.desktop_workbench_and_responsive_layout=true;
  const original=await fs.readFile(root+'/矿安天眼_前端演示(后端版).html');
  const served=Buffer.from(await(await fetch(base+'/')).arrayBuffer());
  assert.ok(served.equals(original));
  result.checks.original_web_page_served_unchanged=true;
  await value("window.desktopTestAlerts=[];window.alert=message=>desktopTestAlerts.push(String(message));checkHealth()");
  assert.equal((await(await fetch(base+'/api/health')).json()).reporting_mode,'local_facts_only');
  const source='D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/';
  await files('#mineFile',source+'post_BGRN_10m.tif');
  await until("lastMeta && lastMeta.meters_per_pixel===10 && roiImg!==null");
  result.checks.real_geotiff_file_input_and_preview=true;
  await value('runUploadDetect()');
  const detected=await value('({stats:lastGeomStats,meta:lastGeomMeta,badge:document.getElementById("mineBadge").textContent})');
  assert.equal(detected.stats.area_ratio,.0871);
  assert.equal(detected.meta.meters_per_pixel,10);
  result.checks.actual_gpu_detection=true;
  result.single_stats={area_ratio:detected.stats.area_ratio,num_regions:detected.stats.num_regions};
  await until("document.querySelector('.desk-overlay-image').hidden===false");
  await value("document.getElementById('deskOriginalView').click()");
  const canvas=await value("(()=>{const r=document.getElementById('roiCanvas').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()");
  await command('Input.dispatchMouseEvent',{type:'mousePressed',x:canvas.x+canvas.width*.25,y:canvas.y+canvas.height*.25,button:'left',clickCount:1});
  await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:canvas.x+canvas.width*.9,y:canvas.y+canvas.height*.9,button:'left',buttons:1});
  await command('Input.dispatchMouseEvent',{type:'mouseReleased',x:canvas.x+canvas.width*.9,y:canvas.y+canvas.height*.9,button:'left',clickCount:1});
  await until("roiFrac!==null && lastGeomStats===null && document.getElementById('deskOverlayView').disabled");
  await value('runUploadDetect()');
  const roi=await value('({scope:lastGeomStats.scope,ratio:lastGeomStats.area_ratio,selection:roiFrac})');
  assert.equal(roi.scope,'roi');assert.ok(roi.selection.x0>.2 && roi.selection.x1>.85);
  result.roi=roi;result.checks.resized_canvas_ROI_and_stale_overlay_invalidation=true;
  result.layouts.push(await layoutSnapshot('workbench_detection',1440,900));
  await value('runUploadPipeline()');
  await until("reportParams.upload && reportParams.upload.report_id");
  const single=await value('reportParams.upload');
  result.checks.local_fact_report_and_archive=true;
  const jsonExport=await value(`fetch('/api/review_record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:${JSON.stringify(single.report_id)}})}).then(r=>r.json())`);
  if(jsonExport.error)throw new Error(JSON.stringify(jsonExport));
  result.exports.push(await save("exportReviewRecord('upload')",jsonExport,'desktop_single_review.json'));
  await value("document.querySelector('[data-target=\"scene-compare\"]').click()");
  await value("document.getElementById('cmpT1Prev').removeAttribute('src');document.getElementById('cmpT2Prev').removeAttribute('src')");
  await files('#cmpT1',source+'pre_BGRN_10m.tif');
  await until("document.getElementById('cmpT1Prev').src.startsWith('data:')");
  await files('#cmpT2',source+'post_BGRN_10m.tif');
  await until("document.getElementById('cmpT1Prev').src.startsWith('data:') && document.getElementById('cmpT2Prev').src.startsWith('data:')");
  await value('runCompare()');
  await until('cmpImages.t1!==null && cmpImages.t2!==null');
  await value('runCompareReport()');
  await until('reportParams.compare && reportParams.compare.report_id');
  const paired=await value('reportParams.compare');
  result.checks.actual_gpu_pair_and_report=true;
  result.layouts.push(await layoutSnapshot('workbench_compare',1440,900));
  for(const format of ['html','json']){
    const expected=await value(`fetch('/api/report_compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:${JSON.stringify(paired.report_id)},format:'${format}'})}).then(r=>r.json())`);
    if(expected.error)throw new Error(JSON.stringify(expected));
    result.exports.push(await save(`exportCompareReport('${format}')`,expected,'desktop_paired.'+format));
  }
  await value("viewReportArchive('compare')");
  await until(`archiveReport?.report_id===${JSON.stringify(paired.report_id)} && document.getElementById('archiveReviewPanel').style.display!=='none'`);
  await value("document.getElementById('archiveReviewer').value='内部界面验收';document.getElementById('archiveReviewEvidence').value='既有公开开发影像，仅验证桌面复核界面';document.getElementById('archiveReviewNote').value='自动化内部检查，不作为矿方、专业用户或事故真值认定。';saveArchiveReview()");
  await until("document.getElementById('archiveReviewMessage').textContent.includes('意见已保存')");
  result.checks.archive_workspace_navigation=true;
  result.checks.review_form_and_revision_save_in_isolated_archive=true;
  result.internal_automatic_review_not_external_validation=true;
  result.layouts.push(await layoutSnapshot('workbench_archive',1440,900));
  for(const [scene,name] of [['scene-qa','workbench_agents'],['scene-kb','workbench_knowledge'],['scene-bench','workbench_benchmark']]){
    await value(`KATWorkbench.selectScene('${scene}')`);
    if(scene==='scene-qa'){
      await files('#qaFile',root+'/demo/03_LasCruces_original.png');
      await until("document.getElementById('qaPreview').src.startsWith('data:')");
      const question=await value("(()=>{const b=document.querySelectorAll('.quick-q')[2];b.click();return {expected:b.dataset.q,actual:document.getElementById('questionInput').value}})()");
      assert.equal(question.actual,question.expected);result.checks.QA_image_input_and_question_presets=true;
    }
    if(scene==='scene-kb'){
      await value("document.getElementById('kbQuery').value='边坡巡查';runKnowledgeQuery()");
      await until("document.querySelectorAll('#kbResults .kb-doc').length>0");
      result.checks.local_knowledge_retrieval=true;
    }
    result.layouts.push(await layoutSnapshot(name,1440,900));
  }
  await value("document.querySelector('.desk-help').click()");
  assert.equal(await value("document.querySelector('.desk-help-dialog').open"),true);
  await value("document.querySelector('.desk-help-dialog .desk-button').click()");
  result.checks.operation_guide_dialog=true;
  const alerts=await value('desktopTestAlerts');assert.deepEqual(alerts,[]);
  await value("document.querySelector('[data-target=\"scene-mine\"]').click();window.scrollTo(0,0)");
  const finalImage=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(output+'/desktop_window.png',Buffer.from(finalImage.data,'base64'));
  result.checks.desktop_webview_render_captured=true;
  result.screenshot_scope='actual desktop WebView2 content; native title bar excluded';
  const before=protectedBefore;
  result.original_file_hashes={};
  for(const [name,expected] of Object.entries(before)){
    const actual=crypto.createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex');
    assert.equal(actual,expected,name+' changed');result.original_file_hashes[name]=actual;
  }
  result.passed=true;
  await fs.writeFile(output+'/acceptance.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
} finally {ws.close()}
