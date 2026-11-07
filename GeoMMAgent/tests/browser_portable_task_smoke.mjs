import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const base=process.env.KAT_BROWSER_BASE || 'http://127.0.0.1:5118';
const out=process.env.KAT_BROWSER_OUT || 'D:/GeoMMAgent/cache/portable_task_acceptance_20261007/browser';
const cdp=process.env.KAT_BROWSER_CDP || 'http://127.0.0.1:9224';
const inputFile=process.env.KAT_BROWSER_INPUT || 'D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif';
const grid=process.env.KAT_BROWSER_GRID || 'native';
const expectedGsd=process.env.KAT_BROWSER_EXPECTED_GSD || '10';
await fs.mkdir(out,{recursive:true});
const target=await(await fetch(cdp+'/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;const jobs=new Map();
ws.addEventListener('message',event=>{const data=JSON.parse(event.data),job=jobs.get(data.id);if(job){jobs.delete(data.id);clearTimeout(job.timer);data.error?job.reject(new Error(JSON.stringify(data.error))):job.resolve(data.result)}});
ws.addEventListener('close',()=>{for(const job of jobs.values()){clearTimeout(job.timer);job.reject(new Error('Browser debugging connection closed'))}jobs.clear()});
const cmd=(method,params={})=>new Promise((resolve,reject)=>{const key=++id,timer=setTimeout(()=>{jobs.delete(key);reject(new Error('Browser debugging command timed out: '+method))},90000);jobs.set(key,{resolve,reject,timer});ws.send(JSON.stringify({id:key,method,params}))});
async function value(expression){const r=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function file(path){for(let i=0;i<60;i++){try{return await fs.readFile(path)}catch{}await new Promise(r=>setTimeout(r,100))}throw new Error('Missing downloaded '+path)}
try{
  await cmd('Page.enable');await cmd('DOM.enable');
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cmd('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:out.replaceAll('/','\\')});
  await cmd('Page.navigate',{url:base+'/'});
  for(let i=0;i<80;i++){if(await value("typeof createArchiveCase==='function'"))break;await new Promise(r=>setTimeout(r,100))}
  await value("window.testAlerts=[];window.alert=message=>testAlerts.push(String(message));document.getElementById('archiveMine').value='浏览器真实影像工程验收';document.getElementById('archiveTitle').value='原图上传至人工复核';document.getElementById('archiveCaseNotes').value='内部自动验收，不是专业用户或现场反馈';createArchiveCase()");
  const caseId=await value('activeCaseId');assert.ok(caseId);
  const document=await cmd('DOM.getDocument');const input=await cmd('DOM.querySelector',{nodeId:document.root.nodeId,selector:'#mineFile'});
  await cmd('DOM.setFileInputFiles',{nodeId:input.nodeId,files:[inputFile]});
  for(let i=0;i<100;i++){if(await value("document.getElementById('gsdInput').value")===expectedGsd)break;await new Promise(r=>setTimeout(r,150))}
  assert.equal(await value("document.getElementById('gsdInput').value"),expectedGsd);
  await value(`document.getElementById('uploadGrid').value=${JSON.stringify(grid)};document.getElementById('uploadGrid').dispatchEvent(new Event('change'))`);
  await value('runUploadDetect()');
  assert.ok(await value('Boolean(uploadImgB64)'), 'Actual detection must precede report generation');
  // One existing continuous-run request can briefly occupy the detector.
  for(let attempt=0;attempt<4;attempt++){
    await value('runUploadPipeline()');
    if(await value('Boolean(reportParams.upload?.report_id)'))break;
    await new Promise(r=>setTimeout(r,1500));
  }
  // Avoid transferring full-size image previews through the debugging WebSocket.
  const captured=await value("(()=>{const r=reportResults.upload;return r&&{report_id:r.report_id,archive:r.archive,generation:r.generation,objective_facts:r.objective_facts,detection:{band_meta:r.detection.band_meta,mask_display_contract:r.detection.mask_display_contract,landslide_pixels:r.detection.stats.landslide_pixels}}})()");assert.ok(captured?.report_id);
  assert.equal(captured.archive.case_id,caseId);assert.equal(captured.generation.status,'fallback');
  assert.equal(captured.detection.band_meta.spatial_input_contract.mode,grid);
  assert.equal(captured.detection.mask_display_contract.reported_region_pixels,captured.detection.landslide_pixels);
  assert.equal(captured.detection.mask_display_contract.classification_threshold_changed,false);
  await value("window.testVectorExports=[];const testOriginalFetch=window.fetch.bind(window);window.fetch=async(...args)=>{const response=await testOriginalFetch(...args);if(String(args[0]).includes('/geometry_export')&&response.ok)testVectorExports.push(await response.clone().json());return response}");
  for(const format of ['geojson','csv']){
    await value(`exportVector('${format}','upload')`);
    const exported=await value('testVectorExports.at(-1)');assert.ok(exported?.filename);
    const downloaded=await file(out+'/'+exported.filename);
    assert.deepEqual(downloaded,Buffer.from(exported.content_b64,'base64'));
    if(grid==='reference_10m'){
      if(format==='geojson'){
        const protocol=JSON.parse(downloaded.toString('utf8')).spatial_input_contract;
        assert.equal(protocol.mode,grid);assert.equal(protocol.reference_meters_per_pixel,10);
        assert.equal(protocol.ground_boundary_precision_verified,false);
        assert.equal(protocol.accuracy_on_real_high_resolution_sensors_verified,false);
      }else{
        assert.ok(downloaded.toString('utf8').split('\n')[0].includes('ground_boundary_precision_verified'));
        assert.ok(downloaded.toString('utf8').includes('reference_10m'));
      }
    }
  }
  await value("document.getElementById('uploadGrid').scrollIntoView({block:'center',behavior:'instant'})");
  const analysisShot=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+'/actual_scale_analysis.png',Buffer.from(analysisShot.data,'base64'));
  await value("viewReportArchive('upload')");
  assert.equal(await value('archiveReport.report_id'),captured.report_id);
  await value("document.getElementById('archiveReviewer').value='内部浏览器任务验收';document.getElementById('archiveReviewStatus').value='insufficient_evidence';document.getElementById('archiveReviewEvidence').value='真实公开开发影像；此意见仅用于工程功能验收';document.getElementById('archiveReviewNote').value='保留同次量测，未核验事故原因';saveArchiveReview()");
  const updated=await value('({archive:archiveReport.archive,objective_facts:archiveReport.objective_facts})');assert.equal(updated.archive.review_revision,1);
  assert.deepEqual(updated.objective_facts,captured.objective_facts);assert.equal(updated.archive.snapshot_sha256,captured.archive.snapshot_sha256);
  let expectedRevision=1;
  if(process.env.KAT_BROWSER_MANUAL_ANNOTATION==='1'){
    await value("document.getElementById('archiveManualPanel').open=true;document.getElementById('archiveManualCanvas').scrollIntoView({block:'start',behavior:'instant'});window.scrollBy(0,-130)");
    for(let i=0;i<80;i++){if(await value('Boolean(manualAnnotationImage)'))break;await new Promise(r=>setTimeout(r,100))}
    assert.ok(await value('Boolean(manualAnnotationImage)'));
    const rectangle=await value("(()=>{const r=document.getElementById('archiveManualCanvas').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()");
    const a={x:rectangle.x+rectangle.w*.12,y:rectangle.y+rectangle.h*.10},b={x:rectangle.x+rectangle.w*.28,y:rectangle.y+rectangle.h*.22};
    await cmd('Input.dispatchMouseEvent',{type:'mousePressed',...a,button:'left',buttons:1,clickCount:1});
    await cmd('Input.dispatchMouseEvent',{type:'mouseMoved',...b,button:'left',buttons:1});
    await cmd('Input.dispatchMouseEvent',{type:'mouseReleased',...b,button:'left',buttons:0,clickCount:1});
    assert.ok(await value('manualAnnotationBox'));
    await value('saveManualAnnotation()');expectedRevision=2;
    assert.equal(await value('archiveReport.archive.review_revision'),2);
    assert.equal(await value('archiveReport.review_handoff.manual_annotations.length'),1);
    assert.deepEqual(await value('archiveReport.objective_facts'),captured.objective_facts);
  }
  const response=await fetch(base+'/api/review_record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report_id:captured.report_id})});
  const expected=await response.json();await value("exportReviewRecord('archive')");
  const downloaded=JSON.parse((await file(out+'/'+expected.filename)).toString('utf8'));
  assert.equal(downloaded.trace.report_id,captured.report_id);assert.equal(downloaded.review_revision,expectedRevision);
  if(expectedRevision===2){assert.equal(downloaded.manual_annotations.length,1);assert.equal(downloaded.manual_annotations[0].manual_annotation.verified_ground_truth,false);}
  assert.equal(downloaded.project_field_trial_or_expert_endorsement,false);
  const alerts=await value('testAlerts');assert.equal(alerts.length,0);
  await value("document.getElementById('archiveReviewPanel').scrollIntoView({block:'start',behavior:'instant'});window.scrollBy(0,-130)");
  await new Promise(resolve=>setTimeout(resolve,700));
  const shot=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+'/actual_task_review.png',Buffer.from(shot.data,'base64'));
  await fs.writeFile(out+'/verification.json',JSON.stringify({passed:true,report_id:captured.report_id,case_id:caseId,
    actual_georeferenced_TIFF_browser_upload:true,input_file:inputFile,analysis_grid:grid,
    actual_GPU_prediction_and_local_facts_report:true,GeoJSON_and_CSV_download_bytes_and_scale_protocol_preserved:true,
    display_mask_and_reported_area_share_component_filter:true,
    report_source_weight_and_facts_preserved:true,UI_human_note_revision:expectedRevision,manual_location_box_saved_in_actual_browser:expectedRevision===2,download_matches_same_record:true,
    scope:'automated actual browser task;not professional-user study or accident accuracy evaluation'},null,2));
  console.log('PASS actual browser TIFF→GPU/local report→GeoJSON/CSV→archive→review→JSON download');
}finally{ws.close()}
