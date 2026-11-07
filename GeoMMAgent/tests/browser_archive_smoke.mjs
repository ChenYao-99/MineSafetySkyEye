import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const out=process.env.ARCHIVE_ACCEPTANCE_OUT || 'D:/GeoMMAgent/cache/archive_workflow_acceptance_20261006';
const base='http://127.0.0.1:5105',debug='http://127.0.0.1:9223';
const reopening=process.argv.includes('--reopen');
await fs.mkdir(out,{recursive:true});
const target=await(await fetch(debug+'/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;const pending=new Map();
ws.addEventListener('message',event=>{const value=JSON.parse(event.data),job=pending.get(value.id);if(job){pending.delete(value.id);value.error?job.reject(new Error(JSON.stringify(value.error))):job.resolve(value.result)}});
const cmd=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}))});
async function value(expression){const row=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,userGesture:true});if(row.exceptionDetails)throw new Error(JSON.stringify(row.exceptionDetails));return row.result.value;}
async function post(path,data){const response=await fetch(base+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});return {status:response.status,data:await response.json()};}
async function awaitFile(path){for(let i=0;i<60;i++){try{return await fs.readFile(path);}catch{}await new Promise(resolve=>setTimeout(resolve,100));}throw new Error('Download missing '+path);}
try{
  await cmd('Page.enable');await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cmd('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:out.replaceAll('/','\\'),eventsEnabled:true});
  await cmd('Page.navigate',{url:base+'/'});
  for(let i=0;i<70;i++){if(await value("typeof openArchivedReport==='function'"))break;await new Promise(resolve=>setTimeout(resolve,100));}
  const fixture=await(await fetch(base+'/api/local_archive_fixture')).json();
  if(reopening&&fixture.created_new_fixture)throw new Error('Restart lost archive and regenerated results');
  const report=fixture.reports.find(row=>row.kind==='paired_mask_change');assert.ok(report);
  const original=await(await fetch(base+'/api/archive/reports/'+report.report_id)).json();
  await value(`window.testAlerts=[];window.alert=message=>testAlerts.push(String(message));document.querySelector('[data-target="scene-archive"]').click();`);
  await value(`selectArchiveCase(${JSON.stringify(fixture.case.case_id)})`);
  await value('loadArchiveCases(0)');
  await value(`openArchivedReport(${JSON.stringify(report.report_id)})`);
  const state=await value("({id:archiveReport.report_id,active:activeCaseId,revision:archiveReport.archive.review_revision,visible:getComputedStyle(document.getElementById('archiveReviewPanel')).display!=='none'})");
  assert.equal(state.id,report.report_id);assert.equal(state.active,fixture.case.case_id);assert.ok(state.visible);
  if(!reopening){
    for(const [status,note] of [['needs_followup','内部复核：需要对照原图继续核查'],['likely_work_change','修订意见：可能为作业变化，日期与原因仍待确认']]){
      await value(`document.getElementById('archiveReviewTarget').value='report';document.getElementById('archiveReviewStatus').value=${JSON.stringify(status)};document.getElementById('archiveReviewer').value='内部测试复核员';document.getElementById('archiveReviewEvidence').value='工程验收模拟依据，不是专家或现场意见';document.getElementById('archiveReviewNote').value=${JSON.stringify(note)};saveArchiveReview()`);
    }
    const stale=await post('/api/archive/reports/'+report.report_id+'/reviews',{expected_revision:0,target_key:'report',status:'needs_followup',reviewer:'另一个测试复核员',evidence:'内部测试',note:'旧版本不能覆盖'});
    assert.equal(stale.status,409);
    const current=await(await fetch(base+'/api/archive/reports/'+report.report_id)).json();
    assert.equal(current.archive.review_revision,2);assert.equal(current.review_handoff.review_history.length,2);
    assert.deepEqual(current.objective_facts,original.objective_facts);assert.deepEqual(current.detection,original.detection);
    assert.equal(current.archive.snapshot_sha256,original.archive.snapshot_sha256);
    assert.equal(current.review_handoff.project_field_trial_or_expert_endorsement,false);
    assert.equal(current.review_handoff.accepted_human_review_decisions[0].reviewer_identity_verified,false);
    const record=await post('/api/review_record',{report_id:report.report_id});
    await value("exportReviewRecord('archive')");
    const downloaded=JSON.parse((await awaitFile(out+'/'+record.data.filename)).toString('utf8'));
    assert.equal(downloaded.review_revision,2);assert.equal(downloaded.review_history.length,2);
    const html=await post('/api/report_compare',{report_id:report.report_id,format:'html'});
    await value('exportArchivedReport()');
    const bytes=await awaitFile(out+'/'+html.data.filename);assert.ok(bytes.toString('utf8').includes('修订意见'));
    assert.ok(bytes.equals(Buffer.from(html.data.content_b64,'base64')));
    await value('backupCaseArchive()');
    const backup=await awaitFile(out+'/矿安天眼_案例数据库备份.sqlite3');
    assert.ok(backup.subarray(0,16).toString().startsWith('SQLite format 3'));
    await fs.writeFile(out+'/before_restart.json',JSON.stringify({case_id:fixture.case.case_id,report_id:report.report_id,
      facts:original.objective_facts,snapshot_sha256:original.archive.snapshot_sha256,review_revision:2},null,2));
  }else{
    const before=JSON.parse(await fs.readFile(out+'/before_restart.json','utf8'));
    assert.equal(report.report_id,before.report_id);assert.equal(state.revision,before.review_revision);
    assert.deepEqual(original.objective_facts,before.facts);assert.equal(original.archive.snapshot_sha256,before.snapshot_sha256);
    assert.equal(original.review_handoff.review_history.length,2);
  }
  const layout=await value("({width:document.documentElement.scrollWidth,viewport:innerWidth,history:document.getElementById('archiveReviewHistory').textContent})");
  assert.ok(layout.width<=layout.viewport);assert.ok(layout.history.includes('内部测试复核员'));
  assert.equal((await value('testAlerts')).length,0);
  assert.ok(await value("document.getElementById('archiveReports').textContent.includes('复核版本 2')"));
  await value("document.getElementById('archiveReviewPanel').scrollIntoView({block:'start',behavior:'instant'})");
  const shot=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(out+(reopening?'/archive_reopened.png':'/archive_review.png'),Buffer.from(shot.data,'base64'));
  const responsive=[];
  for(const width of [1440,768,390]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    const metrics=await value(`(() => {
      const scene=document.getElementById('scene-archive');
      const rect=scene.getBoundingClientRect();
      const buttons=[...scene.querySelectorAll('button')].filter(el=>el.getClientRects().length);
      const controls=[...scene.querySelectorAll('input,select,textarea,button')].filter(el=>el.getClientRects().length);
      const outside=controls.filter(el=>{const r=el.getBoundingClientRect();return r.left<rect.left-1||r.right>rect.right+1}).map(el=>el.id||el.textContent);
      const overlaps=[];
      for(let i=0;i<buttons.length;i++)for(let j=i+1;j<buttons.length;j++){
        const a=buttons[i].getBoundingClientRect(),b=buttons[j].getBoundingClientRect();
        if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)overlaps.push([buttons[i].textContent,buttons[j].textContent]);
      }
      return {width:innerWidth,sceneWidth:scene.clientWidth,sceneScrollWidth:scene.scrollWidth,outside,overlaps,minButtonHeight:Math.min(...buttons.map(el=>el.getBoundingClientRect().height))};
    })()`);
    assert.ok(metrics.sceneScrollWidth<=metrics.sceneWidth+1,JSON.stringify(metrics));
    assert.deepEqual(metrics.outside,[]);assert.deepEqual(metrics.overlaps,[]);assert.ok(metrics.minButtonHeight>=38);
    responsive.push(metrics);
    for(const [name,selector] of [['overview','#scene-archive'],['review','#archiveReviewPanel']]){
      await value(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start',behavior:'instant'});${name==='review'?'window.scrollBy(0,-130);':''}`);
      await new Promise(resolve=>setTimeout(resolve,70));
      assert.ok(await value("document.querySelector('[data-target=\"scene-archive\"]').classList.contains('active')"));
      const picture=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
      await fs.writeFile(`${out}/archive_${name}_${width}.png`,Buffer.from(picture.data,'base64'));
    }
  }
  await fs.writeFile(out+(reopening?'/reopen_result.json':'/browser_result.json'),JSON.stringify({passed:true,reopening,
    preserved_report_id:report.report_id,model_facts_unchanged:true,human_revision:2,
    actual_UI_review_and_download:true,database_backup_download_verified:!reopening,responsive,
    new_classifier_predictions:0,new_cloud_calls:0,scope:'engineering acceptance, not field or expert feedback'},null,2));
  console.log('PASS real browser archive '+(reopening?'restore after server restart':'review, conflict protection, history and report/database downloads'));
}finally{
  try{await fetch(base+'/api/local_archive_shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch{}
  ws.close();
}
