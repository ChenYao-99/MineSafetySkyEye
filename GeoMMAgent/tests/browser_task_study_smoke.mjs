import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const out='D:/GeoMMAgent/cache/high_value_sprint_20261008/browser';await fs.mkdir(out,{recursive:true});
const target=await(await fetch('http://127.0.0.1:9227/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;const pending=new Map();
ws.addEventListener('message',event=>{const value=JSON.parse(event.data),job=pending.get(value.id);if(job){pending.delete(value.id);clearTimeout(job.timer);value.error?job.reject(Error(JSON.stringify(value.error))):job.resolve(value.result)}});
ws.addEventListener('close',()=>{for(const job of pending.values()){clearTimeout(job.timer);job.reject(Error('Browser closed'))}pending.clear()});
const cmd=(method,params={})=>new Promise((resolve,reject)=>{const key=++id,timer=setTimeout(()=>{pending.delete(key);reject(Error('Browser command timeout '+method))},30000);pending.set(key,{resolve,reject,timer});ws.send(JSON.stringify({id:key,method,params}))});
async function value(expression){const result=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function until(expression){for(let i=0;i<100;i++){if(await value(expression))return;await new Promise(resolve=>setTimeout(resolve,80))}throw Error('UI not ready: '+expression)}
try{
  await cmd('Page.enable');await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cmd('Page.navigate',{url:'http://127.0.0.1:5198/?mode=automated_test'});await until("typeof api==='function'");
  await value("byId('role').value='内部浏览器自动化验收，不是用户研究';byId('start').click()");
  await until('current && current.number===1 && Boolean(picture)');
  const token=await value('token');
  for(let trial=1;trial<=12;trial++){
    await until(`current && current.number===${trial} && Boolean(picture) && !byId('trial').hidden`);
    assert.equal(await value('current.participant_kind'),'automated_test');
    await value("for(const key of fieldNames)byId(key).value=String(current.evidence[key]);byId('note').value='内部自动化功能验收；定位与计时不代表真实用户或专家意见';byId('image').scrollIntoView({block:'center',behavior:'instant'})");
    const r=await value("(()=>{const r=byId('image').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()");
    const a={x:r.x+r.w*.12,y:r.y+r.h*.15},b={x:r.x+r.w*.32,y:r.y+r.h*.30};
    await cmd('Input.dispatchMouseEvent',{type:'mousePressed',...a,button:'left',buttons:1,clickCount:1});
    await cmd('Input.dispatchMouseEvent',{type:'mouseMoved',...b,button:'left',buttons:1});
    await cmd('Input.dispatchMouseEvent',{type:'mouseReleased',...b,button:'left',buttons:0,clickCount:1});
    assert.ok(await value('box && box[2]>box[0] && box[3]>box[1]'));
    await value("byId('save').click()");await until("!byId('verify').hidden && !byId('save').disabled");
    if(trial===1){const shot=await cmd('Page.captureScreenshot',{format:'png'});await fs.writeFile(out+'/study_saved_record.png',Buffer.from(shot.data,'base64'));}
    await value("byId('confirm').click()");
  }
  await until("!byId('complete').hidden");
  const response=await fetch('http://127.0.0.1:5198/api/study/sessions/'+token+'/result');const summary=await response.json();
  assert.equal(summary.status,'complete');assert.equal(summary.participant_kind,'automated_test');assert.equal(summary.human_self_reported_observations,0);
  for(const arm of Object.values(summary.arms)){assert.equal(arm.tasks_completed,6);assert.equal(arm.incorrect_or_missing_fields,0)}
  await fs.writeFile(out+'/verification.json',JSON.stringify({passed:true,actual_browser_trials_completed:12,
    canvas_pointer_input_and_same_record_save_reopen_verified:true,participant_kind:'automated_test',human_observations:0,
    inference_or_cloud_calls:0,token,summary},null,2));
  console.log('PASS actual browser study:12 trials,both arms,source-bound locations,save/reopen;human evidence0');
}finally{ws.close()}
