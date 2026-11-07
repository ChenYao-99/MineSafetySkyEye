import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const out='D:/GeoMMAgent/cache/case_delivery_20261008_v2';
const target=await(await fetch('http://127.0.0.1:9228/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
let id=0;const pending=new Map(),network=[];
ws.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.method==='Network.requestWillBeSent')network.push(value.params.request.url);const job=pending.get(value.id);if(job){pending.delete(value.id);clearTimeout(job.timer);value.error?job.reject(Error(JSON.stringify(value.error))):job.resolve(value.result)}});
const cmd=(method,params={})=>new Promise((resolve,reject)=>{const key=++id,timer=setTimeout(()=>{pending.delete(key);reject(Error('Browser command timeout'))},15000);pending.set(key,{resolve,reject,timer});ws.send(JSON.stringify({id:key,method,params}))});
async function value(expression){const result=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function until(expression){for(let i=0;i<100;i++){if(await value(expression))return;await new Promise(resolve=>setTimeout(resolve,80))}throw Error('Offline page not ready')}
try{
  await cmd('Page.enable');await cmd('Network.enable');await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const rows=[];
  for(const name of ['single','paired']){
    await cmd('Page.navigate',{url:'file:///'+out+'/'+name+'/review.html'});
    await until("Boolean(document.getElementById('raw')) && document.querySelectorAll('.marker').length===1");
    assert.ok(await value("document.body.textContent.includes('不是事故边界')"));
    assert.ok(await value("document.body.textContent.includes('10.45%') && document.body.textContent.includes('不是本页案例的精度')"));
    await value("document.querySelector('[data-view=prediction]').click()");
    assert.equal(await value("document.getElementById('prediction').style.display"),'');
    await value("document.getElementById('showMarkers').click()");
    assert.equal(await value("document.getElementById('markers').style.display"),'none');
    await value("document.querySelector('[data-view=raw]').click();document.getElementById('showMarkers').click()");
    assert.equal(await value("document.getElementById('prediction').style.display"),'none');
    const shot=await cmd('Page.captureScreenshot',{format:'png'});await fs.writeFile(out+'/'+name+'/offline_browser.png',Buffer.from(shot.data,'base64'));
    rows.push({kind:name,raw_prediction_and_manual_layer_controls:true,failed_independent_case_scope_visible:true});
  }
  assert.equal(network.filter(url=>/^https?:/.test(url)).length,0);
  await fs.writeFile(out+'/offline_browser_verification.json',JSON.stringify({passed:true,rows,external_network_requests:0,
    opened_from_local_files:true,requires_backend_GPU_or_cloud:false,geological_accuracy_or_human_trial:false},null,2));
  console.log('PASS both offline review files;layer controls;no external network;not expert feedback');
}finally{ws.close()}
