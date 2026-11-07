import fs from 'node:fs/promises';
const output=process.env.KAT_BROWSER_SMOKE_DIR || 'D:/GeoMMAgent/cache/raster_browser_smoke';
const base=process.env.KAT_RASTER_SMOKE_URL || 'http://localhost:5100';
const debug=process.env.KAT_RASTER_DEBUG_URL || 'http://127.0.0.1:9222';
await fs.mkdir(output,{recursive:true});
let target;
for(let i=0;i<20;i++){
 try{const list=await(await fetch(debug+'/json')).json();target=list.find(p=>p.type==='page');if(target)break;}catch{}
 await new Promise(r=>setTimeout(r,1000));
}
if(!target)throw new Error('Headless inspection browser not ready');
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
let id=0;const waiting=new Map();
ws.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const job=waiting.get(message.id);waiting.delete(message.id);message.error?job.reject(new Error(JSON.stringify(message.error))):job.resolve(message.result);}});
function command(method,params={}){return new Promise((resolve,reject)=>{const key=++id;waiting.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});}
async function value(expression){const r=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
await command('Page.enable');await command('Runtime.enable');await command('DOM.enable');
await command('Emulation.setDeviceMetricsOverride',{width:1400,height:1000,deviceScaleFactor:1,mobile:false});
await command('Page.navigate',{url:base+'/'});
for(let i=0;i<15;i++){if(await value("Boolean(document.getElementById('mineFile'))"))break;await new Promise(r=>setTimeout(r,500));}
const doc=await command('DOM.getDocument');const input=await command('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#mineFile'});
await command('DOM.setFileInputFiles',{nodeId:input.nodeId,files:['D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif']});
let state;
for(let i=0;i<25;i++){
 state=await value("({gsd:document.getElementById('gsdInput').value,status:document.getElementById('gsdStatus').textContent,image:document.getElementById('mineThumb').src,roiVisible:document.getElementById('roiWrap').style.display,meta:lastMeta})");
 if(state.gsd==='10')break;await new Promise(r=>setTimeout(r,800));
}
if(state.gsd!=='10'||!state.image.startsWith('data:image/png;base64,')||state.meta.georeference.crs!=='EPSG:32629')throw new Error('Real TIFF preview failed: '+JSON.stringify(state));
await value('checkHealth()');
state.screening_title=await value('document.title');
state.desktop_workbench_style_leaked=await value("Boolean(document.getElementById('katWorkbench')||document.body.classList.contains('kat-workbench'))");
if(state.desktop_workbench_style_leaked)throw new Error('Desktop layout leaked into the original browser frontend');
const originalPage=await fs.readFile(new URL('../矿安天眼_前端演示(后端版).html',import.meta.url),'utf8');
const originalTitle=originalPage.match(/<title>([\s\S]*?)<\/title>/i)?.[1].trim();
if(state.screening_title!==originalTitle)throw new Error('Browser page title differs from the original project frontend');
state.connection=await value("document.getElementById('statusText').textContent");
if(state.connection.includes('检测器离线')||!/(检测器就绪|全部就绪)/.test(state.connection))throw new Error('Reachable detector service incorrectly offline');
if(process.argv.includes('--detect')){
 await value('runUploadDetect()');
 const detected=await value("({stats:lastGeomStats,meta:lastGeomMeta,badge:document.getElementById('mineBadge').textContent})");
 if(!detected.stats||detected.meta.meters_per_pixel!==10||detected.meta.georeference.crs!=='EPSG:32629')throw new Error('Real browser original-file detection failed: '+JSON.stringify(detected));
 await fs.writeFile(output+'/real_detection.json',JSON.stringify({...detected,role:'observed_development_interface_smoke_not_accuracy_evidence'},null,2));
 state.no_detection_prediction=false;
 const label=await value("document.getElementById('scene-mine').textContent");
 if(label.includes('蓝色-低风险'))throw new Error('Negative screening shown as certified low risk');
}
await value("document.querySelectorAll('.scene').forEach(el=>el.style.display='none');document.getElementById('mineFile').closest('.scene').style.display='block'");
await value("document.getElementById('scene-mine').scrollIntoView({block:'start',behavior:'instant'})");
await new Promise(r=>setTimeout(r,400));
const screenshot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
await fs.writeFile(output+'/preview.png',Buffer.from(screenshot.data,'base64'));
delete state.image;state.no_detection_prediction ??= true;state.real_browser='Microsoft Edge headless';
await fs.writeFile(output+'/result.json',JSON.stringify(state,null,2));
console.log('Real browser raw TIFF preview/GSD/georeference passed');ws.close();
