import fs from 'node:fs/promises';
const out='D:/GeoMMAgent/cache/boundary_review_smoke';await fs.mkdir(out,{recursive:true});
const targets=await(await fetch('http://127.0.0.1:9222/json')).json();const target=targets.find(p=>p.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true})});
let id=0;const waiting=new Map();ws.addEventListener('message',e=>{const r=JSON.parse(e.data),p=waiting.get(r.id);if(p){waiting.delete(r.id);r.error?p.reject(new Error(JSON.stringify(r.error))):p.resolve(r.result)}});
function cmd(method,params={}){return new Promise((resolve,reject)=>{const k=++id;waiting.set(k,{resolve,reject});ws.send(JSON.stringify({id:k,method,params}))})}
async function value(expression){const r=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await cmd('Page.enable');await cmd('Runtime.enable');await cmd('Emulation.setDeviceMetricsOverride',{width:1400,height:1050,deviceScaleFactor:1,mobile:false});
await cmd('Page.navigate',{url:'file:///D:/GeoMMAgent/%E7%AB%9E%E8%B5%9B%E6%9D%90%E6%96%99/%E7%8B%AC%E7%AB%8B%E8%BE%B9%E7%95%8C%E5%A4%8D%E6%A0%B8%E5%B7%A5%E5%85%B7.html'});
for(let i=0;i<30;i++){if(await value("typeof CASES!=='undefined' && Boolean(bitmap)"))break;await new Promise(r=>setTimeout(r,300))}
const meta=await value("({ids:CASES.map(c=>c.id),shape:[canvas.height,canvas.width],source:'original_image_only'})");
if(meta.ids.join(',')!=='Amyntaio,LasCruces,Hpakant'||meta.shape[1]!==947)throw new Error('Actual independent source review not loaded');
const screenshot=await cmd('Page.captureScreenshot',{format:'png'});await fs.writeFile(out+'/preview.png',Buffer.from(screenshot.data,'base64'));
// Exercise the actual export callback while suppressing browser downloads.
await value("window.testBlob=null;const originalURL=URL.createObjectURL;URL.createObjectURL=b=>{window.testBlob=b;return originalURL(b)};HTMLAnchorElement.prototype.click=function(){};document.getElementById('reviewer').value='AUTOMATED_SMOKE_NOT_AN_EXPERT';document.getElementById('notes').value='Internal UI test only; not real review feedback.';state.Amyntaio.draft=[[10,10],[30,10],[30,30]];document.getElementById('finish').click();document.getElementById('export').click()");
const record=await value("window.testBlob.text().then(JSON.parse)");
if(record.model_outputs_shown_by_tool!==false||record.expert_identity_verified!==false||record.cases[0].polygons.length!==1)throw new Error('Review export truthfulness failed');
const c=record.cases[0],p=c.polygons[0],t=c.transform;
if(Math.abs(p.vertices_projected[0][0]-(t[0]*10+t[1]*10+t[2]))>.001||p.vertices_pixel.length!==3)throw new Error('Pixel-to-projected mapping failed');
await fs.writeFile(out+'/synthetic_export_NOT_A_REVIEW.json',JSON.stringify({...record,automated_smoke_not_real_review:true},null,2));
await fs.writeFile(out+'/result.json',JSON.stringify({...meta,passed:true,expert_reviews_created:0,actual_browser:true},null,2));
console.log('Offline original-image drawing/export passed; external reviews created0');ws.close();
