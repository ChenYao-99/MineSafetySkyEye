// Execute the actual page with deliberately delayed and reordered responses.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('D:/GeoMMAgent/矿安天眼_前端演示(后端版).html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const escape=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function deferred(){let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j});return {promise,resolve,reject};}
function harness(){
  const elements={},calls=[],alerts=[],anchors=[];
  class Element{
    constructor(id){Object.assign(this,{id,style:{},value:'',files:[],children:[],handlers:{},disabled:false,_text:'',_html:''});
      const classes=new Set();this.classList={add:(...names)=>names.forEach(n=>classes.add(n)),remove:(...names)=>names.forEach(n=>classes.delete(n)),contains:n=>classes.has(n),toggle(){}};}
    get textContent(){return this._text;} set textContent(value){this._text=String(value);this._html=escape(value);}
    get innerHTML(){return this._html;} set innerHTML(value){this._html=value;this.children=[];}
    addEventListener(type,callback){(this.handlers[type]??=[]).push(callback);}
    fire(type,event={}){return Promise.all((this.handlers[type]||[]).map(callback=>callback({target:this,preventDefault(){},...event})));}
    click(){this.fire('click');} scrollIntoView(){} appendChild(child){this.children.push(child);}
    querySelectorAll(){return this.children;}
    getBoundingClientRect(){return {left:0,top:0,width:320,height:240};}
    getContext(){return {clearRect(){},drawImage(){},fillRect(){},strokeRect(){}};}
  }
  const el=id=>elements[id]??=new Element(id);
  class Reader{
    readAsDataURL(file){queueMicrotask(()=>file.readError?this.onerror?.():file.readAbort?this.onabort?.():this.onload?.({target:{result:'data:image/png;base64,'+(file.bytes||'SU1BR0U=')}}));}
    readAsText(file){queueMicrotask(()=>this.onload?.({target:{result:file.text||'{}'}}));}
  }
  const sandbox={console,AbortController,Blob,Uint8Array,FileReader:Reader,
    document:{getElementById:el,querySelectorAll:()=>[],createElement(tag){const item=new Element(tag);if(tag==='a')anchors.push(item);return item;}},
    window:{addEventListener(){}},Image:class{width=640;height=480;set src(value){this._src=value;this.onload?.();}get src(){return this._src;}},
    URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},alert:value=>alerts.push(String(value)),
    atob:value=>Buffer.from(value,'base64').toString('binary'),btoa:value=>Buffer.from(value).toString('base64'),
    setInterval(){},setTimeout(){},clearInterval(){},requestAnimationFrame(){},
    fetch:(url,options)=>{const pending=deferred();calls.push({url,options,body:JSON.parse(options.body),...pending});return pending.promise;}};
  vm.createContext(sandbox);vm.runInContext(source,sandbox);
  const run=code=>vm.runInContext(code,sandbox);
  const give=(index,body,status=200)=>calls[index].resolve({ok:status<400,status,json:async()=>body});
  return {el,run,sandbox,calls,alerts,anchors,give};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function detection(id){return {request_id:'request-'+id,overlay_b64:'OVERLAY_'+id,risk_color:'#123456',risk_name:'待复核',
  stats:{area_ratio:.1,num_regions:1,num_regions_total:1,max_region_area:4,regions:[{id,area_px:4,bbox:[0,0,2,2]}]},
  scale_meta:{meters_per_pixel:10},band_meta:{georeference:{crs:'EPSG:32629'}},model_info:{}};}
function report(id){return {report_id:id,detection:detection(id),agents:[],knowledge:[],final_answer:'REPORT_'+id};}
const pair={t1_overlay_b64:'T1',t2_overlay_b64:'T2',change_overlay_b64:'CHANGE',
  stats:{t1_ratio:.1,t2_ratio:.2,net_ratio:.1,n_added_regions:1,n_expanded_regions:0,n_removed_regions:0},
  change_instances:{summary:{n_stable:1,n_added:1,n_removed:0}},registration:{status:'registered',valid_ratio:1}};
const tests=[];
function test(name,fn){tests.push({name,fn});}

test('late upload cannot bind an old mask to a new file',async()=>{
  const h=harness();h.sandbox.old={name:'old.png'};h.sandbox.newFile={name:'new.png'};
  await h.run('handleMineFile(old)');const task=h.run('runUploadDetect()');await tick();
  assert.equal(h.calls.length,1);await h.run('handleMineFile(newFile)');
  assert.equal(h.calls[0].options.signal.aborted,true);h.give(0,detection('OLD'));await task;
  assert.equal(h.run('lastDetectedFile'),null);assert.equal(h.run('lastGeomStats'),null);
  assert.ok(!h.el('uploadResult').innerHTML.includes('OVERLAY_OLD'));
  assert.equal(h.el('uploadBtn').disabled,false);
});
test('older completion cannot overwrite or unlock a newer request',async()=>{
  const h=harness();h.sandbox.file={name:'same.png'};await h.run('handleMineFile(file)');
  const first=h.run('runUploadDetect()');await tick();const second=h.run('runUploadDetect()');await tick();
  h.give(0,detection('FIRST'));await first;assert.equal(h.el('uploadBtn').disabled,true);
  h.give(1,detection('SECOND'));await second;
  assert.equal(h.run('lastGeomStats.regions[0].id'),'SECOND');assert.equal(h.el('uploadBtn').disabled,false);
});
test('changing GSD during a report discards its late output and geometry',async()=>{
  const h=harness();h.sandbox.file={name:'same.png'};await h.run('handleMineFile(file)');
  const detect=h.run('runUploadDetect()');await tick();h.give(0,detection('DETECT'));await detect;
  const pipeline=h.run('runUploadPipeline()');await tick();assert.ok(h.calls[1].url.endsWith('/pipeline_upload'));
  h.el('gsdInput').value='2';await h.el('gsdInput').fire('input');h.give(1,report('OLD'));await pipeline;
  assert.equal(h.run('reportParams.upload'),null);assert.equal(h.run('lastGeomStats'),null);
  assert.ok(!h.el('rp-upload-text').textContent.includes('REPORT_OLD'));
});
test('dataset request cannot render after region changes',async()=>{
  const h=harness();const task=h.run('runDetect()');await tick();h.run("currentRegion='OTHER';invalidateScene('dataset')");
  h.give(0,{samples:[{...detection('OLD'),index:1}]});await task;
  assert.equal(h.el('sampleGrid').children.length,0);assert.equal(h.run('selectedSample'),null);
});
test('changing selected sample rejects a late dataset report',async()=>{
  const h=harness();h.sandbox.a={...detection('A'),index:1};h.sandbox.b={...detection('B'),index:2};
  h.run('selectedSample=a');const task=h.run('runPipeline()');await tick();
  h.run('renderSamples([b])');h.el('sampleGrid').children[0].click();h.give(0,report('A'));await task;
  assert.equal(h.run('selectedSample.index'),2);assert.equal(h.run('reportParams.dataset'),null);
});
test('changing T2 invalidates in-flight paired detection',async()=>{
  const h=harness();h.el('cmpT1').files=[{name:'t1.png'}];h.el('cmpT2').files=[{name:'old.png'}];
  const task=h.run('runCompare()');await tick();h.el('cmpT2').files=[{name:'new.png'}];await h.el('cmpT2').fire('change');
  h.give(0,pair);await task;assert.equal(h.run('cmpImages.t1'),null);
  assert.equal(h.el('cmpReportBtn').style.display,'none');assert.ok(!h.el('cmpResult').innerHTML.includes('CHANGE'));
});
test('comparison options cannot reuse a previous pair report',async()=>{
  const h=harness();h.el('cmpT1').files=[{name:'t1.png'}];h.el('cmpT2').files=[{name:'t2.png'}];
  const task=h.run('runCompare()');await tick();h.give(0,pair);await task;
  assert.ok(h.el('cmpResult').innerHTML.includes('10.0 个百分点'));
  assert.ok(!h.el('cmpResult').innerHTML.includes('同一滑坡体'));
  h.el('cmpModel').value='changed';await h.run('runCompareReport()');
  assert.equal(h.calls.length,1);assert.equal(h.run('cmpImages.t1'),null);
});
test('changing analysis grid discards late output and forwards the selected grid',async()=>{
  const h=harness();h.sandbox.file={name:'input.png'};await h.run('handleMineFile(file)');
  h.el('uploadGrid').value='reference_10m';h.el('gsdInput').value='2.5';
  const task=h.run('runUploadDetect()');await tick();assert.equal(h.calls[0].body.analysis_grid,'reference_10m');
  h.el('uploadGrid').value='native';await h.el('uploadGrid').fire('change');
  h.give(0,detection('OLD_GRID'));await task;
  assert.equal(h.run('lastDetectedFile'),null);assert.ok(!h.el('uploadResult').innerHTML.includes('OVERLAY_OLD_GRID'));
});
test('changing files while reading a pair never sends mixed epochs',async()=>{
  const h=harness(),a=deferred(),b=deferred();h.sandbox.a=a;h.sandbox.b=b;
  h.el('cmpT1').files=[{name:'t1.png'}];h.el('cmpT2').files=[{name:'t2.png'}];
  h.run("fileToB64=file=>file.name==='t1.png'?a.promise:b.promise");const task=h.run('runCompare()');
  h.el('cmpT2').files=[{name:'replacement.png'}];h.run("invalidateScene('compare')");
  a.resolve('OLD_T1');b.resolve('OLD_T2');await task;assert.equal(h.calls.length,0);
  assert.equal(h.el('cmpBtn').disabled,false);
});
test('file-read error and oversized uploads finish with explicit failure',async()=>{
  const h=harness();h.sandbox.file={name:'bad.png',readError:true};await h.run('handleMineFile(file)');
  await h.run('runUploadDetect()');assert.equal(h.calls.length,0);assert.equal(h.el('uploadBtn').disabled,false);
  assert.ok(h.el('uploadResult').innerHTML.includes('读取失败'));
  await assert.rejects(h.run("fileToB64({name:'large.png',size:33*1024*1024})"),/32 MiB/);
  await assert.rejects(h.run("fileToB64({name:'abort.png',readAbort:true})"),/取消/);
});
test('both vector slots retain their own regions and scale after a report',async()=>{
  const h=harness();h.run("lastGeomStats={regions:[{id:1}]};lastGeomMeta={meters_per_pixel:10,georeference:{crs:'EPSG:32629'}};datasetGeomStats={regions:[{id:2}]};datasetGeomMeta={meters_per_pixel:3,georeference:{crs:'EPSG:32719'}}");
  const one=h.run("exportVector('geojson','upload')");await tick();assert.equal(h.calls[0].body.regions[0].id,1);
  h.give(0,{filename:'a.geojson',content_b64:'e30=',n_features:1});await one;
  const two=h.run("exportVector('geojson','dataset')");await tick();assert.equal(h.calls[1].body.regions[0].id,2);
  assert.equal(h.calls[1].body.georeference.crs,'EPSG:32719');assert.equal(h.calls[1].body.meters_per_pixel,3);
  h.give(1,{filename:'b.geojson',content_b64:'e30=',n_features:1});await two;
  h.sandbox.current=report('CURRENT');h.run("renderPipelineResult(current,'upload')");
  assert.equal(h.run('lastGeomStats.regions[0].id'),'CURRENT');assert.equal(h.run('datasetGeomStats.regions[0].id'),2);
});
test('non-JSON HTTP failure does not retain a success result',async()=>{
  const h=harness();h.sandbox.file={name:'input.png'};await h.run('handleMineFile(file)');
  const task=h.run('runUploadDetect()');await tick();h.calls[0].resolve({ok:false,status:413,json:async()=>{throw new Error('HTML');}});
  await task;assert.ok(h.el('uploadResult').innerHTML.includes('HTTP 413'));assert.equal(h.run('lastDetectedFile'),null);
});
test('unknown mask coverage cannot display a false zero percent',async()=>{
  const h=harness();h.sandbox.file={name:'input.png'};await h.run('handleMineFile(file)');
  const task=h.run('runUploadDetect()');await tick();const unknown=detection('UNKNOWN');
  unknown.stats.area_ratio=null;unknown.stats.area_ratio_available=false;unknown.stats.unverified_pixels=64;
  unknown.roi={area_ratio:.5,risk_inside:0,risk_outside:0,mean_prob_inside:0};h.give(0,unknown);await task;
  assert.ok(h.el('uploadResult').innerHTML.includes('不可计算'));
  assert.ok(h.el('uploadResult').innerHTML.includes('待复核 64 像素'));
  assert.ok(!h.el('uploadResult').innerHTML.includes('>0.0%</span>'));
});

(async()=>{for(const {name,fn} of tests){await fn();console.log('PASS '+name);}console.log('PASS '+tests.length+' asynchronous scene and export checks; no network or model calls');})().catch(error=>{console.error(error);process.exitCode=1;});
