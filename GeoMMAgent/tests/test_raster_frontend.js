const fs=require('fs');
const vm=require('vm');
const assert=require('assert');
const html=fs.readFileSync('D:/GeoMMAgent/矿安天眼_前端演示(后端版).html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const els={};
function el(id){return els[id] ||= {id,style:{},value:'',files:[],src:'',textContent:'',innerHTML:'',
  classList:{add(){},remove(){},toggle(){}},addEventListener(){},scrollIntoView(){},click(){},
  getContext(){return {clearRect(){},drawImage(){},fillRect(){},strokeRect(){}}},querySelectorAll(){return []}};}
class Img {get width(){return 640} get height(){return 480} set src(v){this._src=v;this.onload?.()} get src(){return this._src}}
const sb={console,document:{getElementById:el,querySelectorAll(){return []},createElement(){return el('created')}},
 window:{addEventListener(){}},Image:Img,URL:{createObjectURL(){return 'blob:source'},revokeObjectURL(){}},
 FileReader:class{},setInterval(){return 0},setTimeout(){return 0},clearInterval(){},requestAnimationFrame(){},
 alert(){},Blob:class{},atob:x=>x,btoa:x=>x,fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sb);vm.runInContext(script,sb);
async function run(){
 sb.fetch=async()=>({ok:true,json:async()=>({status:'ok',geomm_loaded:true,detector_online:true,detector_loaded:false})});
 await vm.runInContext('checkHealth()',sb);
 assert.ok(!el('statusText').textContent.includes('离线'));console.log('OK reachable unloaded detector service is not labelled offline');
 sb.fetch=async()=>({ok:true,json:async()=>({status:'ok',geomm_loaded:true,detector_online:true,detector_runtime_compatible:false})});
 await vm.runInContext('checkHealth()',sb);
 assert.ok(el('statusText').textContent.includes('版本不一致'));console.log('OK stale detector runtime is not advertised as ready');
 const geoA={crs:'EPSG:32629',transform:[10,0,754950,0,-10,4156920]};
 const info={preview_b64:'PNG_PREVIEW',band_meta:{nir_source:'measured',georeference:geoA,nodata_pixels:10},scale_meta:{meters_per_pixel:10}};
 let sent;
 sb.fetch=async(url,opts)=>{sent=JSON.parse(opts.body);return {ok:true,json:async()=>info}};
 vm.runInContext("fileToB64=async()=> 'ORIGINAL_TIFF_BYTES'",sb);
 await vm.runInContext("handleMineFile({name:'original.tif',type:'image/tiff'})",sb);
 assert.equal(sent.image,'ORIGINAL_TIFF_BYTES');assert.equal(el('gsdInput').value,10);
 assert.equal(el('mineThumb').src,'data:image/png;base64,PNG_PREVIEW');
 assert.equal(vm.runInContext('lastMeta.georeference.crs',sb),'EPSG:32629');
 assert.ok(!el('gsdStatus').textContent.includes('<'));console.log('OK TIFF preview retains original file and captures metadata');
 el('uploadBands').value='BGRN';
 await vm.runInContext("handleMineFile(mineFileObj,true)",sb);
 assert.equal(sent.channel_order,'BGRN');console.log('OK explicitly declared TIFF bands reach the preview decoder');
 vm.runInContext("lastGeomStats={regions:[{id:1,area_px:256,contour:[[1,1],[2,1],[2,2],[1,1]]}]}; lastGeomMeta={meters_per_pixel:10,georeference:lastMeta.georeference}; lastMeta={georeference:{crs:'EPSG:32719'}};document.getElementById('gsdInput').value=1",sb);
 sb.fetch=async(url,opts)=>{sent=JSON.parse(opts.body);return {ok:true,json:async()=>({filename:'a.geojson',content_b64:'DATA',n_features:1,note:'ok'})}};
 await vm.runInContext("exportVector('geojson','upload')",sb);
 assert.equal(sent.meters_per_pixel,10);assert.equal(sent.georeference.crs,'EPSG:32629');
 console.log('OK geometry uses captured result metadata');
 await vm.runInContext("handleMineFile({name:'new.png',type:'image/png'})",sb);
 assert.equal(vm.runInContext('lastMeta',sb),null);assert.equal(vm.runInContext('lastGeomStats',sb),null);
 assert.equal(el('gsdInput').value,'');console.log('OK new file clears stale geometry and scale');
 assert.equal(el('uploadBands').value,'auto');
}
run().catch(e=>{console.error(e);process.exitCode=1});
