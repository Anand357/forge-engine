import { ForgeEngine } from './engine/ForgeEngine';
import { SceneNode, ObjType } from './engine/scene/Scene';
import { Vec3 } from './engine/math/Vec3';
import { GizmoAxis } from './engine/renderer/GizmoRenderer';
import { loadGLBFromFile, LoadedModel } from './engine/renderer/GLTFLoader';
import { Blueprint, NODE_TEMPLATES } from './engine/blueprint/Blueprint';
import { buildAssetLibrary, AssetDef } from './engine/renderer/AssetLibrary';

// ── DOM ──────────────────────────────────────────────────────────
const setupEl   = document.getElementById('setup')!;
const editorEl  = document.getElementById('editor')!;
const canvas    = document.getElementById('main-canvas') as HTMLCanvasElement;
const noWebGPU  = document.getElementById('no-webgpu') as HTMLDivElement;
const sceneList = document.getElementById('scene-list')!;
const objCount  = document.getElementById('obj-count')!;
const inspBody  = document.getElementById('insp-body')!;
const inspType  = document.getElementById('insp-type')!;
const layerList = document.getElementById('layer-list')!;
const fpsBadge  = document.getElementById('fps')!;
const curPos    = document.getElementById('cur-pos')!;
const modeBanner= document.getElementById('mode-banner')!;
const cpPanel   = document.getElementById('cp')!;

let engine!: ForgeEngine;
let pendingType: ObjType | null = null;
let placeMode: 'select'|'click'|'coord' = 'select';
let activeLayerId = '';
let frameCount = 0, fpsTimer = 0;

const undoStack: SceneNode[] = [];

// ── Gizmo drag state ─────────────────────────────────────────────
let dragAxis: GizmoAxis = 'none';
let dragParam0 = 0;
let dragNode: SceneNode | null = null;
let isDragging = false;

// ── SETUP SCREEN ─────────────────────────────────────────────────
document.querySelectorAll('.wt').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.wt').forEach(e=>e.classList.remove('active'));
    el.classList.add('active');
    const wt = (el as HTMLElement).dataset.world!;
    document.getElementById('cdims')!.classList.toggle('show', wt==='finite'||wt==='custom');
  });
});

document.getElementById('btn-start')!.addEventListener('click', async () => {
  setupEl.style.display = 'none';
  editorEl.classList.add('on');
  await boot();
});

// ── BOOT ─────────────────────────────────────────────────────────
async function boot() {
  engine = new ForgeEngine(canvas);
  const ok = await engine.init();
  if (!ok) { noWebGPU.style.display='flex'; return; }

  const tstyle = (document.getElementById('tstyle') as HTMLSelectElement).value;
  const grav   = parseFloat((document.getElementById('grav') as HTMLSelectElement).value);
  engine.physics.setGravity(grav);

  const objLayer     = engine.scene.layers[2];
  activeLayerId      = objLayer.id;

  // The engine owns an editable heightfield terrain that starts FLAT.
  // Apply an optional starting style from the setup screen.
  if (tstyle === 'mountains') {
    engine.addTerrainFeature('mountain', -30, -20, 60);
    engine.addTerrainFeature('mountain',  40,  30, 50);
    engine.addTerrainFeature('hills',     0,   0,  90);
  } else if (tstyle === 'hills') {
    engine.addTerrainFeature('hills', 0, 0, 100);
  }
  // 'flat' and 'none' → leave terrain flat

  engine.onSelect = id => { renderInspector(id ? engine.scene.get(id)??null : null); renderHierarchy(); };
  engine.onFrame  = dt => {
    frameCount++; fpsTimer+=dt;
    if (fpsTimer>=0.5) { fpsBadge.textContent=`${Math.round(frameCount/fpsTimer)} fps`; frameCount=0; fpsTimer=0; }
  };

  setupLayers();
  renderHierarchy();
  wireViewport();
  wireModes();
  wireTabs();
  wireTopbarAdd();
  wireKeys();
  wireAssets();
  wireBlueprint();
  wirePlayMode();
  wireTerrain();
  wireGameMenu();
  wireTutorial();
  wireGraphics();
  engine.start();
  tryLoadFromURL();   // restore a shared world if the URL carries one
}

// ── NDC helper ───────────────────────────────────────────────────
function toNDC(e: MouseEvent): [number,number] {
  const r = canvas.getBoundingClientRect();
  return [
    ((e.clientX-r.left)/r.width)*2-1,
    -(((e.clientY-r.top)/r.height)*2-1),
  ];
}

// ── Ray-axis intersection ─────────────────────────────────────────
function rayAxisParam(nx: number, ny: number, axisOrigin: Vec3, axisDir: Vec3): number {
  const aspect = canvas.width/canvas.height;
  const { origin: ro, dir: rd } = engine.camera.getRay(nx, ny, aspect);
  const w = ro.sub(axisOrigin);
  const b = rd.dot(axisDir);
  const c = axisDir.dot(axisDir);
  const d = rd.dot(w);
  const e = axisDir.dot(w);
  const a = rd.dot(rd);
  const denom = a*c - b*b;
  if (Math.abs(denom)<1e-8) return 0;
  return (b*d - a*e)/denom;
}

// ── VIEWPORT ─────────────────────────────────────────────────────
function wireViewport() {
  canvas.addEventListener('mousemove', e => {
    if (engine.isPlaying) return;
    if (sculptActive) { handleSculptMove(e); return; }
    const [nx,ny] = toNDC(e);
    const p = engine.placeOnTerrain(nx,ny);
    curPos.textContent = `${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`;

    if (isDragging && dragNode && dragAxis!=='none') {
      // Move object along dragged axis
      const dirs: Record<string,Vec3> = { x:new Vec3(1,0,0), y:new Vec3(0,1,0), z:new Vec3(0,0,1) };
      const axisDir = dirs[dragAxis];
      const param = rayAxisParam(nx, ny, dragNode.position, axisDir);
      const delta = param - dragParam0;
      dragNode.position = dragNode.position.add(axisDir.mul(delta));
      dragParam0 = param;
      engine.updateNode(dragNode);
      renderInspector(dragNode);
    } else if (engine.selectedId) {
      // Hover-test gizmo
      const aspect = canvas.width/canvas.height;
      const node = engine.scene.get(engine.selectedId);
      if (node) engine.gizmoAxis = engine.gizmo.hitTest(nx, ny, node.position, engine.camera, aspect);
    }
  });

  let mdX=0, mdY=0;
  canvas.addEventListener('mousedown', e => {
    if (e.button!==0 || engine.isPlaying) return;
    if (sculptMode!=='off') { sculptActive=true; handleSculptMove(e); return; }
    mdX=e.clientX; mdY=e.clientY;

    if (engine.selectedId && engine.gizmoAxis!=='none') {
      // Start gizmo drag
      isDragging = true;
      dragAxis   = engine.gizmoAxis;
      dragNode   = engine.scene.get(engine.selectedId)??null;
      if (dragNode) {
        const [nx,ny] = toNDC(e);
        const dirs: Record<string,Vec3> = { x:new Vec3(1,0,0), y:new Vec3(0,1,0), z:new Vec3(0,0,1) };
        dragParam0 = rayAxisParam(nx,ny, dragNode.position, dirs[dragAxis]);
      }
      e.stopPropagation();
    }
  });

  window.addEventListener('mouseup', e => {
    if (e.button!==0) return;
    if (sculptActive) { sculptActive=false; return; }
    if (engine.isPlaying) return;
    if (isDragging) { isDragging=false; dragAxis='none'; dragNode=null; return; }
    if (Math.abs(e.clientX-mdX)>5||Math.abs(e.clientY-mdY)>5) return;
    const [nx,ny] = toNDC(e as MouseEvent);
    if (placeMode==='click'&&pendingType) {
      const pos = engine.placeOnTerrain(nx,ny);
      const node = engine.addObject(pendingType,undefined,activeLayerId);
      node.position = new Vec3(pos.x,pos.y+yOff(pendingType),pos.z);
      engine.selectedId = node.id;
      engine.onSelect?.(node.id);
      renderHierarchy();
    } else {
      engine.raycastSelect(nx,ny);
    }
  });
}

function yOff(t: ObjType) { return t==='terrain'||t==='plane'?0:1; }

// ── MODES ─────────────────────────────────────────────────────────
function setMode(m:'select'|'click'|'coord') {
  placeMode=m;
  document.getElementById('m-select')!.classList.toggle('on',m==='select');
  document.getElementById('m-click')!.classList.toggle('on',m==='click');
  document.getElementById('m-coord')!.classList.toggle('on',m==='coord');
  if (m==='click') {
    modeBanner.textContent = pendingType?`📍 Click terrain to place ${pendingType}`:'📍 Select an object type first';
    modeBanner.classList.add('show'); cpPanel.classList.remove('show'); canvas.style.cursor='crosshair';
  } else if (m==='coord') {
    if (pendingType) { cpPanel.classList.add('show'); modeBanner.textContent=`🎯 Coordinate Placement – ${pendingType}`; modeBanner.classList.add('show'); }
    else { modeBanner.textContent='🎯 Select an object type first'; modeBanner.classList.add('show'); setTimeout(()=>modeBanner.classList.remove('show'),2500); }
    canvas.style.cursor='default';
  } else {
    modeBanner.classList.remove('show'); cpPanel.classList.remove('show'); canvas.style.cursor='default';
  }
}

function setPendingType(t: ObjType) {
  pendingType=t;
  document.querySelectorAll('.abtn2').forEach(b=>b.classList.toggle('on',(b as HTMLElement).dataset.type===t));
  document.querySelectorAll('#addbar .abtn').forEach(b=>b.classList.toggle('on',(b as HTMLElement).dataset.type===t));
  if (placeMode==='select') setMode('click'); else setMode(placeMode);
}

function wireModes() {
  document.getElementById('m-select')!.addEventListener('click',()=>setMode('select'));
  document.getElementById('m-click')!.addEventListener('click',()=>{ if(!pendingType){showBanner('📍 Select an object type first');return;} setMode('click'); });
  document.getElementById('m-coord')!.addEventListener('click',()=>{ if(!pendingType){showBanner('🎯 Select an object type first');return;} setMode('coord'); });

  document.getElementById('cp-do')!.addEventListener('click',()=>{
    if(!pendingType) return;
    const x=parseFloat((document.getElementById('cp-x') as HTMLInputElement).value)||0;
    const y=parseFloat((document.getElementById('cp-y') as HTMLInputElement).value)||1;
    const z=parseFloat((document.getElementById('cp-z') as HTMLInputElement).value)||0;
    const node=engine.addObject(pendingType,undefined,activeLayerId);
    node.position=new Vec3(x,y,z);
    engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy(); setMode('select');
  });
  document.getElementById('cp-cancel')!.addEventListener('click',()=>setMode('select'));

  const btnGrid  = document.getElementById('btn-grid')!;
  const btnFly   = document.getElementById('btn-fly')!;
  const btnFocus = document.getElementById('btn-focus')!;
  const btnDel   = document.getElementById('btn-del')!;

  btnGrid.addEventListener('click',()=>{ engine.showGrid=!engine.showGrid; btnGrid.classList.toggle('on',engine.showGrid); });
  btnFly.addEventListener('click',()=>{
    const fly=!engine.camera['flyMode']; engine.camera.setFlyMode(fly);
    btnFly.classList.toggle('on',fly); btnFly.textContent=fly?'✈ Flying':'✈ Fly';
    if(fly){modeBanner.textContent='✈ Fly Mode: WASD · Q/E up/down'; modeBanner.classList.add('show');}
    else modeBanner.classList.remove('show');
  });
  btnFocus.addEventListener('click',()=>engine.focusSelected());
  btnDel.addEventListener('click',()=>deleteSelected());

  // (AO/SSR/Fog toggles now live in the ✨ Graphics menu — see wireGraphics)

  document.getElementById('btn-play')!.addEventListener('click', startPlay);
  document.getElementById('btn-stop')!.addEventListener('click', stopPlay);

  document.getElementById('btn-clear')!.addEventListener('click',()=>{
    if(!engine.scene.nodes.length||!confirm('Clear scene?')) return;
    [...engine.scene.nodes].forEach(n=>engine.removeObject(n.id));
    undoStack.length=0; renderHierarchy(); renderInspector(null);
  });
  document.getElementById('btn-export')!.addEventListener('click',exportScene);
  document.getElementById('btn-load')!.addEventListener('click',importWorldFile);
  document.getElementById('btn-share')!.addEventListener('click',shareWorldLink);

  // Light button
  document.getElementById('btn-addlight')!.addEventListener('click',()=>{
    engine.addPointLight(); renderLightList();
  });

  // Sun controls
  document.getElementById('sun-x')!.addEventListener('input', updateSun);
  document.getElementById('sun-y')!.addEventListener('input', updateSun);
  document.getElementById('sun-z')!.addEventListener('input', updateSun);
}

function updateSun() {
  const x=parseFloat((document.getElementById('sun-x') as HTMLInputElement).value);
  const y=parseFloat((document.getElementById('sun-y') as HTMLInputElement).value);
  const z=parseFloat((document.getElementById('sun-z') as HTMLInputElement).value);
  engine.autoSun = false;             // manual sun overrides time-of-day cycle
  engine.daySpeed = 0;
  engine.sunDir = new Vec3(x,y,z).normalize();
}

function showBanner(msg: string) {
  modeBanner.textContent=msg; modeBanner.classList.add('show');
  setTimeout(()=>modeBanner.classList.remove('show'),2500);
}

// ── DELETE / UNDO / DUPLICATE ─────────────────────────────────────
function deleteSelected() {
  if(!engine.selectedId) return;
  const node=engine.scene.get(engine.selectedId);
  if(node){ undoStack.push(node); if(undoStack.length>30) undoStack.shift(); }
  engine.removeObject(engine.selectedId);
  renderHierarchy(); renderInspector(null);
}

function undoDelete() {
  const node=undoStack.pop(); if(!node) return;
  if(!engine.scene.getLayer(node.layerId)) node.layerId=engine.scene.defaultLayerId;
  engine.restoreNode(node); engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
}

function duplicateSelected() {
  if(!engine.selectedId) return;
  const src=engine.scene.get(engine.selectedId); if(!src) return;
  const node=engine.addObject(src.type,src.name+' Copy',src.layerId);
  node.position=new Vec3(src.position.x+1,src.position.y,src.position.z+1);
  node.rotation=src.rotation.clone(); node.scale=src.scale.clone();
  node.color=[...src.color] as [number,number,number];
  node.metalness=src.metalness; node.roughness=src.roughness; node.emissive=src.emissive;
  node.textured=src.textured;
  engine.updateNode(node); engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
}

// ── WORLD SAVE / LOAD / SHARE ─────────────────────────────────────
// A "world" = terrain heights + objects + lights + sun/weather settings.
// Library/primitive objects round-trip fully. Imported-glTF meshes are saved
// only in the .json file (embedded geometry), not in share links (too large).

function serializeWorld(includeMeshGeo: boolean) {
  const f32ToB64 = (arr: Float32Array) => {
    const bytes = new Uint8Array(arr.buffer);
    let bin = ''; for (let i=0;i<bytes.length;i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  };
  return {
    version: 3,
    terrain: {
      size: engine.terrain.size, res: engine.terrain.res,
      heights: f32ToB64(engine.terrain.heights),
    },
    sun: { timeOfDay: engine.timeOfDay, daySpeed: engine.daySpeed, autoSun: engine.autoSun, wetness: engine.wetness },
    layers: engine.scene.layers.map(l => ({ id:l.id, name:l.name, color:l.color, visible:l.visible, locked:l.locked, opacity:l.opacity })),
    nodes: engine.scene.nodes.map(n => ({
      name:n.name, type:n.type, layerId:n.layerId,
      position:[n.position.x,n.position.y,n.position.z],
      rotation:[n.rotation.x,n.rotation.y,n.rotation.z],
      scale:[n.scale.x,n.scale.y,n.scale.z],
      color:n.color, metalness:n.metalness, roughness:n.roughness, emissive:n.emissive,
      textured:n.textured, isWater:n.isWater, gameRole:n.gameRole,
      visible:n.visible, locked:n.locked, hasPhysics:n.hasPhysics, isStatic:n.isStatic,
      // library asset name (so we can rebuild its geometry on load)
      asset: (n as any)._assetName ?? null,
      // embed raw mesh only for file export
      meshGeo: (includeMeshGeo && n.meshGeo) ? {
        v: f32ToB64(n.meshGeo.vertices),
        i: btoa(String.fromCharCode(...new Uint8Array(n.meshGeo.indices.buffer))),
      } : null,
    })),
  };
}

function deserializeWorld(data: any) {
  const b64ToF32 = (s: string) => {
    const bin = atob(s); const bytes = new Uint8Array(bin.length);
    for (let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  };
  const b64ToU16 = (s: string) => {
    const bin = atob(s); const bytes = new Uint8Array(bin.length);
    for (let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
    return new Uint16Array(bytes.buffer);
  };

  // wipe current world
  [...engine.scene.nodes].forEach(n => engine.removeObject(n.id));
  undoStack.length = 0;

  // terrain
  if (data.terrain?.heights) {
    const h = b64ToF32(data.terrain.heights);
    if (h.length === engine.terrain.heights.length) engine.terrain.heights.set(h);
    engine.rebuildTerrain();
  }
  // sun / weather
  if (data.sun) {
    engine.timeOfDay = data.sun.timeOfDay ?? 0.35;
    engine.daySpeed  = data.sun.daySpeed ?? 0;
    engine.autoSun   = data.sun.autoSun ?? true;
    engine.wetness   = data.sun.wetness ?? 0;
    engine.renderer.wetness = engine.wetness;
  }
  // nodes
  for (const nd of data.nodes ?? []) {
    let node;
    if (nd.type === 'mesh') {
      let geo = null;
      if (nd.meshGeo) geo = { vertices: b64ToF32(nd.meshGeo.v), indices: b64ToU16(nd.meshGeo.i) };
      else if (nd.asset) { const a = library.find(l => l.name === nd.asset); if (a) geo = a.build(); }
      if (!geo) continue;   // can't rebuild this mesh
      node = engine.addMesh(geo, nd.name, nd.layerId);
      if (nd.asset) (node as any)._assetName = nd.asset;
    } else {
      node = engine.addObject(nd.type, nd.name, nd.layerId);
    }
    node.position = new Vec3(nd.position[0], nd.position[1], nd.position[2]);
    node.rotation = new Vec3(nd.rotation[0], nd.rotation[1], nd.rotation[2]);
    node.scale    = new Vec3(nd.scale[0], nd.scale[1], nd.scale[2]);
    node.color = nd.color; node.metalness = nd.metalness; node.roughness = nd.roughness;
    node.emissive = nd.emissive ?? 0; node.textured = !!nd.textured; node.isWater = !!nd.isWater;
    node.gameRole = nd.gameRole ?? 'none';
    node.visible = nd.visible !== false; node.locked = !!nd.locked;
    node.hasPhysics = !!nd.hasPhysics; node.isStatic = nd.isStatic !== false;
    engine.updateNode(node);
  }
  renderHierarchy();
  showBanner('📂 World loaded');
}

function exportScene() {
  const json = JSON.stringify(serializeWorld(true));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([json], { type:'application/json' }));
  a.download = 'forge-world.json'; a.click(); URL.revokeObjectURL(a.href);
  showBanner('💾 World saved to forge-world.json');
}

function importWorldFile() {
  const input = document.createElement('input');
  input.type = 'file'; input.accept = '.json,application/json';
  input.addEventListener('change', async () => {
    const file = input.files?.[0]; if (!file) return;
    try { deserializeWorld(JSON.parse(await file.text())); }
    catch (e) { alert('Load failed: ' + (e as Error).message); }
  });
  input.click();
}

function utf8ToB64(str: string): string {
  const bytes = new TextEncoder().encode(str);
  let bin = ''; for (let i=0;i<bytes.length;i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
function b64ToUtf8(b64: string): string {
  const bin = atob(b64); const bytes = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function shareWorldLink() {
  // share links exclude raw glTF geometry to stay small
  const json = JSON.stringify(serializeWorld(false));
  const b64 = utf8ToB64(json);
  const url = `${location.origin}${location.pathname}#world=${b64}`;
  if (url.length > 30000) {
    alert('This world is too large for a share link. Use 💾 Save to a file instead.');
    return;
  }
  navigator.clipboard?.writeText(url).then(
    () => showBanner('🔗 Share link copied to clipboard!'),
    () => prompt('Copy this share link:', url),
  );
}

// On load, if URL has a #world= payload, restore it.
function tryLoadFromURL() {
  const m = location.hash.match(/#world=(.+)$/);
  if (!m) return;
  try {
    deserializeWorld(JSON.parse(b64ToUtf8(m[1])));
    showBanner('🔗 Loaded shared world');
  } catch (e) { console.warn('shared world load failed', e); }
}

function wireKeys() {
  window.addEventListener('keydown',e=>{
    if(e.target instanceof HTMLInputElement||e.target instanceof HTMLSelectElement) return;
    const k=e.key.toLowerCase();
    const fly=engine?.camera['flyMode']??false;
    if(!fly){
      if(k==='q') setMode('select');
      if(k==='e'&&pendingType) setMode('click');
      if(k==='r'&&pendingType) setMode('coord');
    }
    if(k==='f'){ const nf=!engine.camera['flyMode']; engine.camera.setFlyMode(nf); document.getElementById('btn-fly')!.classList.toggle('on',nf); document.getElementById('btn-fly')!.textContent=nf?'✈ Flying':'✈ Fly'; }
    if(k===' '){ e.preventDefault(); engine.focusSelected(); }
    if(k==='delete'||k==='backspace') deleteSelected();
    if(k==='escape') setMode('select');
    if((e.ctrlKey||e.metaKey)&&k==='d'){ e.preventDefault(); duplicateSelected(); }
    if((e.ctrlKey||e.metaKey)&&k==='z'){ e.preventDefault(); undoDelete(); }
    if((e.ctrlKey||e.metaKey)&&k==='s'){ e.preventDefault(); exportScene(); }
  });
}

// ── TOPBAR ADD ────────────────────────────────────────────────────
function wireTopbarAdd() {
  document.querySelectorAll('#obj-btns .abtn2').forEach(btn=>{
    btn.addEventListener('click',()=>setPendingType((btn as HTMLElement).dataset.type as ObjType));
  });
  document.querySelectorAll('#addbar .abtn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const type=(btn as HTMLElement).dataset.type as ObjType;
      const node=engine.addObject(type,undefined,activeLayerId);
      node.position=new Vec3((Math.random()-0.5)*14, yOff(type), (Math.random()-0.5)*14);
      engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
    });
  });
}

// ── ASSET BROWSER (procedural library + glTF import) ──────────────
const importedAssets: LoadedModel[] = [];
const library: AssetDef[] = buildAssetLibrary();
let assetFilter = '';
let assetCategory = 'All';

const CAT_ICON: Record<string,string> = {
  Characters:'🧍', Vehicles:'🚗', Nature:'🌲', Buildings:'🏠', Props:'📦', Imported:'⬆',
};

function wireAssets() {
  const btn  = document.getElementById('btn-import-glb')!;
  const input= document.getElementById('glb-file') as HTMLInputElement;
  btn.addEventListener('click',()=>input.click());
  input.addEventListener('change', async ()=>{
    const files = input.files; if(!files||!files.length) return;
    for(const file of Array.from(files)){
      try {
        const model = await loadGLBFromFile(file);
        importedAssets.push(model);
        renderAssetList();
      } catch(err){
        console.error('glTF import failed', err);
        alert('Import failed: '+(err as Error).message);
      }
    }
    input.value='';
  });

  // Category tabs + search
  document.querySelectorAll('.asset-cat').forEach(t=>{
    t.addEventListener('click',()=>{
      document.querySelectorAll('.asset-cat').forEach(x=>x.classList.remove('on'));
      t.classList.add('on');
      assetCategory=(t as HTMLElement).dataset.cat!;
      renderAssetList();
    });
  });
  const search=document.getElementById('asset-search') as HTMLInputElement;
  search.addEventListener('input',()=>{ assetFilter=search.value.toLowerCase(); renderAssetList(); });

  // "Set as player" picker is wired in wirePlayMode()
  renderAssetList();
}

function placeLibraryAsset(a: AssetDef) {
  const geo=a.build();
  const node=engine.addMesh(geo, a.name, activeLayerId);
  (node as any)._assetName = a.name;          // for save/load round-trip
  node.color=[...a.color] as [number,number,number];
  node.roughness=a.roughness; node.metalness=a.metalness;
  // drop onto terrain near origin
  const x=(Math.random()-0.5)*16, z=(Math.random()-0.5)*16;
  node.position=new Vec3(x, engine.groundHeightAt(x,z), z);
  engine.updateNode(node);
  engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
}

function renderAssetList() {
  const list=document.getElementById('asset-list')!;
  list.innerHTML='';

  // Imported .glb section
  const showImported = assetCategory==='All' || assetCategory==='Imported';
  if(showImported && importedAssets.length){
    const hdr=document.createElement('div'); hdr.className='asset-group'; hdr.textContent='Imported';
    list.appendChild(hdr);
    importedAssets.forEach(model=>{
      if(assetFilter && !model.name.toLowerCase().includes(assetFilter)) return;
      const el=document.createElement('div'); el.className='asset-item';
      el.innerHTML=`<span class="asset-thumb">⬆</span><div class="asset-meta"><div class="asset-name">${model.name}</div><div class="asset-sub">${model.triangleCount.toLocaleString()} tris · imported</div></div><span class="asset-add">＋ add</span>`;
      el.addEventListener('click',()=>{
        const node=engine.addMesh(model.geo, model.name, activeLayerId);
        const x=(Math.random()-0.5)*12, z=(Math.random()-0.5)*12;
        node.position=new Vec3(x, engine.groundHeightAt(x,z), z);
        engine.updateNode(node);
        engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
      });
      list.appendChild(el);
    });
  }

  // Library grouped by category
  const cats=['Characters','Vehicles','Nature','Buildings','Props'];
  for(const cat of cats){
    if(assetCategory!=='All' && assetCategory!==cat) continue;
    const items=library.filter(a=>a.category===cat &&
      (!assetFilter || a.name.toLowerCase().includes(assetFilter)));
    if(!items.length) continue;
    const hdr=document.createElement('div'); hdr.className='asset-group';
    hdr.textContent=`${CAT_ICON[cat]} ${cat} (${items.length})`;
    list.appendChild(hdr);
    for(const a of items){
      const el=document.createElement('div'); el.className='asset-item';
      const rgb=`rgb(${Math.round(a.color[0]*255)},${Math.round(a.color[1]*255)},${Math.round(a.color[2]*255)})`;
      el.innerHTML=`<span class="asset-thumb" style="background:linear-gradient(135deg,${rgb},#1a1a30)">${CAT_ICON[cat]}</span><div class="asset-meta"><div class="asset-name">${a.name}</div><div class="asset-sub">${cat}</div></div><span class="asset-add">＋ add</span>`;
      el.addEventListener('click',()=>placeLibraryAsset(a));
      list.appendChild(el);
    }
  }

  if(!list.children.length){
    list.innerHTML='<div style="padding:20px 14px;color:var(--dim);font-size:11px;text-align:center">No matching assets.</div>';
  }
}

// ── PLAY MODE (3rd-person player) ─────────────────────────────────
let playMouseDown = false;

function startPlay() {
  engine.startPlay();
  document.getElementById('btn-play')!.style.display='none';
  document.getElementById('btn-stop')!.style.display='';
  document.getElementById('editor')!.classList.add('playing');
  document.getElementById('play-hud')!.classList.add('show');
  canvas.style.cursor='none';
}

function stopPlay() {
  engine.stopPlay();
  document.getElementById('btn-play')!.style.display='';
  document.getElementById('btn-stop')!.style.display='none';
  document.getElementById('editor')!.classList.remove('playing');
  document.getElementById('play-hud')!.classList.remove('show');
  document.getElementById('game-hud')!.classList.remove('show');
  document.getElementById('game-end')!.classList.remove('show');
  canvas.style.cursor='default';
  renderHierarchy();
}

function wirePlayMode() {
  // Vehicle enter/exit prompt
  engine.onVehicleStateChange = (driving, prompt) => {
    const promptEl = document.getElementById('drive-prompt')!;
    const footHud  = document.getElementById('play-hud')!;
    const driveHud = document.getElementById('drive-hud')!;
    promptEl.style.display = (!driving && prompt) ? 'block' : 'none';
    promptEl.textContent = prompt;
    footHud.classList.toggle('show', engine.isPlaying && !driving);
    driveHud.classList.toggle('show', engine.isPlaying && driving);
  };

  // Gameplay HUD updates
  engine.onGameUpdate = (s) => {
    const hud = document.getElementById('game-hud')!;
    // only show HUD if the world actually has game objects
    const hasGame = s.totalCollectibles > 0 || s.totalCheckpoints > 0 || s.maxHealth !== s.health || s.score > 0;
    hud.classList.toggle('show', engine.isPlaying && (s.totalCollectibles > 0 || s.totalCheckpoints > 0 || hasGame));
    (document.getElementById('gh-health') as HTMLElement).style.width = `${(s.health/s.maxHealth)*100}%`;
    document.getElementById('gh-score')!.textContent = String(s.score);
    document.getElementById('gh-collect')!.textContent = `${s.collected}/${s.totalCollectibles}`;
    document.getElementById('gh-cp')!.textContent = `${s.checkpoints}/${s.totalCheckpoints}`;
    document.getElementById('gh-collect-wrap')!.style.display = s.totalCollectibles>0?'':'none';
    document.getElementById('gh-cp-wrap')!.style.display = s.totalCheckpoints>0?'':'none';
    const mm = Math.floor(s.time/60), ss = Math.floor(s.time%60);
    document.getElementById('gh-time')!.textContent = `${mm}:${ss.toString().padStart(2,'0')}`;
    // win/lose
    if (s.status !== 'playing') {
      const end = document.getElementById('game-end')!;
      end.classList.add('show');
      document.getElementById('game-end-msg')!.textContent = s.message;
      document.getElementById('game-end-sub')!.textContent = `Score ${s.score} · Time ${mm}:${ss.toString().padStart(2,'0')}`;
    }
  };
  document.getElementById('game-end-btn')!.addEventListener('click', () => {
    document.getElementById('game-end')!.classList.remove('show');
    stopPlay(); startPlay();
  });

  // Keyboard → on-foot OR driving input
  const setKey = (e: KeyboardEvent, down: boolean) => {
    if (!engine.isPlaying) return;
    const k = e.key.toLowerCase();

    if (k==='e' && down) { engine.toggleVehicle(); e.preventDefault(); return; }

    if (engine.isDriving) {
      const v = engine.vehicleInput;
      if (k==='w'||k==='arrowup')    v.throttle  = down;
      if (k==='s'||k==='arrowdown')  v.brake     = down;
      if (k==='a'||k==='arrowleft')  v.steerLeft = down;
      if (k==='d'||k==='arrowright') v.steerRight= down;
      if (k===' ') { v.handbrake = down; e.preventDefault(); }
      return;
    }

    const inp = engine.playerInput;
    if (k==='w'||k==='arrowup')    inp.forward = down;
    if (k==='s'||k==='arrowdown')  inp.back    = down;
    if (k==='a'||k==='arrowleft')  inp.left    = down;
    if (k==='d'||k==='arrowright') inp.right   = down;
    if (k==='shift')               inp.sprint  = down;
    if (k===' ') { if(down) inp.jumpPressed = true; e.preventDefault(); }
  };
  window.addEventListener('keydown', e=>setKey(e, true));
  window.addEventListener('keyup',   e=>setKey(e, false));

  // Mouse drag → orbit follow camera (works on-foot and driving)
  canvas.addEventListener('mousedown', e=>{ if(engine.isPlaying && e.button===0){ playMouseDown=true; } });
  window.addEventListener('mouseup',   ()=>{ playMouseDown=false; });
  window.addEventListener('mousemove', e=>{
    if(!engine.isPlaying || !playMouseDown) return;
    const ctrl = engine.isDriving ? engine.vehicle : engine.player;
    ctrl?.addMouseDelta(e.movementX, e.movementY);
  });
  canvas.addEventListener('wheel', e=>{
    if(!engine.isPlaying) return;
    const ctrl = engine.isDriving ? engine.vehicle : engine.player;
    if(ctrl){ ctrl.zoom(e.deltaY); e.preventDefault(); }
  }, { passive:false });

  // Player model picker
  const sel = document.getElementById('player-model') as HTMLSelectElement;
  for (const a of library.filter(a=>a.category==='Characters')) {
    const opt=document.createElement('option'); opt.value=a.name; opt.textContent=a.name;
    if(a.name===engine.playerAssetName) opt.selected=true;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', ()=>{ engine.playerAssetName = sel.value; });
}

// ── GAME TEMPLATES ────────────────────────────────────────────────
function spawnAsset(name: string, x: number, z: number, yaw = 0, scale = 1) {
  const a = library.find(l => l.name === name) ?? library.find(l => l.name.startsWith(name));
  if (!a) return null;
  const node = engine.addMesh(a.build(), a.name, activeLayerId);
  (node as any)._assetName = a.name;
  node.color = [...a.color] as [number,number,number];
  node.roughness = a.roughness; node.metalness = a.metalness;
  node.scale = new Vec3(scale, scale, scale);
  node.rotation = new Vec3(0, yaw, 0);
  node.position = new Vec3(x, engine.groundHeightAt(x, z), z);
  engine.updateNode(node);
  return node;
}

function clearWorld() {
  [...engine.scene.nodes].forEach(n => engine.removeObject(n.id));
  undoStack.length = 0;
  engine.flattenTerrain();
}

function buildTemplate(kind: 'driving'|'openworld'|'obstacle') {
  if (engine.scene.nodes.length && !confirm('Replace the current scene with this template?')) return;
  clearWorld();

  if (kind === 'driving') {
    // Rolling hills + cars to drive + roadside props
    engine.addTerrainFeature('hills', 0, 0, 110);
    engine.addTerrainFeature('mountain', -70, -60, 50);
    spawnAsset('Car (Red)', 4, 4, 0.6);
    spawnAsset('Car (Blue)', -6, 2, -0.4);
    spawnAsset('Sports Car', 0, -8, 1.2);
    spawnAsset('Truck (Red)', 12, -4, 0.2);
    for (let i = 0; i < 12; i++) {
      const ang = (i / 12) * Math.PI * 2;
      spawnAsset('Pine Tree', Math.cos(ang) * 55, Math.sin(ang) * 55, 0, 1.4);
    }
    for (let i = 0; i < 6; i++) spawnAsset('Lamp Post', -40 + i * 16, 14);
    showBanner('🏁 Driving Map ready — press Play, then E to drive');
  } else if (kind === 'openworld') {
    engine.addTerrainFeature('mountain', -50, -40, 70);
    engine.addTerrainFeature('mountain', 60, 30, 60);
    engine.addTerrainFeature('hills', 0, 0, 130);
    engine.addTerrainFeature('canyon', 20, -50, 40);
    for (let i = 0; i < 20; i++) {
      const x = (Math.random()-0.5)*120, z = (Math.random()-0.5)*120;
      spawnAsset(Math.random()<0.5?'Oak Tree':'Pine Tree', x, z, Math.random()*6, 0.8+Math.random()*0.8);
    }
    for (let i = 0; i < 6; i++) spawnAsset('Rock', (Math.random()-0.5)*80, (Math.random()-0.5)*80, 0, 1+Math.random()*2);
    spawnAsset('House (Red)', 10, 10, 0.5);
    spawnAsset('House (Blue)', -14, 8, -0.3);
    spawnAsset('Windmill', 30, -20);
    spawnAsset('Car (Green)', 6, 2, 0);
    addWater(1.5);
    showBanner('🌍 Open World ready — explore on foot or drive');
  } else {
    // Obstacle course: flat ground + jumpable platforms
    for (let i = 0; i < 14; i++) {
      const z = -10 - i * 6;
      const x = Math.sin(i * 0.9) * 8;
      const node = engine.addObject('cube', `Platform ${i+1}`, activeLayerId);
      node.position = new Vec3(x, 1 + (i % 3), z);
      node.scale = new Vec3(3, 0.5, 3);
      node.color = [0.3, 0.6, 0.9];
      engine.updateNode(node);
    }
    spawnAsset('Crate', 0, -4, 0);
    showBanner('🎯 Obstacle Course ready — press Play and jump across');
  }
  renderHierarchy();
}

function wireGraphics() {
  const btn = document.getElementById('btn-graphics')!;
  const menu = document.getElementById('gfx-menu')!;
  btn.addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('show'); });
  document.addEventListener('click', () => menu.classList.remove('show'));
  menu.addEventListener('click', e => e.stopPropagation());

  const r = () => engine.renderer;

  // Quality presets
  menu.querySelectorAll('.gfx-preset').forEach(b => {
    b.addEventListener('click', () => {
      menu.querySelectorAll('.gfx-preset').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      const q = parseInt((b as HTMLElement).dataset.q!);
      r().quality = q;
      // auto-tune effect toggles per preset, then reflect in checkboxes
      const clouds = q >= 1, godrays = q >= 1, ssao = q >= 1, ssr = q >= 1;
      r().cloudsEnabled = clouds; r().godraysEnabled = godrays;
      r().ssaoEnabled = ssao; r().ssrEnabled = ssr; r().waterEnabled = true;
      (document.getElementById('gfx-clouds') as HTMLInputElement).checked = clouds;
      (document.getElementById('gfx-godrays') as HTMLInputElement).checked = godrays;
      (document.getElementById('gfx-ssao') as HTMLInputElement).checked = ssao;
      (document.getElementById('gfx-ssr') as HTMLInputElement).checked = ssr;
      showBanner(`✨ Graphics: ${['Low','Medium','Ultra'][q]}`);
    });
  });

  // Individual toggles
  const tog = (id: string, set: (v: boolean) => void) => {
    const el = document.getElementById(id) as HTMLInputElement;
    el.addEventListener('change', () => set(el.checked));
  };
  tog('gfx-clouds',  v => r().cloudsEnabled  = v);
  tog('gfx-godrays', v => r().godraysEnabled = v);
  tog('gfx-water',   v => r().waterEnabled   = v);
  tog('gfx-ssao',    v => r().ssaoEnabled    = v);
  tog('gfx-ssr',     v => r().ssrEnabled     = v);
  tog('gfx-fog',     v => r().fogEnabled     = v);

  // ── Time of day + weather ──
  const tod = document.getElementById('gfx-tod') as HTMLInputElement;
  const todV = document.getElementById('gfx-tod-v')!;
  const fmtTime = (t: number) => {
    const h = Math.floor(t * 24), m = Math.floor((t*24 - h) * 60);
    return `${h}:${m.toString().padStart(2,'0')}`;
  };
  tod.addEventListener('input', () => {
    engine.timeOfDay = parseFloat(tod.value); engine.autoSun = true;
    todV.textContent = fmtTime(engine.timeOfDay);
  });
  todV.textContent = fmtTime(0.35);

  const ds = document.getElementById('gfx-dayspeed') as HTMLInputElement;
  const dsV = document.getElementById('gfx-dayspeed-v')!;
  ds.addEventListener('input', () => {
    engine.daySpeed = parseFloat(ds.value);
    dsV.textContent = engine.daySpeed === 0 ? 'off' : `${engine.daySpeed.toFixed(1)}×`;
  });

  const rain = document.getElementById('gfx-rain') as HTMLInputElement;
  const rainV = document.getElementById('gfx-rain-v')!;
  rain.addEventListener('input', () => {
    engine.wetness = parseFloat(rain.value);
    r().wetness = engine.wetness;
    rainV.textContent = `${Math.round(engine.wetness*100)}%`;
  });

  // Keep the Time slider live when the day cycle auto-advances
  setInterval(() => {
    if (engine.daySpeed !== 0) {
      tod.value = String(engine.timeOfDay);
      todV.textContent = fmtTime(engine.timeOfDay);
    }
  }, 250);
}

function wireGameMenu() {
  const btn = document.getElementById('btn-newgame')!;
  const menu = document.getElementById('newgame-menu')!;
  btn.addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('show'); });
  document.addEventListener('click', () => menu.classList.remove('show'));
  menu.addEventListener('click', e => e.stopPropagation());
  menu.querySelectorAll('.ng-item').forEach(it => {
    it.addEventListener('click', () => {
      const tpl = (it as HTMLElement).dataset.tpl!;
      menu.classList.remove('show');
      if (tpl === 'empty') {
        if (engine.scene.nodes.length && !confirm('Replace the current scene?')) return;
        clearWorld(); renderHierarchy(); showBanner('⬜ Empty flat world');
      } else {
        buildTemplate(tpl as any);
      }
    });
  });
}

// ── GUIDED TUTORIAL ───────────────────────────────────────────────
const TUTORIAL_STEPS = [
  `<h3>Welcome to Forge 🎮</h3>Make a 3D game in minutes. Let's walk through the basics — or hit <b>Skip</b> anytime.`,
  `<h3>1 · Start a Game</h3>Click <b>🎮 New Game</b> (top-left) and pick a template like <b>Driving Map</b> or <b>Open World</b>. It builds a ready-to-play scene instantly.`,
  `<h3>2 · Shape the Land</h3>Open the <b>Terrain</b> tab. Use the <b>sculpt brush</b> to raise mountains or carve valleys, drop <b>feature presets</b>, or add <b>water</b>.`,
  `<h3>3 · Add Objects</h3>Open the <b>Assets</b> tab — 100+ models (cars, trees, houses, characters). Click any to drop it into the world.`,
  `<h3>4 · Play It</h3>Press <span class="key">▶ Play</span>. You spawn as a character: <span class="key">WASD</span> move, <span class="key">Shift</span> sprint, <span class="key">Space</span> jump, drag mouse to look.`,
  `<h3>5 · Drive 🚗</h3>Walk up to any car and press <span class="key">E</span> to drive — <span class="key">W</span> gas, <span class="key">S</span> brake, <span class="key">A/D</span> steer, <span class="key">Space</span> handbrake. Press <span class="key">E</span> to get out.`,
  `<h3>You're ready! 🚀</h3>Build, sculpt, place, and Play. Reopen this anytime with the <b>❔ Help</b> button.`,
];
let tutStep = 0;

function wireTutorial() {
  const overlay = document.getElementById('tut-overlay')!;
  const stepEl  = document.getElementById('tut-step')!;
  const dotsEl  = document.getElementById('tut-dots')!;
  const nextBtn = document.getElementById('tut-next')!;

  const render = () => {
    stepEl.innerHTML = TUTORIAL_STEPS[tutStep];
    dotsEl.innerHTML = TUTORIAL_STEPS.map((_,i)=>`<span class="d${i===tutStep?' on':''}"></span>`).join('');
    nextBtn.textContent = tutStep === TUTORIAL_STEPS.length-1 ? 'Start Building ✓' : 'Next ▸';
  };
  const open = () => { tutStep=0; render(); overlay.classList.add('show'); };
  const close = () => overlay.classList.remove('show');

  document.getElementById('btn-tutorial')!.addEventListener('click', open);
  document.getElementById('tut-skip')!.addEventListener('click', close);
  nextBtn.addEventListener('click', () => {
    if (tutStep >= TUTORIAL_STEPS.length-1) { close(); return; }
    tutStep++; render();
  });

  // Auto-open on first visit
  if (!localStorage.getItem('forge-tut-seen')) {
    localStorage.setItem('forge-tut-seen','1');
    setTimeout(open, 400);
  }
}

// ── TERRAIN SCULPT + FEATURES ─────────────────────────────────────
type SculptMode = 'off'|'raise'|'lower'|'smooth';
let sculptMode: SculptMode = 'off';
let sculptActive = false;
let sculptRadius = 14;
let sculptStrength = 0.6;
let featureKind: 'mountain'|'hills'|'canyon'|'plateau' = 'mountain';
let featureSize = 50;

function handleSculptMove(e: MouseEvent) {
  if (sculptMode==='off' || !sculptActive) return;
  const [nx,ny] = toNDC(e);
  const hit = engine.raycastTerrain(nx,ny);
  if (!hit) return;
  const mode = sculptMode==='raise'?1 : sculptMode==='lower'?-1 : 0;
  engine.sculptTerrain(hit.x, hit.z, sculptRadius, sculptStrength * (mode===0?1:0.15), mode);
}

function wireTerrain() {
  // Sculpt mode buttons
  const modes: [string,SculptMode][] = [['ts-raise','raise'],['ts-lower','lower'],['ts-smooth','smooth']];
  for (const [id,mode] of modes) {
    document.getElementById(id)!.addEventListener('click',()=>{
      sculptMode = sculptMode===mode ? 'off' : mode;
      modes.forEach(([i,m])=>document.getElementById(i)!.classList.toggle('on', sculptMode===m));
      canvas.style.cursor = sculptMode==='off' ? 'default' : 'crosshair';
      if (sculptMode!=='off') showBanner(`⛰ Sculpt: ${sculptMode} — drag on terrain`);
    });
  }
  const rad = document.getElementById('ts-radius') as HTMLInputElement;
  const str = document.getElementById('ts-strength') as HTMLInputElement;
  rad.addEventListener('input',()=>{ sculptRadius=parseFloat(rad.value); document.getElementById('ts-radius-v')!.textContent=rad.value; });
  str.addEventListener('input',()=>{ sculptStrength=parseFloat(str.value); document.getElementById('ts-strength-v')!.textContent=str.value; });

  // Feature presets
  document.querySelectorAll('.tf-kind').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('.tf-kind').forEach(x=>x.classList.remove('on'));
      b.classList.add('on'); featureKind=(b as HTMLElement).dataset.kind as any;
    });
  });
  const fsz = document.getElementById('tf-size') as HTMLInputElement;
  fsz.addEventListener('input',()=>{ featureSize=parseFloat(fsz.value); document.getElementById('tf-size-v')!.textContent=fsz.value; });
  document.getElementById('tf-add')!.addEventListener('click',()=>{
    // place feature at center of current view (raycast screen center)
    const hit = engine.raycastTerrain(0,0) ?? new Vec3(0,0,0);
    engine.addTerrainFeature(featureKind, hit.x, hit.z, featureSize);
    showBanner(`⛰ Added ${featureKind} (size ${featureSize})`);
  });
  document.getElementById('tf-flatten')!.addEventListener('click',()=>{
    if(confirm('Flatten the entire terrain?')){ engine.flattenTerrain(); }
  });

  // Water
  const wlevel = document.getElementById('tw-level') as HTMLInputElement;
  wlevel.addEventListener('input',()=>{ document.getElementById('tw-level-v')!.textContent=wlevel.value; });
  document.getElementById('tw-add')!.addEventListener('click',()=>{
    const level = parseFloat(wlevel.value);
    addWater(level);
  });
}

let waterLayerId = '';
function addWater(level: number) {
  // Dedicated semi-transparent layer so water transparency doesn't affect other objects.
  if (!waterLayerId || !engine.scene.getLayer(waterLayerId)) {
    const wl = engine.scene.addLayer({ name:'Water', color:'#3dc8ff' });
    wl.opacity = 0.62;
    waterLayerId = wl.id;
    renderLayers();
  }
  const node = engine.addObject('plane','Water',waterLayerId);
  node.position = new Vec3(0, level, 0);
  node.scale = new Vec3(12, 1, 12);          // plane is 20×20 base → 240×240
  node.color = [0.10, 0.26, 0.40];
  node.metalness = 0.6;
  node.roughness = 0.05;
  node.textured = false;
  node.isWater = true;                        // animated reflective water shader
  engine.updateNode(node);
  engine.selectedId=node.id; engine.onSelect?.(node.id); renderHierarchy();
  showBanner('🌊 Water plane added — adjust height in inspector');
}

// ── BLUEPRINT (string-mode visual scripting) ──────────────────────
let blueprint: Blueprint | null = null;
let bpOpen = false;
let bpRAF = 0;

const BP_PALETTE: { group: string; items: { key: string; label: string }[] }[] = [
  { group: 'Events', items: [
    { key:'event-begin', label:'On Begin Play' },
    { key:'event-click', label:'On Clicked' },
    { key:'event-tick',  label:'On Tick' },
  ]},
  { group: 'Actions', items: [
    { key:'act-move',    label:'Move Object' },
    { key:'act-rotate',  label:'Rotate Object' },
    { key:'act-spawn',   label:'Spawn Object' },
    { key:'act-destroy', label:'Destroy Object' },
    { key:'act-light',   label:'Set Light' },
  ]},
  { group: 'Flow', items: [
    { key:'flow-branch', label:'Branch' },
    { key:'flow-delay',  label:'Delay' },
  ]},
];

function wireBlueprint() {
  const overlay = document.getElementById('bp-overlay')!;
  const canvas  = document.getElementById('bp-canvas') as HTMLCanvasElement;
  blueprint = new Blueprint(canvas);

  // Build palette
  const palette = document.getElementById('bp-palette')!;
  for (const grp of BP_PALETTE) {
    const g = document.createElement('div'); g.className='bp-pgroup'; g.textContent=grp.group;
    palette.appendChild(g);
    for (const it of grp.items) {
      const tpl = NODE_TEMPLATES[it.key];
      const el = document.createElement('div'); el.className='bp-pitem';
      el.innerHTML = `<span class="bp-pdot" style="background:${tpl.color}"></span>${it.label}`;
      el.addEventListener('click', ()=>{
        // drop near center of current view
        const cx = canvas.width/2 - 80, cy = canvas.height/2 - 40;
        blueprint!.addNode(it.key, cx + (Math.random()-0.5)*120, cy + (Math.random()-0.5)*120);
      });
      palette.appendChild(el);
    }
  }

  const open  = ()=>{ overlay.classList.add('on'); bpOpen=true; resizeBP(); loopBP(); document.getElementById('btn-blueprint')!.classList.add('on'); };
  const close = ()=>{ overlay.classList.remove('on'); bpOpen=false; cancelAnimationFrame(bpRAF); document.getElementById('btn-blueprint')!.classList.remove('on'); };

  document.getElementById('btn-blueprint')!.addEventListener('click', ()=> bpOpen ? close() : open());
  document.getElementById('bp-close')!.addEventListener('click', close);
  document.getElementById('bp-close-2')!.addEventListener('click', close);
  document.getElementById('bp-clear')!.addEventListener('click', ()=> blueprint!.clear());

  // Delete selected node with Del/Backspace while overlay open
  window.addEventListener('keydown', e=>{
    if(!bpOpen) return;
    if((e.key==='Delete'||e.key==='Backspace') && blueprint!.selectedNode){
      blueprint!.removeNode(blueprint!.selectedNode.id);
    }
    if(e.key==='Escape') close();
  });

  window.addEventListener('resize', ()=>{ if(bpOpen) resizeBP(); });

  function resizeBP() {
    const wrap = document.getElementById('bp-canvas-wrap')!;
    blueprint!.resize(wrap.clientWidth, wrap.clientHeight);
  }
  function loopBP() {
    if(!bpOpen) return;
    blueprint!.draw();
    bpRAF = requestAnimationFrame(loopBP);
  }

  // Seed a small example graph so it's not empty on first open
  const a = blueprint.addNode('event-begin', 120, 140);
  const b = blueprint.addNode('act-spawn',   380, 140);
  blueprint.wires.push({
    id: crypto.randomUUID(),
    fromNode: a.id, fromPort: a.outputs[0].id,
    toNode: b.id,   toPort: b.inputs[0].id, kind:'exec',
  });
}

// ── TABS ──────────────────────────────────────────────────────────
function wireTabs() {
  document.querySelectorAll('.ptab').forEach(tab=>{
    tab.addEventListener('click',()=>{
      const t=(tab as HTMLElement).dataset.tab!;
      document.querySelectorAll('.ptab').forEach(x=>x.classList.remove('on'));
      document.querySelectorAll('.tab-pane').forEach(x=>x.classList.remove('on'));
      tab.classList.add('on'); document.getElementById(`tab-${t}`)!.classList.add('on');
    });
  });
}

// ── HIERARCHY ─────────────────────────────────────────────────────
const ICONS: Record<string,string> = { cube:'⬛',sphere:'🔵',plane:'▬',terrain:'🏔',cylinder:'🛢',cone:'🔺',mesh:'📦' };

function renderHierarchy() {
  const count=engine.scene.nodes.length;
  objCount.textContent=count>0?String(count):'';
  sceneList.innerHTML='';
  if(count===0){
    sceneList.innerHTML='<div style="padding:24px 14px;color:var(--dim);font-size:11px;text-align:center;line-height:1.8">No objects yet.<br>Click icons above<br>or use the toolbar.</div>';
    return;
  }
  for(const node of engine.scene.nodes){
    const layer=engine.scene.getLayer(node.layerId);
    const el=document.createElement('div');
    el.className='si'+(node.id===engine.selectedId?' sel':'');
    el.innerHTML=`<span class="ico">${ICONS[node.type]??'◻'}</span><span class="lbl">${node.name}</span><span class="dot" style="background:${layer?.color??'#888'}"></span><span class="sop" title="Toggle visibility">${node.visible?'👁':'🚫'}</span><span class="sop del" title="Delete">✕</span>`;
    el.querySelector('.sop.del')!.addEventListener('click',ev=>{
      ev.stopPropagation(); undoStack.push(node); if(undoStack.length>30) undoStack.shift();
      engine.removeObject(node.id); renderHierarchy(); renderInspector(null);
    });
    el.querySelectorAll('.sop:not(.del)')[0]?.addEventListener('click',ev=>{
      ev.stopPropagation(); node.visible=!node.visible; (ev.target as HTMLElement).textContent=node.visible?'👁':'🚫';
    });
    el.addEventListener('click',()=>{ if(node.locked) return; engine.selectedId=node.id; renderHierarchy(); renderInspector(node); });
    sceneList.appendChild(el);
  }
}

// ── LIGHT LIST ────────────────────────────────────────────────────
function renderLightList() {
  const list=document.getElementById('light-list')!;
  list.innerHTML='';
  for(const pl of engine.lights){
    const el=document.createElement('div');
    el.className='li';
    el.innerHTML=`<span class="lsw" style="background:rgb(${Math.round(pl.color.x*255)},${Math.round(pl.color.y*255)},${Math.round(pl.color.z*255)})"></span><span class="lnm">${pl.name}</span><div class="lops"><button class="lop ldel" data-id="${pl.id}" title="Delete">✕</button></div>`;
    el.querySelector('.ldel')!.addEventListener('click',ev=>{
      ev.stopPropagation(); engine.removePointLight((ev.target as HTMLElement).dataset.id!); renderLightList();
    });
    list.appendChild(el);
  }
}

// ── LAYERS ────────────────────────────────────────────────────────
function setupLayers() {
  activeLayerId=engine.scene.defaultLayerId; renderLayers();
  document.getElementById('btn-addlayer')!.addEventListener('click',()=>{
    const name=prompt('Layer name:','New Layer'); if(!name) return;
    const palette=['#7c5cfc','#fc5c7c','#3dffc0','#ffc83d','#3dc8ff','#fc7c5c','#c8fc3d'];
    const layer=engine.scene.addLayer({name,color:palette[engine.scene.layers.length%palette.length]});
    activeLayerId=layer.id; renderLayers();
  });
}

function renderLayers() {
  layerList.innerHTML='';
  for(let i=engine.scene.layers.length-1;i>=0;i--){
    const layer=engine.scene.layers[i];
    const wrap=document.createElement('div');
    const el=document.createElement('div');
    el.className='li'+(layer.id===activeLayerId?' act':'');
    el.innerHTML=`<span class="lsw" style="background:${layer.color}"></span><span class="lnm">${layer.name}</span><div class="lops"><button class="lop" data-a="vis" title="Visibility">${layer.visible?'👁':'🚫'}</button><button class="lop" data-a="lock" title="Lock">${layer.locked?'🔒':'🔓'}</button><button class="lop ldel" data-a="del" title="Delete">✕</button></div>`;
    el.addEventListener('click',ev=>{
      const a=(ev.target as HTMLElement).dataset.a;
      if(a==='vis'){layer.visible=!layer.visible;renderLayers();return;}
      if(a==='lock'){layer.locked=!layer.locked;renderLayers();return;}
      if(a==='del'){if(engine.scene.layers.length<=1){alert('Cannot delete last layer.');return;} engine.scene.removeLayer(layer.id);if(activeLayerId===layer.id)activeLayerId=engine.scene.defaultLayerId;renderLayers();return;}
      activeLayerId=layer.id; renderLayers();
    });
    const opRow=document.createElement('div'); opRow.className='lop-row';
    const slider=document.createElement('input'); slider.type='range';slider.min='0';slider.max='1';slider.step='0.01';slider.value=String(layer.opacity);
    const pct=document.createElement('span'); pct.textContent=`${Math.round(layer.opacity*100)}%`;
    slider.addEventListener('input',()=>{layer.opacity=parseFloat(slider.value);pct.textContent=`${Math.round(layer.opacity*100)}%`;});
    opRow.innerHTML='<span>Opacity</span>'; opRow.appendChild(slider); opRow.appendChild(pct);
    wrap.appendChild(el); wrap.appendChild(opRow); layerList.appendChild(wrap);
  }
}

// ── INSPECTOR ────────────────────────────────────────────────────
function renderInspector(node: SceneNode|null) {
  if(!node){
    inspType.textContent='';
    inspBody.innerHTML=`<div class="empty"><div class="ei">🎯</div>Select an object<br><span style="font-size:11px;color:var(--dim)">Click in the viewport<br>or use Place mode to add objects</span></div>`;
    return;
  }
  inspType.textContent=node.type.toUpperCase();
  inspBody.innerHTML='';

  const ni=crel('input') as HTMLInputElement;
  ni.className='ni'; ni.value=node.name;
  ni.addEventListener('input',()=>{node.name=ni.value;renderHierarchy();});
  inspBody.appendChild(ni);

  cdiv('pdiv');
  const qrow=crel('div') as HTMLDivElement; qrow.style.cssText='display:flex;gap:6px';
  const focBtn=ibtn('◎ Focus','focus'); focBtn.addEventListener('click',()=>{engine.selectedId=node.id;engine.focusSelected();});
  const dupBtn=ibtn('⧉ Dupe','dup');   dupBtn.addEventListener('click',()=>duplicateSelected());
  const delBtn=ibtn('🗑 Delete','del2'); delBtn.addEventListener('click',()=>deleteSelected());
  qrow.appendChild(focBtn); qrow.appendChild(dupBtn); qrow.appendChild(delBtn);
  inspBody.appendChild(qrow);

  cdiv('pdiv'); slbl('Transform');

  // Layer
  const pg0=crel('div') as HTMLDivElement; pg0.className='pg';
  const lb0=crel('label') as HTMLLabelElement; lb0.textContent='Layer'; pg0.appendChild(lb0);
  const lasel=crel('select') as HTMLSelectElement; lasel.className='lasel';
  for(const l of engine.scene.layers){const opt=crel('option') as HTMLOptionElement;opt.value=l.id;opt.textContent=l.name;if(l.id===node.layerId)opt.selected=true;lasel.appendChild(opt);}
  lasel.addEventListener('change',()=>{engine.scene.moveNodeToLayer(node.id,lasel.value);renderHierarchy();});
  pg0.appendChild(lasel); inspBody.appendChild(pg0);

  vec3('Position',node.position,v=>{node.position=v;engine.updateNode(node);});
  vec3('Rotation',node.rotation,v=>{node.rotation=v;engine.updateNode(node);});
  vec3('Scale',   node.scale,   v=>{node.scale=v;   engine.updateNode(node);});

  cdiv('pdiv'); slbl('Appearance');
  colorF('Color',node.color,c=>{node.color=c;engine.updateNode(node);});

  cdiv('pdiv'); slbl('Material (PBR)');
  sliderF('Metalness', node.metalness, 0, 1, 0.01, v=>{node.metalness=v;engine.updateNode(node);});
  sliderF('Roughness', node.roughness, 0, 1, 0.01, v=>{node.roughness=v;engine.updateNode(node);});
  sliderF('Emissive',  node.emissive,  0, 5, 0.1,  v=>{node.emissive=v; engine.updateNode(node);});
  togF('Textured surface', node.textured, v=>{node.textured=v;engine.updateNode(node);});

  cdiv('pdiv'); slbl('Gameplay Role');
  const grPg=crel('div') as HTMLDivElement; grPg.className='pg';
  const grLbl=crel('label') as HTMLLabelElement; grLbl.textContent='When you press Play'; grPg.appendChild(grLbl);
  const grSel=crel('select') as HTMLSelectElement; grSel.className='lasel';
  const roles:[string,string][]=[['none','— None —'],['collectible','🪙 Collectible (+score)'],['checkpoint','🚩 Checkpoint'],['hazard','💥 Hazard (-health)'],['goal','🏁 Goal (win)']];
  for(const [val,lbl] of roles){const o=crel('option') as HTMLOptionElement;o.value=val;o.textContent=lbl;if(node.gameRole===val)o.selected=true;grSel.appendChild(o);}
  grSel.addEventListener('change',()=>{ node.gameRole=grSel.value as any; });
  grPg.appendChild(grSel); inspBody.appendChild(grPg);

  cdiv('pdiv'); slbl('Visibility');
  togF('Visible',node.visible,v=>{node.visible=v;renderHierarchy();});
  togF('Locked', node.locked, v=>{node.locked=v; renderHierarchy();});

  cdiv('pdiv'); slbl('Physics');
  const physBox=crel('div') as HTMLDivElement; physBox.className='phys-box';
  const phRow=crel('div') as HTMLDivElement; phRow.className='trow';
  const phLbl=crel('span') as HTMLSpanElement; phLbl.textContent='Enable Physics';
  const phTog=togEl(node.hasPhysics,v=>{node.hasPhysics=v;physDetails.style.display=v?'flex':'none';});
  phRow.appendChild(phLbl); phRow.appendChild(phTog); physBox.appendChild(phRow);
  const physDetails=crel('div') as HTMLDivElement;
  physDetails.style.cssText=`display:${node.hasPhysics?'flex':'none'};flex-direction:column;gap:7px;padding-top:4px`;
  const stRow=crel('div') as HTMLDivElement; stRow.className='trow';
  const stLbl=crel('span') as HTMLSpanElement; stLbl.textContent='Static Body';
  const stTog=togEl(node.isStatic,v=>{node.isStatic=v;});
  stRow.appendChild(stLbl); stRow.appendChild(stTog); physDetails.appendChild(stRow);
  physBox.appendChild(physDetails); inspBody.appendChild(physBox);
  const applyPhys=ibtn(node.hasPhysics?'⚡ Update Physics Body':'⚡ Enable Physics Body','phy');
  applyPhys.addEventListener('click',()=>{engine.enablePhysics(node);applyPhys.textContent='✓ Physics Active';});
  inspBody.appendChild(applyPhys);

  // ── Helpers ──────────────────────────────────────────────────
  function crel(t: string){ return document.createElement(t); }
  function cdiv(cls: string){ const d=crel('div') as HTMLDivElement;d.className=cls;inspBody.appendChild(d); }
  function slbl(txt: string){ const d=crel('div') as HTMLDivElement;d.className='slbl';d.textContent=txt;inspBody.appendChild(d); }
  function ibtn(txt: string,cls: string):HTMLButtonElement{ const b=crel('button') as HTMLButtonElement;b.className=`ibtn ${cls}`;b.textContent=txt;return b; }

  function vec3(label: string, vec: Vec3, cb:(v:Vec3)=>void){
    const g=crel('div') as HTMLDivElement;g.className='pg';
    const lb=crel('label') as HTMLLabelElement;lb.textContent=label;g.appendChild(lb);
    const row=crel('div') as HTMLDivElement;row.className='v3r';
    const fmt=(n:number)=>parseFloat(n.toFixed(3)).toString();
    for(const axis of ['x','y','z'] as const){
      const wrap=crel('div') as HTMLDivElement;wrap.className='axw';
      const axl=crel('span') as HTMLSpanElement;axl.className='axl';axl.textContent=axis.toUpperCase();
      const inp=crel('input') as HTMLInputElement;inp.type='number';inp.className='ni2';inp.step='0.1';inp.value=fmt(vec[axis]);
      inp.addEventListener('input',()=>{const v=vec.clone();v[axis]=parseFloat(inp.value)||0;cb(v);});
      let drag=false,sy=0,sv=0;
      axl.addEventListener('mousedown',e=>{drag=true;sy=e.clientY;sv=parseFloat(inp.value)||0;e.preventDefault();});
      window.addEventListener('mousemove',e=>{if(!drag)return;const nv=parseFloat((sv+(sy-e.clientY)*0.05).toFixed(2));inp.value=String(nv);const v=vec.clone();v[axis]=nv;cb(v);});
      window.addEventListener('mouseup',()=>{drag=false;});
      wrap.appendChild(axl);wrap.appendChild(inp);row.appendChild(wrap);
    }
    g.appendChild(row);inspBody.appendChild(g);
  }

  function colorF(label: string, color:[number,number,number], cb:(c:[number,number,number])=>void){
    const g=crel('div') as HTMLDivElement;g.className='pg';
    const lb=crel('label') as HTMLLabelElement;lb.textContent=label;g.appendChild(lb);
    const row=crel('div') as HTMLDivElement;row.className='crow';
    const toHex=(c:[number,number,number])=>'#'+c.map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('');
    const fromHex=(h:string):[number,number,number]=>[parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];
    const pk=crel('input') as HTMLInputElement;pk.type='color';pk.className='cpk';pk.value=toHex(color);
    const hx=crel('input') as HTMLInputElement;hx.type='text';hx.className='chx';hx.value=toHex(color);hx.maxLength=7;
    pk.addEventListener('input',()=>{hx.value=pk.value;cb(fromHex(pk.value));});
    hx.addEventListener('input',()=>{if(/^#[0-9a-fA-F]{6}$/.test(hx.value)){pk.value=hx.value;cb(fromHex(hx.value));}});
    row.appendChild(pk);row.appendChild(hx);g.appendChild(row);inspBody.appendChild(g);
  }

  function sliderF(label: string, val: number, mn: number, mx: number, step: number, cb:(v:number)=>void){
    const g=crel('div') as HTMLDivElement;g.className='pg';
    const lb=crel('label') as HTMLLabelElement;lb.textContent=label;g.appendChild(lb);
    const row=crel('div') as HTMLDivElement;row.style.cssText='display:flex;gap:7px;align-items:center';
    const sl=crel('input') as HTMLInputElement;sl.type='range';sl.min=String(mn);sl.max=String(mx);sl.step=String(step);sl.value=String(val);sl.style.cssText='flex:1;accent-color:var(--accent);height:3px';
    const disp=crel('span') as HTMLSpanElement;disp.className='axl';disp.textContent=val.toFixed(2);disp.style.minWidth='32px';
    sl.addEventListener('input',()=>{const v=parseFloat(sl.value);disp.textContent=v.toFixed(2);cb(v);});
    row.appendChild(sl);row.appendChild(disp);g.appendChild(row);inspBody.appendChild(g);
  }

  function togF(label: string, val:boolean, cb:(v:boolean)=>void){
    const row=crel('div') as HTMLDivElement;row.className='trow';
    const sp=crel('span') as HTMLSpanElement;sp.textContent=label;
    const tog=togEl(val,cb); row.appendChild(sp);row.appendChild(tog);inspBody.appendChild(row);
  }
  function togEl(val:boolean,cb:(v:boolean)=>void):HTMLElement{
    const lbl=crel('label') as HTMLLabelElement;lbl.className='tog';
    const inp=crel('input') as HTMLInputElement;inp.type='checkbox';inp.checked=val;
    const sl=crel('span') as HTMLSpanElement;sl.className='tslider';
    inp.addEventListener('change',()=>cb(inp.checked));
    lbl.appendChild(inp);lbl.appendChild(sl);return lbl;
  }
}
