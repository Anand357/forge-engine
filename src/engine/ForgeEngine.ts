import { Renderer, RenderObject } from './renderer/Renderer';
import { GizmoRenderer, GizmoAxis } from './renderer/GizmoRenderer';
import { Scene, SceneNode, ObjType } from './scene/Scene';
import { Camera } from './scene/Camera';
import { PhysicsWorld } from './physics/Physics';
import { PointLight } from './scene/Light';
import { Vec3 } from './math/Vec3';
import { Mat4 } from './math/Mat4';
import { Terrain } from './scene/Terrain';
import { PlayerController, PlayerInput } from './scene/PlayerController';
import { VehicleController, VehicleInput } from './scene/VehicleController';
import { CollisionWorld } from './scene/Collision';
import { GameState, GameSnapshot } from './scene/GameState';
import { buildAssetLibrary } from './renderer/AssetLibrary';
import { createCube, createSphere, createPlane, createTerrain, createCylinder, createCone } from './renderer/Geometry';

export class ForgeEngine {
  renderer: Renderer;
  scene:    Scene;
  camera:   Camera;
  physics:  PhysicsWorld;
  gizmo:    GizmoRenderer;

  private ros = new Map<string, RenderObject>();
  private running  = false;
  private playMode = false;
  private time     = 0;

  // Editable heightfield terrain (separate from scene nodes)
  terrain:   Terrain = new Terrain(220, 161);
  private terrainRO?: RenderObject;
  terrainColor: [number,number,number] = [0.34, 0.52, 0.26];

  showGrid    = true;
  selectedId: string | null = null;
  gizmoAxis:  GizmoAxis = 'none';
  lights:     PointLight[] = [];

  // ── Play mode / 3rd-person player ──────────────────────────────
  player?: PlayerController;
  private playerNode?: SceneNode;
  playerInput: PlayerInput = { forward:false, back:false, left:false, right:false, sprint:false, jumpPressed:false };
  playerAssetName = 'Player (Hero)';   // which library asset represents the player
  onPlayStateChange?: (playing: boolean) => void;
  get isPlaying() { return this.playMode; }

  // ── Collision ──────────────────────────────────────────────────
  collisionWorld = new CollisionWorld();

  // ── Gameplay ───────────────────────────────────────────────────
  game = new GameState();
  onGameUpdate?: (snap: GameSnapshot) => void;

  // ── Driving ────────────────────────────────────────────────────
  vehicle?: VehicleController;
  private vehicleNode?: SceneNode;      // the car the player is currently driving
  vehicleInput: VehicleInput = { throttle:false, brake:false, steerLeft:false, steerRight:false, handbrake:false };
  get isDriving() { return !!this.vehicle; }
  onVehicleStateChange?: (driving: boolean, prompt: string) => void;
  // Vehicle scene nodes are tagged so we know what's drivable.
  private isVehicleNode(n: SceneNode): boolean {
    return /car|truck|suv|bike|vehicle|jeep|sports/i.test(n.name);
  }

  // Sun
  sunDir   = new Vec3(0.6, 0.8, 0.4).normalize();
  sunColor = new Vec3(1.0, 0.95, 0.85);

  // Time of day (0..1 = midnight→noon→midnight) + weather
  timeOfDay  = 0.35;        // ~morning
  daySpeed   = 0;           // hours per second; 0 = paused
  wetness    = 0;           // 0 dry, 1 fully wet (rain)
  autoSun    = true;        // when true, sunDir/Color computed from timeOfDay

  // Recompute sun direction & color from timeOfDay.
  updateDayNight(dt: number) {
    if (this.daySpeed !== 0) {
      this.timeOfDay = (this.timeOfDay + dt * this.daySpeed / 24) % 1;
      if (this.timeOfDay < 0) this.timeOfDay += 1;
    }
    if (!this.autoSun) return;
    // sun arcs across the sky: angle 0 at dawn (east), PI at dusk (west)
    const a = (this.timeOfDay - 0.25) * Math.PI * 2;   // noon at 0.5
    const elev = Math.sin(this.timeOfDay * Math.PI);    // 0 at midnight, 1 at noon
    const sx = Math.cos(a) * 0.9;
    const sy = Math.max(-0.15, elev * 1.1 - 0.05);
    const sz = 0.35;
    this.sunDir = new Vec3(sx, sy, sz).normalize();
    // color: warm at dawn/dusk, white at noon, dim blue at night
    const noon = Math.max(0, elev);
    const dawn = 1 - Math.min(1, Math.abs(this.timeOfDay - 0.25) * 6);   // peak near dawn
    const dusk = 1 - Math.min(1, Math.abs(this.timeOfDay - 0.75) * 6);
    const warm = Math.max(dawn, dusk);
    const day = new Vec3(1.0, 0.96, 0.88).mul(noon);
    const golden = new Vec3(1.0, 0.55, 0.28).mul(warm * 0.8);
    const night = new Vec3(0.10, 0.14, 0.26);
    const lit = day.add(golden);
    const k = Math.max(noon, warm);
    this.sunColor = night.mul(1 - k).add(lit);
  }

  onFrame?:  (dt: number) => void;
  onSelect?: (id: string | null) => void;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new Renderer(canvas);
    this.scene    = new Scene();
    this.camera   = new Camera();
    this.physics  = new PhysicsWorld();
    this.gizmo    = new GizmoRenderer(undefined as any); // device set in init()
  }

  async init(): Promise<boolean> {
    const canvas = this.renderer.canvas;
    const wrap   = canvas.parentElement!;
    canvas.width  = Math.max(1, wrap.clientWidth)  * devicePixelRatio;
    canvas.height = Math.max(1, wrap.clientHeight) * devicePixelRatio;

    const ok = await this.renderer.init();
    if (!ok) return false;
    await this.physics.init();
    this.camera.attachControls(canvas);

    // Init gizmo renderer now that device is ready
    (this.gizmo as any).device = this.renderer.device;
    this.gizmo.build(this.renderer['fmt'] as GPUTextureFormat);

    new ResizeObserver(() => {
      const w = Math.max(1, Math.floor(wrap.clientWidth  * devicePixelRatio));
      const h = Math.max(1, Math.floor(wrap.clientHeight * devicePixelRatio));
      this.renderer.resize(w, h);
    }).observe(wrap);

    this._setupClick(canvas);
    this._buildTerrainRO();
    return true;
  }

  // ── Editable terrain ───────────────────────────────────────────
  private _buildTerrainRO() {
    const mesh = this.terrain.buildMesh();
    const { uniformBuffer, bindGroup, shadowUniformBuffer, shadowBindGroup, prepassBindGroup } =
      this.renderer.mkUniformResources();
    this.terrainRO = {
      vertexBuffer:        this.renderer.mkVBuf(mesh.vertices),
      indexBuffer:         this.renderer.mkIBuf32(mesh.indices),
      indexFormat:         'uint32',
      uniformBuffer, bindGroup,
      shadowUniformBuffer, shadowBindGroup, prepassBindGroup,
      indexCount:          mesh.indices.length,
      modelMatrix:         new Mat4(),
      color:               this.terrainColor,
      metalness:           0.0,
      roughness:           0.92,
      emissive:            0.0,
      opacity:             1,
      selected:            false,
      textured:            1,   // procedural surface detail on by default
      material:            1,   // natural terrain material blend
    };
  }

  // Rebuild terrain GPU geometry after sculpt/preset edits.
  rebuildTerrain() {
    if (!this.terrainRO) return;
    const mesh = this.terrain.buildMesh();
    this.terrainRO.vertexBuffer.destroy();
    this.terrainRO.indexBuffer.destroy();
    this.terrainRO.vertexBuffer = this.renderer.mkVBuf(mesh.vertices);
    this.terrainRO.indexBuffer  = this.renderer.mkIBuf32(mesh.indices);
    this.terrainRO.indexCount   = mesh.indices.length;
  }

  sculptTerrain(wx: number, wz: number, radius: number, strength: number, mode: number) {
    this.terrain.sculpt(wx, wz, radius, strength, mode);
    this.rebuildTerrain();
  }

  addTerrainFeature(kind: 'mountain'|'hills'|'canyon'|'plateau', wx: number, wz: number, scale: number) {
    this.terrain.stampFeature(kind, wx, wz, scale);
    this.rebuildTerrain();
  }

  flattenTerrain() { this.terrain.flatten(); this.rebuildTerrain(); }

  groundHeightAt(wx: number, wz: number): number { return this.terrain.heightAt(wx, wz); }

  // Raycast the terrain from a screen ray; returns world hit or null.
  raycastTerrain(nx: number, ny: number): Vec3 | null {
    const aspect = this.renderer.canvas.width / this.renderer.canvas.height;
    const { origin, dir } = this.camera.getRay(nx, ny, aspect);
    const hit = this.terrain.raycast(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z);
    return hit ? new Vec3(hit.x, hit.y, hit.z) : null;
  }

  // ── Click / ray ───────────────────────────────────────────────
  private _setupClick(canvas: HTMLCanvasElement) {
    let mdX=0, mdY=0;
    canvas.addEventListener('mousedown', e => { if(e.button===0){mdX=e.clientX;mdY=e.clientY;} });
    canvas.addEventListener('mouseup', e => {
      if (e.button!==0) return;
      if (Math.abs(e.clientX-mdX)>5||Math.abs(e.clientY-mdY)>5) return;
      const rect = canvas.getBoundingClientRect();
      const nx = ((e.clientX-rect.left)/rect.width)*2-1;
      const ny = -(((e.clientY-rect.top)/rect.height)*2-1);
      this.raycastSelect(nx, ny);
    });
  }

  raycastSelect(nx: number, ny: number) {
    const aspect = this.renderer.canvas.width / this.renderer.canvas.height;
    const { origin, dir } = this.camera.getRay(nx, ny, aspect);
    let best: string|null=null, bestT=Infinity;
    for (const node of this.scene.visibleNodes()) {
      const t = this._rayAABB(origin, dir, node);
      if (t>0&&t<bestT) { bestT=t; best=node.id; }
    }
    this.selectedId = best;
    this.onSelect?.(best);
  }

  private _rayAABB(ro: Vec3, rd: Vec3, node: SceneNode): number {
    const p=node.position, s=node.scale;
    const hs = new Vec3(s.x*0.65, s.y*0.65, s.z*0.65);
    const mn=p.sub(hs), mx=p.add(hs);
    let tmin=-Infinity, tmax=Infinity;
    for (const ax of ['x','y','z'] as const) {
      const d=rd[ax]; if(Math.abs(d)<1e-8) continue;
      const t1=(mn[ax]-ro[ax])/d, t2=(mx[ax]-ro[ax])/d;
      tmin=Math.max(tmin,Math.min(t1,t2));
      tmax=Math.min(tmax,Math.max(t1,t2));
    }
    return tmin<=tmax&&tmax>0 ? tmin : -1;
  }

  placeOnTerrain(nx: number, ny: number): Vec3 {
    // Prefer a real heightfield hit; fall back to the y=0 plane.
    const hit = this.raycastTerrain(nx, ny);
    if (hit) return hit;
    const aspect = this.renderer.canvas.width/this.renderer.canvas.height;
    const { origin, dir } = this.camera.getRay(nx, ny, aspect);
    if (Math.abs(dir.y)<1e-6) return new Vec3(0,0,0);
    const t = -origin.y/dir.y;
    return origin.add(dir.mul(Math.max(0,t)));
  }

  // ── Object management ─────────────────────────────────────────
  addObject(type: ObjType, name?: string, layerId?: string): SceneNode {
    const node = new SceneNode(name ?? `${type}_${this.scene.nodes.length+1}`, type);
    node.layerId = layerId ?? this.scene.defaultLayerId;
    const colors: Partial<Record<ObjType,[number,number,number]>> = {
      terrain:[0.30,0.58,0.22], sphere:[0.22,0.52,0.98],
      cube:[0.92,0.38,0.18],    plane:[0.48,0.48,0.55],
      cylinder:[0.18,0.72,0.82],cone:[0.82,0.28,0.72],
    };
    node.color = colors[type] ?? [0.8,0.8,0.8];
    // Large natural surfaces get procedural texture detail by default
    if (type==='terrain' || type==='plane') node.textured = true;
    this.scene.add(node);
    this._buildRO(node);
    return node;
  }

  private _buildRO(node: SceneNode, customGeo?: {vertices:Float32Array;indices:Uint16Array}) {
    const geo: Partial<Record<ObjType,()=>{vertices:Float32Array;indices:Uint16Array}>> = {
      cube:     ()=>createCube(1),
      sphere:   ()=>createSphere(1,48),
      plane:    ()=>createPlane(20,20,32,32),
      terrain:  ()=>createTerrain(80,80,100),
      cylinder: ()=>createCylinder(0.5,2,24),
      cone:     ()=>createCone(0.7,2,24),
    };
    let g: {vertices:Float32Array;indices:Uint16Array};
    if (customGeo) g = customGeo;
    else { const fn=geo[node.type]; if(!fn) return; g=fn(); }
    const { uniformBuffer, bindGroup, shadowUniformBuffer, shadowBindGroup, prepassBindGroup } = this.renderer.mkUniformResources();
    this.ros.set(node.id, {
      vertexBuffer:        this.renderer.mkVBuf(g.vertices),
      indexBuffer:         this.renderer.mkIBuf(g.indices),
      uniformBuffer, bindGroup,
      shadowUniformBuffer, shadowBindGroup, prepassBindGroup,
      indexCount:          g.indices.length,
      modelMatrix:         node.getModelMatrix(),
      color:               node.color,
      metalness:           node.metalness,
      roughness:           node.roughness,
      emissive:            node.emissive,
      opacity:             1,
      selected:            false,
      textured:            node.textured ? 1 : 0,
      material:            node.isWater ? 2 : 0,
    });
  }

  updateNode(node: SceneNode) {
    const ro = this.ros.get(node.id);
    if (ro) {
      ro.modelMatrix = node.getModelMatrix();
      ro.color    = node.color;
      ro.metalness = node.metalness;
      ro.roughness = node.roughness;
      ro.emissive  = node.emissive;
      ro.textured  = node.textured ? 1 : 0;
      ro.material  = node.isWater ? 2 : 0;
    }
  }

  removeObject(id: string) {
    const ro = this.ros.get(id);
    if (ro) {
      ro.vertexBuffer.destroy();
      ro.indexBuffer.destroy();
      ro.uniformBuffer.destroy();
      ro.shadowUniformBuffer.destroy();
      this.ros.delete(id);
    }
    this.scene.remove(id);
    this.physics.removeBody(id);
    if (this.selectedId===id) { this.selectedId=null; this.onSelect?.(null); }
  }

  restoreNode(node: SceneNode) {
    this.scene.add(node);
    this._buildRO(node, node.meshGeo);
  }

  // Add an imported glTF/.glb mesh as a scene node.
  addMesh(geo: {vertices:Float32Array;indices:Uint16Array}, name?: string, layerId?: string): SceneNode {
    const node = new SceneNode(name ?? `mesh_${this.scene.nodes.length+1}`, 'mesh');
    node.layerId = layerId ?? this.scene.defaultLayerId;
    node.color = [0.78, 0.78, 0.82];
    node.roughness = 0.55;
    node.meshGeo = geo;
    this.scene.add(node);
    this._buildRO(node, geo);
    return node;
  }

  focusSelected() {
    const node = this.selectedId ? this.scene.get(this.selectedId) : null;
    if (node) this.camera.focusOn(node.position);
  }

  enablePhysics(node: SceneNode) { node.hasPhysics=true; this.physics.addBody(node); }

  startPlay() {
    this.playMode = true;
    this.camera.locked = true;
    for (const n of this.scene.nodes) if (n.hasPhysics) this.physics.addBody(n);
    this.selectedId = null; this.onSelect?.(null);

    // Build solid-collision boxes from current scene objects (exclude water).
    this.rebuildCollision();

    // Start gameplay tracking (collectibles/health/score/objectives).
    this.game.start(this.scene.nodes);
    this.onGameUpdate?.(this.game.snapshot());

    // Spawn the player at world center on the terrain.
    const start = new Vec3(0, 0, 0);
    this.player = new PlayerController((x,z)=>this.groundHeightAt(x,z), start);
    this.player.collide   = (from,to,r,h) => this.collisionWorld.resolveMove(from,to,r,h);
    this.player.supportAt = (x,z,r)       => this.collisionWorld.supportHeightAt(x,z,r);
    // align the follow-camera behind the current editor view direction
    this.player.camYaw = Math.atan2(
      this.camera.position.x - this.camera.target.x,
      this.camera.position.z - this.camera.target.z,
    );

    // Build the player character mesh from the asset library.
    const lib = buildAssetLibrary();
    const def = lib.find(a => a.name === this.playerAssetName) ?? lib[0];
    this.playerNode = this.addMesh(def.build(), '▶ Player', this.scene.defaultLayerId);
    this.playerNode.color = [...def.color] as [number,number,number];
    this.playerNode.roughness = def.roughness;
    this.playerNode.metalness = def.metalness;
    this.playerNode.position = this.player.position.clone();

    this.onPlayStateChange?.(true);
  }

  // Build collision boxes from scene objects (skip water planes and the player).
  rebuildCollision() {
    this.collisionWorld.rebuild(this.scene.nodes.map(n => ({
      id: n.id, position: n.position, scale: n.scale,
      skip: n.isWater || n.id === this.playerNode?.id || /water/i.test(n.name),
    })));
  }

  stopPlay() {
    this.playMode = false;
    this.camera.locked = false;
    this.game.stop();
    this.vehicle = undefined; this.vehicleNode = undefined;
    if (this.playerNode) { this.removeObject(this.playerNode.id); this.playerNode = undefined; }
    this.player = undefined;
    this.onPlayStateChange?.(false);
  }

  // Nearest drivable vehicle node within range of the player (for the E prompt).
  private _nearestVehicle(range = 4): SceneNode | null {
    if (!this.player) return null;
    let best: SceneNode | null = null, bestD = range * range;
    for (const n of this.scene.nodes) {
      if (!this.isVehicleNode(n)) continue;
      const dx = n.position.x - this.player.position.x;
      const dz = n.position.z - this.player.position.z;
      const d2 = dx*dx + dz*dz;
      if (d2 < bestD) { bestD = d2; best = n; }
    }
    return best;
  }

  // Toggle enter/exit nearest vehicle (bound to E in Play mode).
  toggleVehicle() {
    if (!this.player) return;
    if (this.vehicle) { this._exitVehicle(); return; }
    const car = this._nearestVehicle();
    if (car) this._enterVehicle(car);
  }

  private _enterVehicle(node: SceneNode) {
    if (!this.player) return;
    this.vehicleNode = node;
    this.vehicle = new VehicleController(
      (x,z)=>this.groundHeightAt(x,z),
      node.position.clone(),
      node.rotation.y,
    );
    // collide against everything except the car we're driving
    this.vehicle.collide = (from,to,r,h) => {
      const saved = this.collisionWorld.boxes.filter(b => b.id !== node.id);
      const tmp = this.collisionWorld.boxes;
      this.collisionWorld.boxes = saved;
      const res = this.collisionWorld.resolveMove(from,to,r,h);
      this.collisionWorld.boxes = tmp;
      return res;
    };
    this.vehicle.camYaw = node.rotation.y;
    // hide the on-foot character while driving
    if (this.playerNode) this.playerNode.visible = false;
    this.onVehicleStateChange?.(true, '');
  }

  private _exitVehicle() {
    if (!this.vehicle || !this.vehicleNode || !this.player) return;
    // drop the player beside the car
    const right = new Vec3(Math.cos(this.vehicle.yaw), 0, -Math.sin(this.vehicle.yaw));
    const exitPos = this.vehicle.position.add(right.mul(2.2));
    this.player.position = new Vec3(exitPos.x, this.groundHeightAt(exitPos.x, exitPos.z), exitPos.z);
    this.player.camYaw = this.vehicle.camYaw;
    if (this.playerNode) this.playerNode.visible = true;
    this.vehicle = undefined; this.vehicleNode = undefined;
    this.onVehicleStateChange?.(false, '');
  }

  private _updatePlayer(dt: number) {
    if (!this.player || !this.playerNode) return;

    // ── Gameplay tick (collectibles / hazards / objectives) ──
    if (this.game.active) {
      const ppos = this.vehicle ? this.vehicle.position : this.player.position;
      this.game.update(dt, ppos, this.scene.nodes, (id) => {
        this.removeObject(id);
        this.collisionWorld.boxes = this.collisionWorld.boxes.filter(b => b.id !== id);
      });
      // emit every frame so the HUD timer/health stays live (cheap)
      this.onGameUpdate?.(this.game.snapshot());
    }

    // ── Driving ──
    if (this.vehicle && this.vehicleNode) {
      this.vehicle.update(dt, this.vehicleInput);
      this.vehicleNode.position = this.vehicle.position.clone();
      this.vehicleNode.rotation = new Vec3(0, this.vehicle.yaw, 0);
      this.updateNode(this.vehicleNode);
      // keep the (hidden) player attached to the car
      this.player.position = this.vehicle.position.clone();
      const { eye, target } = this.vehicle.cameraTransform();
      this.camera.setView(eye, target);
      return;
    }

    // ── On-foot ──
    this.player.update(dt, this.playerInput);
    this.playerInput.jumpPressed = false; // consume edge
    this.playerNode.position = this.player.position.clone();
    this.playerNode.rotation = new Vec3(0, this.player.yaw, 0);
    this.updateNode(this.playerNode);

    // Update the "Press E to enter" prompt
    const near = this._nearestVehicle();
    this.onVehicleStateChange?.(false, near ? `Press E to drive ${near.name}` : '');

    const { eye, target } = this.player.cameraTransform();
    this.camera.setView(eye, target);
  }

  // ── Light management ──────────────────────────────────────────
  addPointLight(): PointLight {
    const pl: PointLight = {
      id: crypto.randomUUID(), name:`Light_${this.lights.length+1}`,
      position: new Vec3(0,5,0), color: new Vec3(1,1,1), intensity:3, radius:10,
    };
    this.lights.push(pl); return pl;
  }
  removePointLight(id: string) { this.lights=this.lights.filter(l=>l.id!==id); }

  // ── Render loop ───────────────────────────────────────────────
  start() {
    if (this.running) return;
    this.running = true;
    let last = performance.now();
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = Math.min((now-last)/1000, 0.05);
      last=now; this.time+=dt;
      this.updateDayNight(dt);
      if (this.playMode) {
        this._updatePlayer(dt);
        this.physics.step(dt);
      } else {
        this.camera.update(dt);
      }
      this.physics.syncToScene(this.scene.nodes);
      this.onFrame?.(dt);
      this._frame();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() { this.running=false; }

  private _frame() {
    const { width, height } = this.renderer.canvas;
    if (width<1||height<1) return;
    const aspect = width/height;

    // Camera basis vectors for sky ray reconstruction
    const fwd = this.camera.target.sub(this.camera.position).normalize();
    const rgt = fwd.cross(Vec3.up()).normalize();
    const upV = rgt.cross(fwd);
    const tanH = Math.tan(this.camera.fov/2);
    const camRight   = rgt.mul(tanH*aspect);
    const camUp      = upV.mul(tanH);

    // Compute light view-projection (orthographic from sun)
    const sunPos  = this.sunDir.mul(-120);
    const lightUp = Math.abs(this.sunDir.dot(Vec3.up()))>0.95 ? new Vec3(1,0,0) : Vec3.up();
    const lightView = Mat4.lookAt(sunPos, Vec3.zero(), lightUp);
    const orthoSize = 70;
    const lightProj = Mat4.orthographic(-orthoSize,orthoSize,-orthoSize,orthoSize,1,240);
    const lightVP   = lightProj.multiply(lightView);

    const viewProj = this.camera.getVP(aspect);
    const viewMat  = this.camera.getViewMatrix();
    const objects: RenderObject[] = [];
    // Terrain renders first (it's the ground; not a selectable scene node)
    if (this.terrainRO) {
      this.terrainRO.color    = this.terrainColor;
      this.terrainRO.textured = 1;
      objects.push(this.terrainRO);
    }
    for (const node of this.scene.visibleNodes()) {
      const ro    = this.ros.get(node.id);
      const layer = this.scene.getLayer(node.layerId);
      if (ro && layer) {
        ro.modelMatrix = node.getModelMatrix();
        ro.color       = node.color;
        ro.metalness   = node.metalness;
        ro.roughness   = node.roughness;
        ro.emissive    = node.emissive;
        ro.textured    = node.textured ? 1 : 0;
        ro.material    = node.isWater ? 2 : 0;
        ro.opacity     = layer.opacity;
        ro.selected    = node.id===this.selectedId;
        objects.push(ro);
      }
    }

    this.renderer.render(
      objects, viewProj, lightVP, viewMat,
      this.camera.position, camRight, camUp, fwd,
      this.sunDir, this.sunColor,
      this.lights, this.time, this.showGrid,
    );

    // Gizmo pass (separate render pass on top)
    if (this.selectedId) {
      const node = this.scene.get(this.selectedId);
      if (node) this._renderGizmoPass(node.position, viewProj);
    }
  }

  private _renderGizmoPass(pos: Vec3, viewProj: Mat4) {
    const dev = this.renderer.device;
    const ctx = this.renderer['ctx'] as GPUCanvasContext;
    const enc = dev.createCommandEncoder({ label:'gizmo' });
    const pass = enc.beginRenderPass({
      colorAttachments:[{
        view: ctx.getCurrentTexture().createView(),
        loadOp:'load', storeOp:'store',
      }]
    });
    this.gizmo.render(pass, pos, this.camera, viewProj, this.gizmoAxis);
    pass.end();
    dev.queue.submit([enc.finish()]);
  }
}
