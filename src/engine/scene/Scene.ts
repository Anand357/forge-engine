import { Vec3 } from '../math/Vec3';
import { Mat4 } from '../math/Mat4';

export type ObjType = 'cube'|'sphere'|'plane'|'terrain'|'cylinder'|'cone'|'mesh';

export interface Layer {
  id: string; name: string;
  visible: boolean; locked: boolean;
  opacity: number; color: string;
}

export class SceneNode {
  id = crypto.randomUUID();
  name: string; type: ObjType;
  layerId = 'default';
  position = Vec3.zero();
  rotation = Vec3.zero();
  scale    = new Vec3(1, 1, 1);
  color:   [number,number,number] = [0.8, 0.8, 0.8];
  metalness  = 0.0;
  roughness  = 0.5;
  emissive   = 0.0;
  textured   = false;   // triplanar procedural surface detail
  isWater    = false;   // animated reflective water material
  gameRole: 'none'|'collectible'|'checkpoint'|'hazard'|'goal' = 'none';
  visible  = true;
  locked   = false;
  hasPhysics = false;
  isStatic   = true;
  mass       = 1;
  physicsHandle: any = null;
  // For imported glTF/.glb meshes: pre-built interleaved geometry
  meshGeo?: { vertices: Float32Array; indices: Uint16Array };

  constructor(name: string, type: ObjType) { this.name = name; this.type = type; }

  getModelMatrix(): Mat4 {
    // TRS: Translation * RotY * RotX * RotZ * Scale
    const T  = Mat4.translation(this.position);
    const Ry = Mat4.rotationY(this.rotation.y);
    const Rx = Mat4.rotationX(this.rotation.x);
    const Rz = Mat4.rotationZ(this.rotation.z);
    const S  = Mat4.scale(this.scale);
    return T.multiply(Ry).multiply(Rx).multiply(Rz).multiply(S);
  }
}

export class Scene {
  nodes:  SceneNode[] = [];
  layers: Layer[]     = [];
  worldW = 100; worldD = 100;
  worldType: 'finite'|'infinite' = 'finite';
  private nmap = new Map<string, SceneNode>();
  private lmap = new Map<string, Layer>();

  constructor() {
    this._addLayer('Default', '#7c5cfc');
    this._addLayer('Terrain', '#3dffc0');
    this._addLayer('Objects', '#fc5c7c');
  }

  private _addLayer(name: string, color: string): Layer {
    const l: Layer = { id: crypto.randomUUID(), name, visible: true, locked: false, opacity: 1, color };
    this.layers.push(l); this.lmap.set(l.id, l); return l;
  }

  addLayer(opts: { name: string; color?: string }): Layer {
    return this._addLayer(opts.name, opts.color ?? '#888888');
  }

  removeLayer(id: string) {
    this.layers = this.layers.filter(l => l.id !== id);
    this.lmap.delete(id);
    const def = this.layers[0];
    if (def) this.nodes.forEach(n => { if (n.layerId === id) n.layerId = def.id; });
  }

  getLayer(id: string) { return this.lmap.get(id); }
  get defaultLayerId() { return this.layers[0]?.id ?? ''; }

  add(n: SceneNode)    { this.nodes.push(n); this.nmap.set(n.id, n); return n; }
  remove(id: string)   { this.nodes = this.nodes.filter(n => n.id !== id); this.nmap.delete(id); }
  get(id: string)      { return this.nmap.get(id); }

  visibleNodes(): SceneNode[] {
    const vis = new Set(this.layers.filter(l => l.visible).map(l => l.id));
    return this.nodes.filter(n => n.visible && vis.has(n.layerId));
  }

  moveNodeToLayer(nodeId: string, layerId: string) {
    const n = this.nmap.get(nodeId); if (n) n.layerId = layerId;
  }
}
