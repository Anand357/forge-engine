// Procedural asset library: 100+ ready-to-place models built from primitives.
// No external files. Each asset returns interleaved [pos.xyz, normal.xyz]
// (stride 24) geometry + Uint16 indices + a suggested color/material, matching
// the engine's mesh format. Models are recentered on XZ and rest on y=0.

import type { GeoData } from './Geometry';

export interface AssetDef {
  id: string;
  name: string;
  category: 'Characters' | 'Vehicles' | 'Nature' | 'Buildings' | 'Props';
  color: [number, number, number];
  roughness: number;
  metalness: number;
  build: () => GeoData;
}

// ── Mesh builder: accumulate transformed primitive parts ───────────
class MeshBuilder {
  v: number[] = [];
  i: number[] = [];

  private push(verts: number[], idx: number[], tx: number, ty: number, tz: number,
               sx: number, sy: number, sz: number, ry = 0) {
    const base = this.v.length / 6;
    const cy = Math.cos(ry), sy2 = Math.sin(ry);
    for (let k = 0; k < verts.length; k += 6) {
      let px = verts[k] * sx, py = verts[k + 1] * sy, pz = verts[k + 2] * sz;
      let nx = verts[k + 3], ny = verts[k + 4], nz = verts[k + 5];
      // rotate around Y
      const rpx = px * cy + pz * sy2, rpz = -px * sy2 + pz * cy;
      const rnx = nx * cy + nz * sy2, rnz = -nx * sy2 + nz * cy;
      this.v.push(rpx + tx, py + ty, rpz + tz, rnx, ny, rnz);
    }
    for (const id of idx) this.i.push(id + base);
  }

  box(tx: number, ty: number, tz: number, sx: number, sy: number, sz: number, ry = 0) {
    this.push(UNIT_BOX.v, UNIT_BOX.i, tx, ty, tz, sx, sy, sz, ry);
  }
  sphere(tx: number, ty: number, tz: number, r: number, ry = 0) {
    this.push(UNIT_SPHERE.v, UNIT_SPHERE.i, tx, ty, tz, r, r, r, ry);
  }
  ellipsoid(tx: number, ty: number, tz: number, sx: number, sy: number, sz: number, ry = 0) {
    this.push(UNIT_SPHERE.v, UNIT_SPHERE.i, tx, ty, tz, sx, sy, sz, ry);
  }
  cyl(tx: number, ty: number, tz: number, r: number, h: number, ry = 0) {
    this.push(UNIT_CYL.v, UNIT_CYL.i, tx, ty, tz, r, h, r, ry);
  }
  cone(tx: number, ty: number, tz: number, r: number, h: number, ry = 0) {
    this.push(UNIT_CONE.v, UNIT_CONE.i, tx, ty, tz, r, h, r, ry);
  }

  finish(): GeoData {
    return { vertices: new Float32Array(this.v), indices: new Uint16Array(this.i) };
  }
}

// ── Unit primitives (built once) ──────────────────────────────────
function unitBox() {
  const h = 0.5;
  const f = [
    [-h,-h, h,0,0,1],[ h,-h, h,0,0,1],[ h, h, h,0,0,1],[-h, h, h,0,0,1],
    [ h,-h,-h,0,0,-1],[-h,-h,-h,0,0,-1],[-h, h,-h,0,0,-1],[ h, h,-h,0,0,-1],
    [-h, h, h,0,1,0],[ h, h, h,0,1,0],[ h, h,-h,0,1,0],[-h, h,-h,0,1,0],
    [-h,-h,-h,0,-1,0],[ h,-h,-h,0,-1,0],[ h,-h, h,0,-1,0],[-h,-h, h,0,-1,0],
    [ h,-h, h,1,0,0],[ h,-h,-h,1,0,0],[ h, h,-h,1,0,0],[ h, h, h,1,0,0],
    [-h,-h,-h,-1,0,0],[-h,-h, h,-1,0,0],[-h, h, h,-1,0,0],[-h, h,-h,-1,0,0],
  ];
  const i: number[] = [];
  for (let q = 0; q < 6; q++) { const b = q * 4; i.push(b, b+1, b+2, b, b+2, b+3); }
  return { v: f.flat(), i };
}
function unitSphere(seg = 14) {
  const v: number[] = [], i: number[] = [];
  for (let lat = 0; lat <= seg; lat++) {
    const t = (lat / seg) * Math.PI;
    for (let lon = 0; lon <= seg; lon++) {
      const p = (lon / seg) * 2 * Math.PI;
      const x = Math.sin(t) * Math.cos(p), y = Math.cos(t), z = Math.sin(t) * Math.sin(p);
      v.push(x, y, z, x, y, z);
    }
  }
  for (let lat = 0; lat < seg; lat++) for (let lon = 0; lon < seg; lon++) {
    const a = lat * (seg + 1) + lon, b = a + seg + 1;
    i.push(a, b, a + 1, b, b + 1, a + 1);
  }
  return { v, i };
}
function unitCyl(seg = 16) {
  const v: number[] = [], i: number[] = [];
  for (let s = 0; s <= seg; s++) {
    const a = (s / seg) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
    v.push(x, -0.5, z, x, 0, z); v.push(x, 0.5, z, x, 0, z);
  }
  for (let s = 0; s < seg; s++) { const b = s * 2; i.push(b, b+2, b+1, b+1, b+2, b+3); }
  const topC = v.length / 6; v.push(0, 0.5, 0, 0, 1, 0);
  const botC = v.length / 6; v.push(0, -0.5, 0, 0, -1, 0);
  for (let s = 0; s < seg; s++) {
    const a0 = (s / seg) * Math.PI * 2, a1 = ((s + 1) / seg) * Math.PI * 2;
    const t0 = v.length / 6; v.push(Math.cos(a0), 0.5, Math.sin(a0), 0, 1, 0);
    const t1 = v.length / 6; v.push(Math.cos(a1), 0.5, Math.sin(a1), 0, 1, 0);
    i.push(topC, t0, t1);
    const b0 = v.length / 6; v.push(Math.cos(a0), -0.5, Math.sin(a0), 0, -1, 0);
    const b1 = v.length / 6; v.push(Math.cos(a1), -0.5, Math.sin(a1), 0, -1, 0);
    i.push(botC, b1, b0);
  }
  return { v, i };
}
function unitCone(seg = 16) {
  const v: number[] = [], i: number[] = [];
  const sl = Math.hypot(1, 1);
  for (let s = 0; s <= seg; s++) {
    const a = (s / seg) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
    v.push(x, -0.5, z, x / sl, 1 / sl, z / sl);
    v.push(0, 0.5, 0, x / sl, 1 / sl, z / sl);
  }
  for (let s = 0; s < seg; s++) { const b = s * 2; i.push(b, b+2, b+1); }
  const c = v.length / 6; v.push(0, -0.5, 0, 0, -1, 0);
  for (let s = 0; s < seg; s++) {
    const a0 = (s / seg) * Math.PI * 2, a1 = ((s + 1) / seg) * Math.PI * 2;
    const b0 = v.length / 6; v.push(Math.cos(a0), -0.5, Math.sin(a0), 0, -1, 0);
    const b1 = v.length / 6; v.push(Math.cos(a1), -0.5, Math.sin(a1), 0, -1, 0);
    i.push(c, b1, b0);
  }
  return { v, i };
}

const UNIT_BOX = unitBox();
const UNIT_SPHERE = unitSphere();
const UNIT_CYL = unitCyl();
const UNIT_CONE = unitCone();

// ── Generators ─────────────────────────────────────────────────────
// Characters (humanoid built from primitives) ----------------------
function humanoid(opts: { headR?: number; bulk?: number; legLen?: number } = {}): GeoData {
  const m = new MeshBuilder();
  const bulk = opts.bulk ?? 1, headR = opts.headR ?? 0.32, legLen = opts.legLen ?? 0.9;
  m.box(0, legLen + 0.55, 0, 0.7 * bulk, 0.9, 0.4 * bulk);      // torso
  m.sphere(0, legLen + 1.25, 0, headR);                         // head
  m.box(-0.45 * bulk, legLen + 0.6, 0, 0.18, 0.8, 0.18);       // left arm
  m.box( 0.45 * bulk, legLen + 0.6, 0, 0.18, 0.8, 0.18);       // right arm
  m.box(-0.2, legLen * 0.5, 0, 0.22, legLen, 0.24);            // left leg
  m.box( 0.2, legLen * 0.5, 0, 0.22, legLen, 0.24);            // right leg
  return m.finish();
}
function robot(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 1.3, 0, 0.8, 1.0, 0.6);
  m.box(0, 2.05, 0, 0.55, 0.5, 0.55);    // boxy head
  m.cyl(-0.55, 1.3, 0, 0.12, 1.0);
  m.cyl( 0.55, 1.3, 0, 0.12, 1.0);
  m.box(-0.22, 0.45, 0, 0.25, 0.9, 0.3);
  m.box( 0.22, 0.45, 0, 0.25, 0.9, 0.3);
  m.sphere(0, 2.4, 0, 0.08);             // antenna tip
  m.cyl(0, 2.28, 0, 0.03, 0.25);
  return m.finish();
}
function critter(): GeoData {
  const m = new MeshBuilder();
  m.ellipsoid(0, 0.5, 0, 0.6, 0.5, 0.85);
  m.sphere(0, 0.7, 0.7, 0.38);
  m.cone(-0.18, 1.0, 0.7, 0.12, 0.25);   // ears
  m.cone( 0.18, 1.0, 0.7, 0.12, 0.25);
  for (const [x, z] of [[-0.3,-0.4],[0.3,-0.4],[-0.3,0.4],[0.3,0.4]] as const)
    m.cyl(x, 0.18, z, 0.1, 0.36);
  return m.finish();
}

// Vehicles ----------------------------------------------------------
function car(opts: { len?: number; cabin?: number } = {}): GeoData {
  const m = new MeshBuilder();
  const len = opts.len ?? 2.6;
  m.box(0, 0.55, 0, 1.2, 0.5, len);                 // lower body
  m.box(0, 0.95, -0.1, 0.95, 0.45, (opts.cabin ?? len * 0.5)); // cabin
  for (const [x, z] of [[-0.6,len*0.32],[0.6,len*0.32],[-0.6,-len*0.32],[0.6,-len*0.32]] as const) {
    m.cyl(x, 0.3, z, 0.32, 0.25, Math.PI / 2);      // wheels (cylinders on side)
  }
  return m.finish();
}
function truck(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.7, -0.6, 1.4, 0.6, 1.4);    // cab
  m.box(0, 1.25, -0.6, 1.3, 0.5, 1.2);   // cab top
  m.box(0, 0.85, 1.3, 1.5, 0.9, 2.6);    // cargo
  for (const [x, z] of [[-0.7,1.2],[0.7,1.2],[-0.7,-1.0],[0.7,-1.0],[-0.7,2.2],[0.7,2.2]] as const)
    m.cyl(x, 0.4, z, 0.42, 0.3, Math.PI / 2);
  return m.finish();
}
function bike(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 0.5, 0.7, 0.42, 0.12, Math.PI / 2);   // front wheel
  m.cyl(0, 0.5, -0.7, 0.42, 0.12, Math.PI / 2);  // rear wheel
  m.box(0, 0.7, 0, 0.12, 0.18, 1.3);             // frame
  m.box(0, 0.95, -0.4, 0.5, 0.2, 0.5);           // seat
  m.box(0, 1.0, 0.7, 0.5, 0.1, 0.12);            // handlebar
  return m.finish();
}
function plane(): GeoData {
  const m = new MeshBuilder();
  m.ellipsoid(0, 1, 0, 0.5, 0.5, 2.6);   // fuselage
  m.box(0, 1, 0, 4.2, 0.12, 0.7);        // wings
  m.box(0, 1.4, -2.2, 0.12, 0.8, 0.5);   // tail fin
  m.box(0, 1, -2.2, 1.4, 0.1, 0.4);      // tail wing
  m.cone(0, 1, 2.7, 0.4, 0.6);           // nose
  return m.finish();
}
function boat(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.5, 0, 1.4, 0.6, 3.2);
  m.cone(0, 0.5, 1.8, 0.7, 1.0);         // bow
  m.box(0, 1.1, -0.5, 1.0, 0.6, 1.0);    // cabin
  return m.finish();
}

// Nature ------------------------------------------------------------
function treePine(scale = 1): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 0.7 * scale, 0, 0.18 * scale, 1.4 * scale);
  m.cone(0, 1.7 * scale, 0, 1.0 * scale, 1.4 * scale);
  m.cone(0, 2.5 * scale, 0, 0.8 * scale, 1.2 * scale);
  m.cone(0, 3.2 * scale, 0, 0.55 * scale, 1.0 * scale);
  return m.finish();
}
function treeOak(scale = 1): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 1.0 * scale, 0, 0.25 * scale, 2.0 * scale);
  m.sphere(0, 2.8 * scale, 0, 1.3 * scale);
  m.sphere(-0.8 * scale, 2.4 * scale, 0.4 * scale, 0.8 * scale);
  m.sphere(0.7 * scale, 2.5 * scale, -0.5 * scale, 0.85 * scale);
  return m.finish();
}
function bush(): GeoData {
  const m = new MeshBuilder();
  m.sphere(0, 0.5, 0, 0.55);
  m.sphere(-0.4, 0.4, 0.2, 0.4);
  m.sphere(0.4, 0.45, -0.2, 0.42);
  return m.finish();
}
function rock(seed = 1): GeoData {
  const m = new MeshBuilder();
  const r = 0.6 + (seed % 3) * 0.2;
  m.ellipsoid(0, r * 0.6, 0, r, r * 0.7, r * 0.85, seed);
  m.ellipsoid(r * 0.4, r * 0.4, 0, r * 0.5, r * 0.4, r * 0.5, seed * 2);
  return m.finish();
}
function cactus(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 1.0, 0, 0.3, 2.0);
  m.cyl(-0.5, 1.2, 0, 0.14, 0.8);
  m.cyl(-0.5, 1.7, 0, 0.14, 0.6); // bend approximation
  m.cyl(0.5, 1.0, 0, 0.14, 0.7);
  m.cyl(0.5, 1.5, 0, 0.14, 0.5);
  return m.finish();
}
function mushroom(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 0.4, 0, 0.12, 0.8);
  m.ellipsoid(0, 0.85, 0, 0.45, 0.3, 0.45);
  return m.finish();
}
function flower(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 0.4, 0, 0.04, 0.8);
  m.sphere(0, 0.85, 0, 0.18);
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2;
    m.ellipsoid(Math.cos(a) * 0.22, 0.85, Math.sin(a) * 0.22, 0.16, 0.05, 0.1);
  }
  return m.finish();
}
function logPile(): GeoData {
  const m = new MeshBuilder();
  m.cyl(-0.25, 0.3, 0, 0.28, 1.6, Math.PI / 2);
  m.cyl(0.25, 0.3, 0, 0.28, 1.6, Math.PI / 2);
  m.cyl(0, 0.75, 0, 0.28, 1.6, Math.PI / 2);
  return m.finish();
}

// Buildings ---------------------------------------------------------
function house(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 1.0, 0, 2.4, 2.0, 2.0);
  m.cone(0, 2.6, 0, 1.9, 1.2);       // roof (4-sided-ish cone)
  m.box(0, 0.6, 1.02, 0.5, 1.0, 0.1); // door
  return m.finish();
}
function tower(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 2.0, 0, 0.9, 4.0);
  m.cone(0, 4.6, 0, 1.1, 1.2);
  m.box(0, 1.2, 0.92, 0.4, 0.6, 0.1);
  return m.finish();
}
function skyscraper(floors = 8): GeoData {
  const m = new MeshBuilder();
  m.box(0, floors * 0.5, 0, 2.0, floors, 2.0);
  m.box(0, floors + 0.3, 0, 0.6, 0.6, 0.6); // rooftop unit
  return m.finish();
}
function shed(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.8, 0, 2.0, 1.6, 1.6);
  m.box(0, 1.7, 0, 2.2, 0.2, 1.8); // flat slanted roof approx
  return m.finish();
}
function tent(): GeoData {
  const m = new MeshBuilder();
  m.cone(0, 0.9, 0, 1.2, 1.8);
  return m.finish();
}
function bridge(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.6, 0, 1.6, 0.2, 4.0);
  m.box(-0.7, 0.9, 0, 0.1, 0.6, 4.0);
  m.box(0.7, 0.9, 0, 0.1, 0.6, 4.0);
  for (let z = -1.8; z <= 1.8; z += 0.9) {
    m.cyl(-0.7, 0.3, z, 0.12, 0.6);
    m.cyl(0.7, 0.3, z, 0.12, 0.6);
  }
  return m.finish();
}
function windmill(): GeoData {
  const m = new MeshBuilder();
  m.cone(0, 1.6, 0, 1.0, 3.2);
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI * 2;
    m.box(Math.cos(a) * 1.0, 3.0, 0.5, 0.15, 1.6, 0.05, a);
  }
  return m.finish();
}

// Props -------------------------------------------------------------
function barrel(): GeoData { const m = new MeshBuilder(); m.cyl(0, 0.55, 0, 0.45, 1.1); return m.finish(); }
function crate(): GeoData { const m = new MeshBuilder(); m.box(0, 0.5, 0, 1.0, 1.0, 1.0); return m.finish(); }
function lamp(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 1.5, 0, 0.08, 3.0);
  m.box(0, 3.0, 0.35, 0.1, 0.1, 0.7);
  m.sphere(0, 2.95, 0.7, 0.22);
  return m.finish();
}
function bench(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.45, 0, 1.6, 0.1, 0.5);
  m.box(0, 0.75, -0.2, 1.6, 0.5, 0.08);
  m.box(-0.7, 0.2, 0, 0.1, 0.4, 0.5);
  m.box(0.7, 0.2, 0, 0.1, 0.4, 0.5);
  return m.finish();
}
function fence(): GeoData {
  const m = new MeshBuilder();
  for (let x = -1.5; x <= 1.5; x += 1.0) m.box(x, 0.6, 0, 0.1, 1.2, 0.1);
  m.box(0, 0.9, 0, 3.2, 0.1, 0.08);
  m.box(0, 0.4, 0, 3.2, 0.1, 0.08);
  return m.finish();
}
function streetSign(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 1.2, 0, 0.06, 2.4);
  m.box(0, 2.2, 0, 0.8, 0.4, 0.05);
  return m.finish();
}
function table(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 1.0, 0, 1.6, 0.1, 1.0);
  for (const [x, z] of [[-0.7,-0.4],[0.7,-0.4],[-0.7,0.4],[0.7,0.4]] as const) m.box(x, 0.5, z, 0.1, 1.0, 0.1);
  return m.finish();
}
function chair(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.5, 0, 0.6, 0.1, 0.6);
  m.box(0, 0.9, -0.25, 0.6, 0.8, 0.08);
  for (const [x, z] of [[-0.25,-0.25],[0.25,-0.25],[-0.25,0.25],[0.25,0.25]] as const) m.box(x, 0.25, z, 0.08, 0.5, 0.08);
  return m.finish();
}
function streetlamp(): GeoData { return lamp(); }
function fountain(): GeoData {
  const m = new MeshBuilder();
  m.cyl(0, 0.3, 0, 1.4, 0.6);
  m.cyl(0, 0.9, 0, 0.25, 1.2);
  m.cyl(0, 1.6, 0, 0.7, 0.2);
  return m.finish();
}
function statue(): GeoData {
  const m = new MeshBuilder();
  m.box(0, 0.4, 0, 1.0, 0.8, 1.0);       // pedestal
  const h = humanoid({ bulk: 1.1 });
  const mb = new MeshBuilder();
  mb.v = m.v.slice(); mb.i = m.i.slice();
  // append humanoid lifted onto pedestal
  const base = mb.v.length / 6;
  for (let k = 0; k < h.vertices.length; k += 6) {
    mb.v.push(h.vertices[k], h.vertices[k+1] + 0.8, h.vertices[k+2],
              h.vertices[k+3], h.vertices[k+4], h.vertices[k+5]);
  }
  for (let k = 0; k < h.indices.length; k++) mb.i.push(h.indices[k] + base);
  return mb.finish();
}

// ── Library assembly ───────────────────────────────────────────────
let _seq = 0;
function def(name: string, category: AssetDef['category'], color: [number,number,number],
            build: () => GeoData, rough = 0.6, metal = 0): AssetDef {
  return { id: 'asset_' + (_seq++), name, category, color, roughness: rough, metalness: metal, build };
}

const C = {
  skin: [0.85, 0.68, 0.55] as [number,number,number],
  red: [0.78, 0.2, 0.2] as [number,number,number],
  blue: [0.22, 0.42, 0.85] as [number,number,number],
  green: [0.25, 0.6, 0.28] as [number,number,number],
  dark: [0.18, 0.2, 0.24] as [number,number,number],
  gray: [0.5, 0.52, 0.56] as [number,number,number],
  brown: [0.42, 0.3, 0.18] as [number,number,number],
  wood: [0.55, 0.4, 0.24] as [number,number,number],
  stone: [0.55, 0.55, 0.58] as [number,number,number],
  yellow: [0.85, 0.72, 0.2] as [number,number,number],
  white: [0.9, 0.9, 0.92] as [number,number,number],
  orange: [0.85, 0.45, 0.15] as [number,number,number],
  purple: [0.55, 0.3, 0.7] as [number,number,number],
  cyan: [0.2, 0.7, 0.75] as [number,number,number],
  pink: [0.9, 0.5, 0.65] as [number,number,number],
};

// Builds the full 100+ asset list with color variations.
export function buildAssetLibrary(): AssetDef[] {
  const list: AssetDef[] = [];

  // Characters (12) — color variants of humanoid/robot/critter + a default player
  list.push(def('Player (Hero)', 'Characters', C.blue, () => humanoid({ bulk: 1.05 }), 0.5));
  list.push(def('Villager', 'Characters', C.brown, () => humanoid(), 0.6));
  list.push(def('Guard', 'Characters', C.dark, () => humanoid({ bulk: 1.15 }), 0.45, 0.2));
  list.push(def('Child', 'Characters', C.yellow, () => humanoid({ bulk: 0.75, legLen: 0.6, headR: 0.3 }), 0.6));
  list.push(def('Mage', 'Characters', C.purple, () => humanoid(), 0.55));
  list.push(def('Knight', 'Characters', C.gray, () => humanoid({ bulk: 1.2 }), 0.35, 0.5));
  list.push(def('Robot', 'Characters', C.gray, robot, 0.3, 0.7));
  list.push(def('Robot (Red)', 'Characters', C.red, robot, 0.3, 0.6));
  list.push(def('Robot (Cyan)', 'Characters', C.cyan, robot, 0.3, 0.6));
  list.push(def('Critter', 'Characters', C.brown, critter, 0.7));
  list.push(def('Critter (Gray)', 'Characters', C.gray, critter, 0.7));
  list.push(def('Critter (White)', 'Characters', C.white, critter, 0.7));

  // Vehicles (20)
  const carColors: [string, [number,number,number]][] = [
    ['Red', C.red], ['Blue', C.blue], ['Black', C.dark], ['White', C.white],
    ['Yellow', C.yellow], ['Green', C.green], ['Orange', C.orange], ['Cyan', C.cyan],
  ];
  for (const [cn, col] of carColors) list.push(def(`Car (${cn})`, 'Vehicles', col, () => car(), 0.25, 0.6));
  list.push(def('Sports Car', 'Vehicles', C.red, () => car({ len: 2.9, cabin: 1.0 }), 0.2, 0.7));
  list.push(def('SUV', 'Vehicles', C.dark, () => car({ len: 3.0, cabin: 1.8 }), 0.3, 0.5));
  list.push(def('Truck', 'Vehicles', C.blue, truck, 0.35, 0.4));
  list.push(def('Truck (Red)', 'Vehicles', C.red, truck, 0.35, 0.4));
  list.push(def('Bike', 'Vehicles', C.dark, bike, 0.3, 0.6));
  list.push(def('Bike (Red)', 'Vehicles', C.red, bike, 0.3, 0.6));
  list.push(def('Bike (Blue)', 'Vehicles', C.blue, bike, 0.3, 0.6));
  list.push(def('Plane', 'Vehicles', C.white, plane, 0.3, 0.5));
  list.push(def('Plane (Blue)', 'Vehicles', C.blue, plane, 0.3, 0.5));
  list.push(def('Boat', 'Vehicles', C.wood, boat, 0.5));
  list.push(def('Boat (White)', 'Vehicles', C.white, boat, 0.4));

  // Nature (32)
  for (let k = 0; k < 4; k++) list.push(def(`Pine Tree ${k+1}`, 'Nature', C.green, () => treePine(0.8 + k * 0.25), 0.85));
  for (let k = 0; k < 4; k++) list.push(def(`Oak Tree ${k+1}`, 'Nature', C.green, () => treeOak(0.8 + k * 0.25), 0.85));
  list.push(def('Autumn Tree', 'Nature', C.orange, () => treeOak(1.1), 0.85));
  list.push(def('Autumn Pine', 'Nature', C.brown, () => treePine(1.0), 0.85));
  for (let k = 0; k < 3; k++) list.push(def(`Bush ${k+1}`, 'Nature', C.green, bush, 0.9));
  for (let k = 0; k < 6; k++) list.push(def(`Rock ${k+1}`, 'Nature', C.stone, () => rock(k + 1), 0.95));
  list.push(def('Boulder', 'Nature', C.gray, () => rock(7), 0.95));
  list.push(def('Cactus', 'Nature', C.green, cactus, 0.8));
  list.push(def('Cactus (Tall)', 'Nature', C.green, cactus, 0.8));
  for (let k = 0; k < 3; k++) list.push(def(`Mushroom ${k+1}`, 'Nature', [0.8, 0.3, 0.25], mushroom, 0.7));
  list.push(def('Flower (Red)', 'Nature', C.red, flower, 0.7));
  list.push(def('Flower (Yellow)', 'Nature', C.yellow, flower, 0.7));
  list.push(def('Flower (Pink)', 'Nature', C.pink, flower, 0.7));
  list.push(def('Flower (Purple)', 'Nature', C.purple, flower, 0.7));
  list.push(def('Log Pile', 'Nature', C.wood, logPile, 0.8));
  list.push(def('Palm Tree', 'Nature', C.green, () => treePine(1.3), 0.85));

  // Buildings (20)
  const houseColors: [string, [number,number,number]][] = [
    ['Red', C.red], ['Blue', C.blue], ['Beige', [0.8,0.72,0.55]], ['Green', C.green],
  ];
  for (const [cn, col] of houseColors) list.push(def(`House (${cn})`, 'Buildings', col, house, 0.7));
  list.push(def('Tower', 'Buildings', C.stone, tower, 0.8));
  list.push(def('Castle Tower', 'Buildings', C.gray, tower, 0.85));
  for (let k = 0; k < 4; k++) list.push(def(`Skyscraper ${k+1}`, 'Buildings', [0.4,0.5,0.6], () => skyscraper(6 + k * 2), 0.2, 0.4));
  list.push(def('Shed', 'Buildings', C.wood, shed, 0.75));
  list.push(def('Barn', 'Buildings', C.red, shed, 0.75));
  list.push(def('Tent', 'Buildings', C.green, tent, 0.8));
  list.push(def('Tent (Orange)', 'Buildings', C.orange, tent, 0.8));
  list.push(def('Bridge', 'Buildings', C.wood, bridge, 0.7));
  list.push(def('Stone Bridge', 'Buildings', C.stone, bridge, 0.85));
  list.push(def('Windmill', 'Buildings', [0.8,0.75,0.6], windmill, 0.75));
  list.push(def('Lighthouse', 'Buildings', C.white, tower, 0.7));
  list.push(def('Hut', 'Buildings', C.brown, house, 0.8));
  list.push(def('Office Block', 'Buildings', [0.45,0.55,0.62], () => skyscraper(5), 0.2, 0.4));

  // Props (25)
  list.push(def('Barrel', 'Props', C.wood, barrel, 0.6));
  list.push(def('Barrel (Metal)', 'Props', C.gray, barrel, 0.3, 0.6));
  for (let k = 0; k < 3; k++) list.push(def(`Crate ${k+1}`, 'Props', C.wood, crate, 0.7));
  list.push(def('Crate (Dark)', 'Props', C.dark, crate, 0.6));
  list.push(def('Street Lamp', 'Props', C.dark, lamp, 0.4, 0.3));
  list.push(def('Lamp Post', 'Props', C.gray, lamp, 0.4, 0.3));
  list.push(def('Bench', 'Props', C.wood, bench, 0.7));
  list.push(def('Bench (Park)', 'Props', C.green, bench, 0.6));
  list.push(def('Fence', 'Props', C.wood, fence, 0.8));
  list.push(def('Fence (White)', 'Props', C.white, fence, 0.7));
  list.push(def('Street Sign', 'Props', C.blue, streetSign, 0.5, 0.3));
  list.push(def('Stop Sign', 'Props', C.red, streetSign, 0.5, 0.3));
  list.push(def('Table', 'Props', C.wood, table, 0.7));
  list.push(def('Chair', 'Props', C.wood, chair, 0.7));
  list.push(def('Chair (Red)', 'Props', C.red, chair, 0.7));
  list.push(def('Fountain', 'Props', C.stone, fountain, 0.7));
  list.push(def('Statue', 'Props', C.stone, statue, 0.8));
  list.push(def('Statue (Gold)', 'Props', C.yellow, statue, 0.3, 0.8));
  list.push(def('Crate Stack', 'Props', C.wood, () => { const m = new MeshBuilder(); m.box(0,0.5,0,1,1,1); m.box(0,1.5,0,0.9,1,0.9); return m.finish(); }, 0.7));
  list.push(def('Boulder Prop', 'Props', C.gray, () => rock(3), 0.95));
  list.push(def('Signpost', 'Props', C.wood, streetSign, 0.7));
  list.push(def('Lantern', 'Props', C.yellow, lamp, 0.4));

  return list;
}
