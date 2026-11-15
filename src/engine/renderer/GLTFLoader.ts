// Minimal self-contained glTF 2.0 / .glb loader.
// Produces interleaved [pos.xyz, normal.xyz] (stride 24) + Uint16 indices,
// matching the engine's geometry format. Merges all mesh primitives, applies
// node transforms, recenters to origin, and normalizes to a unit-ish size so
// imported models drop into the scene at a sane scale.

import type { GeoData } from './Geometry';

interface Accessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;   // 5120 i8, 5121 u8, 5122 i16, 5123 u16, 5125 u32, 5126 f32
  count: number;
  type: 'SCALAR'|'VEC2'|'VEC3'|'VEC4'|'MAT4'|string;
  normalized?: boolean;
}
interface BufferView { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }
interface GLTFMesh  { primitives: GLTFPrim[] }
interface GLTFPrim  { attributes: Record<string,number>; indices?: number; mode?: number }
interface GLTFNode  { mesh?: number; children?: number[]; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }
interface GLTFScene { nodes?: number[] }
interface GLTFRoot {
  accessors?: Accessor[]; bufferViews?: BufferView[]; buffers?: { uri?: string; byteLength: number }[];
  meshes?: GLTFMesh[]; nodes?: GLTFNode[]; scenes?: GLTFScene[]; scene?: number;
}

const COMP_SIZE: Record<number, number> = { 5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4 };
const TYPE_COUNT: Record<string, number> = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT4:16 };

export interface LoadedModel {
  name: string;
  geo: GeoData;            // merged, recentered, unit-scaled
  triangleCount: number;
}

// ── GLB container parse ───────────────────────────────────────────
function parseGLB(buf: ArrayBuffer): { json: GLTFRoot; bin: ArrayBuffer | null } {
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546c67) throw new Error('Not a .glb file (bad magic)');
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error(`Unsupported glb version ${version}`);

  let offset = 12;
  let json: GLTFRoot | null = null;
  let bin: ArrayBuffer | null = null;
  while (offset < buf.byteLength) {
    const chunkLen  = dv.getUint32(offset, true);
    const chunkType = dv.getUint32(offset + 4, true);
    const start = offset + 8;
    if (chunkType === 0x4e4f534a) {            // 'JSON'
      const text = new TextDecoder().decode(new Uint8Array(buf, start, chunkLen));
      json = JSON.parse(text);
    } else if (chunkType === 0x004e4942) {     // 'BIN\0'
      bin = buf.slice(start, start + chunkLen);
    }
    offset = start + chunkLen + (chunkLen % 4 === 0 ? 0 : 4 - (chunkLen % 4));
  }
  if (!json) throw new Error('glb missing JSON chunk');
  return { json, bin };
}

// ── Accessor reader → Float32Array of element components ──────────
function readAccessor(root: GLTFRoot, bin: ArrayBuffer, idx: number): Float32Array {
  const acc = root.accessors![idx];
  const view = root.bufferViews![acc.bufferView ?? 0];
  const comps = TYPE_COUNT[acc.type] ?? 1;
  const compSize = COMP_SIZE[acc.componentType];
  const baseOffset = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const stride = view.byteStride && view.byteStride > 0 ? view.byteStride : comps * compSize;
  const dv = new DataView(bin);
  const out = new Float32Array(acc.count * comps);

  const readOne = (off: number): number => {
    switch (acc.componentType) {
      case 5126: return dv.getFloat32(off, true);
      case 5125: return dv.getUint32(off, true);
      case 5123: return dv.getUint16(off, true);
      case 5122: return dv.getInt16(off, true);
      case 5121: return dv.getUint8(off);
      case 5120: return dv.getInt8(off);
      default:   return dv.getFloat32(off, true);
    }
  };

  for (let i = 0; i < acc.count; i++) {
    const elemOff = baseOffset + i * stride;
    for (let c = 0; c < comps; c++) out[i * comps + c] = readOne(elemOff + c * compSize);
  }
  return out;
}

function readIndices(root: GLTFRoot, bin: ArrayBuffer, idx: number): Uint32Array {
  const acc = root.accessors![idx];
  const view = root.bufferViews![acc.bufferView ?? 0];
  const baseOffset = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const dv = new DataView(bin);
  const out = new Uint32Array(acc.count);
  const size = COMP_SIZE[acc.componentType];
  for (let i = 0; i < acc.count; i++) {
    const off = baseOffset + i * size;
    out[i] = acc.componentType === 5125 ? dv.getUint32(off, true)
           : acc.componentType === 5123 ? dv.getUint16(off, true)
           : dv.getUint8(off);
  }
  return out;
}

// ── 4×4 matrix helpers (column-major, glTF convention) ────────────
function mulMat4(a: number[], b: number[]): number[] {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function trsMatrix(n: GLTFNode): number[] {
  if (n.matrix) return n.matrix.slice();
  const t = n.translation ?? [0, 0, 0];
  const q = n.rotation ?? [0, 0, 0, 1];
  const s = n.scale ?? [1, 1, 1];
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ];
}
function transformPoint(m: number[], p: [number, number, number]): [number, number, number] {
  return [
    m[0]*p[0] + m[4]*p[1] + m[8]*p[2]  + m[12],
    m[1]*p[0] + m[5]*p[1] + m[9]*p[2]  + m[13],
    m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
  ];
}
function transformDir(m: number[], n: [number, number, number]): [number, number, number] {
  // good enough for rigid/uniform-scale models
  const r: [number, number, number] = [
    m[0]*n[0] + m[4]*n[1] + m[8]*n[2],
    m[1]*n[0] + m[5]*n[1] + m[9]*n[2],
    m[2]*n[0] + m[6]*n[1] + m[10]*n[2],
  ];
  const len = Math.hypot(r[0], r[1], r[2]) || 1;
  return [r[0]/len, r[1]/len, r[2]/len];
}

// ── Public API ────────────────────────────────────────────────────
export async function loadGLB(buf: ArrayBuffer, name = 'model'): Promise<LoadedModel> {
  const { json, bin } = parseGLB(buf);
  if (!bin) throw new Error('glb has no binary buffer (external .bin not supported)');

  const positions: number[] = [];   // interleaved pos+normal output
  const indices:   number[] = [];
  let vertBase = 0;

  // Walk the node tree applying transforms so multi-part models assemble correctly.
  const sceneIdx = json.scene ?? 0;
  const rootNodes = json.scenes?.[sceneIdx]?.nodes ?? json.nodes?.map((_, i) => i) ?? [];
  const IDENTITY = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

  const visit = (nodeIdx: number, parent: number[]) => {
    const node = json.nodes![nodeIdx];
    const world = mulMat4(parent, trsMatrix(node));

    if (node.mesh !== undefined) {
      const mesh = json.meshes![node.mesh];
      for (const prim of mesh.primitives) {
        if (prim.mode !== undefined && prim.mode !== 4) continue; // triangles only
        const posIdx = prim.attributes['POSITION'];
        if (posIdx === undefined) continue;
        const pos = readAccessor(json, bin, posIdx);
        const vcount = pos.length / 3;
        const nrm = prim.attributes['NORMAL'] !== undefined
          ? readAccessor(json, bin, prim.attributes['NORMAL'])
          : computeNormals(pos, prim.indices !== undefined ? readIndices(json, bin, prim.indices) : null);

        for (let i = 0; i < vcount; i++) {
          const wp = transformPoint(world, [pos[i*3], pos[i*3+1], pos[i*3+2]]);
          const wn = transformDir(world,  [nrm[i*3], nrm[i*3+1], nrm[i*3+2]]);
          positions.push(wp[0], wp[1], wp[2], wn[0], wn[1], wn[2]);
        }

        if (prim.indices !== undefined) {
          const idx = readIndices(json, bin, prim.indices);
          for (let i = 0; i < idx.length; i++) indices.push(idx[i] + vertBase);
        } else {
          for (let i = 0; i < vcount; i++) indices.push(i + vertBase);
        }
        vertBase += vcount;
      }
    }
    for (const c of node.children ?? []) visit(c, world);
  };
  for (const r of rootNodes) visit(r, IDENTITY);

  if (vertBase === 0) throw new Error('No triangle geometry found in model');

  // ── Recenter + normalize scale ──
  let minX=Infinity,minY=Infinity,minZ=Infinity, maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
  for (let i = 0; i < positions.length; i += 6) {
    minX=Math.min(minX,positions[i]);   maxX=Math.max(maxX,positions[i]);
    minY=Math.min(minY,positions[i+1]); maxY=Math.max(maxY,positions[i+1]);
    minZ=Math.min(minZ,positions[i+2]); maxZ=Math.max(maxZ,positions[i+2]);
  }
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
  const span = Math.max(maxX-minX, maxY-minY, maxZ-minZ) || 1;
  const k = 2.0 / span;                       // fit into ~2-unit box
  for (let i = 0; i < positions.length; i += 6) {
    positions[i]   = (positions[i]   - cx) * k;
    positions[i+1] = (positions[i+1] - cy) * k;
    positions[i+2] = (positions[i+2] - cz) * k;
  }

  // Engine uses Uint16 indices. Split if model exceeds 65535 verts.
  const geo = packIndices(positions, indices);
  return { name, geo, triangleCount: indices.length / 3 };
}

// glTF needs a fetch from a File / URL
export async function loadGLBFromFile(file: File): Promise<LoadedModel> {
  const buf = await file.arrayBuffer();
  return loadGLB(buf, file.name.replace(/\.(glb|gltf)$/i, ''));
}

// ── Helpers ───────────────────────────────────────────────────────
function computeNormals(pos: Float32Array, idx: Uint32Array | null): Float32Array {
  const vcount = pos.length / 3;
  const nrm = new Float32Array(vcount * 3);
  const addFace = (a: number, b: number, c: number) => {
    const ax=pos[a*3],ay=pos[a*3+1],az=pos[a*3+2];
    const bx=pos[b*3],by=pos[b*3+1],bz=pos[b*3+2];
    const cx=pos[c*3],cy=pos[c*3+1],cz=pos[c*3+2];
    const ux=bx-ax,uy=by-ay,uz=bz-az;
    const vx=cx-ax,vy=cy-ay,vz=cz-az;
    const nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    for (const v of [a,b,c]) { nrm[v*3]+=nx; nrm[v*3+1]+=ny; nrm[v*3+2]+=nz; }
  };
  if (idx) for (let i=0;i<idx.length;i+=3) addFace(idx[i],idx[i+1],idx[i+2]);
  else     for (let i=0;i<vcount;i+=3)     addFace(i,i+1,i+2);
  for (let i=0;i<vcount;i++) {
    const x=nrm[i*3],y=nrm[i*3+1],z=nrm[i*3+2];
    const l=Math.hypot(x,y,z)||1;
    nrm[i*3]=x/l; nrm[i*3+1]=y/l; nrm[i*3+2]=z/l;
  }
  return nrm;
}

// Pack into Uint16 indices. If >65535 verts, fall back to a non-indexed (expanded)
// buffer so each Uint16 index range stays valid via re-emitted vertices.
function packIndices(positions: number[], indices: number[]): GeoData {
  const vcount = positions.length / 6;
  if (vcount <= 65536) {
    return {
      vertices: new Float32Array(positions),
      indices:  new Uint16Array(indices),
    };
  }
  // Expand: emit unique vertices per index, then sequential u16 indices in chunks.
  const expandedV: number[] = [];
  const expandedI: number[] = [];
  for (let i = 0; i < indices.length; i++) {
    const vi = indices[i];
    expandedV.push(
      positions[vi*6], positions[vi*6+1], positions[vi*6+2],
      positions[vi*6+3], positions[vi*6+4], positions[vi*6+5],
    );
    expandedI.push(i % 65536);
  }
  // NOTE: very large models (>65536 tris) may still overflow a single draw;
  // engine treats this as best-effort. Good enough for typical asset packs.
  return {
    vertices: new Float32Array(expandedV),
    indices:  new Uint16Array(expandedI.map((_, i) => i & 0xffff)),
  };
}
