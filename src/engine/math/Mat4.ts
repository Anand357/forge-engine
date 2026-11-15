import { Vec3 } from './Vec3';

// Column-major storage (matches WGSL mat4x4<f32> layout)
// data[col*4 + row]
export class Mat4 {
  data = new Float32Array(16);

  constructor() { this.identity(); }

  identity(): this {
    const d = this.data; d.fill(0);
    d[0]=1; d[5]=1; d[10]=1; d[15]=1;
    return this;
  }

  clone(): Mat4 { const m = new Mat4(); m.data.set(this.data); return m; }

  // Standard matrix multiply, column-major: result[c][r] = sum_k(a[k][r]*b[c][k])
  multiply(b: Mat4): Mat4 {
    const a = this.data, bd = b.data;
    const o = new Mat4(); const od = o.data;
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k*4+r] * bd[c*4+k];
        od[c*4+r] = s;
      }
    }
    return o;
  }

  // Perspective projection (column-major, WebGPU NDC z in [0,1])
  static perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
    const m = new Mat4(); const d = m.data; d.fill(0);
    const f = 1.0 / Math.tan(fov * 0.5);
    const nf = 1.0 / (near - far);
    d[0]  = f / aspect;
    d[5]  = f;
    d[10] = far * nf;
    d[11] = -1;
    d[14] = far * near * nf;
    d[15] = 0;
    return m;
  }

  // Look-at view matrix (column-major)
  static lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
    const f = center.sub(eye).normalize();
    const s = f.cross(up).normalize();
    const u = s.cross(f);
    const m = new Mat4(); const d = m.data;
    // col 0
    d[0]  =  s.x; d[1]  =  u.x; d[2]  = -f.x; d[3]  = 0;
    // col 1
    d[4]  =  s.y; d[5]  =  u.y; d[6]  = -f.y; d[7]  = 0;
    // col 2
    d[8]  =  s.z; d[9]  =  u.z; d[10] = -f.z; d[11] = 0;
    // col 3 (translation)
    d[12] = -s.dot(eye);
    d[13] = -u.dot(eye);
    d[14] =  f.dot(eye);
    d[15] =  1;
    return m;
  }

  static translation(v: Vec3): Mat4 {
    const m = new Mat4();
    m.data[12] = v.x; m.data[13] = v.y; m.data[14] = v.z;
    return m;
  }

  static scale(v: Vec3): Mat4 {
    const m = new Mat4();
    m.data[0] = v.x; m.data[5] = v.y; m.data[10] = v.z;
    return m;
  }

  static rotationX(a: number): Mat4 {
    const m = new Mat4(), c = Math.cos(a), s = Math.sin(a);
    m.data[5]=c; m.data[6]=s; m.data[9]=-s; m.data[10]=c;
    return m;
  }

  static rotationY(a: number): Mat4 {
    const m = new Mat4(), c = Math.cos(a), s = Math.sin(a);
    m.data[0]=c; m.data[2]=-s; m.data[8]=s; m.data[10]=c;
    return m;
  }

  static rotationZ(a: number): Mat4 {
    const m = new Mat4(), c = Math.cos(a), s = Math.sin(a);
    m.data[0]=c; m.data[1]=s; m.data[4]=-s; m.data[5]=c;
    return m;
  }

  // Full 4×4 inverse (column-major). Returns identity if singular.
  invert(): Mat4 {
    const m = this.data;
    const inv = new Float32Array(16);

    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    const out = new Mat4();
    if (Math.abs(det) < 1e-12) return out; // singular → identity
    det = 1.0 / det;
    for (let i = 0; i < 16; i++) out.data[i] = inv[i] * det;
    return out;
  }

  // Orthographic projection for shadow maps (WebGPU z in [0,1])
  static orthographic(l: number, r: number, b: number, t: number, near: number, far: number): Mat4 {
    const m = new Mat4(); m.data.fill(0);
    m.data[0]  =  2 / (r - l);
    m.data[5]  =  2 / (t - b);
    m.data[10] = -1 / (far - near);
    m.data[12] = -(r + l) / (r - l);
    m.data[13] = -(t + b) / (t - b);
    m.data[14] = -near / (far - near);
    m.data[15] =  1;
    return m;
  }
}
