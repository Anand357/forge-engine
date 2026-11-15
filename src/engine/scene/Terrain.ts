// Editable heightfield terrain. Owns a height grid; can be sculpted (raise/
// lower/smooth), stamped with feature presets (mountains/hills/canyon), and
// rebuilds its interleaved [pos.xyz, normal.xyz] mesh on demand.

export class Terrain {
  readonly size: number;          // world units (square), centered at origin
  readonly res: number;           // grid resolution (verts per side)
  heights: Float32Array;          // res*res
  private step: number;           // world units between samples

  constructor(size = 200, res = 129) {
    this.size = size;
    this.res = res;
    this.heights = new Float32Array(res * res);
    this.step = size / (res - 1);
  }

  // world XZ → grid coords (float)
  worldToGrid(wx: number, wz: number): { gx: number; gz: number } {
    const half = this.size / 2;
    return {
      gx: (wx + half) / this.step,
      gz: (wz + half) / this.step,
    };
  }

  // Sample height at world XZ (bilinear) — used by player/physics ground follow.
  heightAt(wx: number, wz: number): number {
    const { gx, gz } = this.worldToGrid(wx, wz);
    const x0 = Math.floor(gx), z0 = Math.floor(gz);
    if (x0 < 0 || z0 < 0 || x0 >= this.res - 1 || z0 >= this.res - 1) return 0;
    const fx = gx - x0, fz = gz - z0;
    const h00 = this.heights[z0 * this.res + x0];
    const h10 = this.heights[z0 * this.res + x0 + 1];
    const h01 = this.heights[(z0 + 1) * this.res + x0];
    const h11 = this.heights[(z0 + 1) * this.res + x0 + 1];
    const a = h00 * (1 - fx) + h10 * fx;
    const b = h01 * (1 - fx) + h11 * fx;
    return a * (1 - fz) + b * fz;
  }

  flatten() { this.heights.fill(0); }

  // ── Sculpt brush ───────────────────────────────────────────────
  // mode: +1 raise, -1 lower, 0 smooth. radius/strength in world units.
  sculpt(wx: number, wz: number, radius: number, strength: number, mode: number) {
    const { gx, gz } = this.worldToGrid(wx, wz);
    const gr = radius / this.step;
    const minX = Math.max(0, Math.floor(gx - gr));
    const maxX = Math.min(this.res - 1, Math.ceil(gx + gr));
    const minZ = Math.max(0, Math.floor(gz - gr));
    const maxZ = Math.min(this.res - 1, Math.ceil(gz + gr));

    if (mode === 0) {
      // Smooth: blend each vertex toward the average of its neighbors
      const copy = this.heights.slice();
      for (let z = minZ; z <= maxZ; z++) {
        for (let x = minX; x <= maxX; x++) {
          const d = Math.hypot(x - gx, z - gz);
          if (d > gr) continue;
          const falloff = 1 - d / gr;
          let sum = 0, n = 0;
          for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, nz = z + dz;
            if (nx < 0 || nz < 0 || nx >= this.res || nz >= this.res) continue;
            sum += copy[nz * this.res + nx]; n++;
          }
          const avg = sum / n;
          const i = z * this.res + x;
          this.heights[i] += (avg - this.heights[i]) * falloff * strength;
        }
      }
      return;
    }

    for (let z = minZ; z <= maxZ; z++) {
      for (let x = minX; x <= maxX; x++) {
        const d = Math.hypot(x - gx, z - gz);
        if (d > gr) continue;
        const falloff = Math.cos((d / gr) * Math.PI * 0.5); // smooth bell
        this.heights[z * this.res + x] += mode * strength * falloff;
      }
    }
  }

  // ── Feature presets ────────────────────────────────────────────
  // Adds a mountain range / hills / canyon at world XZ with a given scale.
  stampFeature(kind: 'mountain'|'hills'|'canyon'|'plateau', wx: number, wz: number, scale: number) {
    const { gx, gz } = this.worldToGrid(wx, wz);
    const gr = scale / this.step;
    const minX = Math.max(0, Math.floor(gx - gr));
    const maxX = Math.min(this.res - 1, Math.ceil(gx + gr));
    const minZ = Math.max(0, Math.floor(gz - gr));
    const maxZ = Math.min(this.res - 1, Math.ceil(gz + gr));
    const amp = scale * 0.55;

    for (let z = minZ; z <= maxZ; z++) {
      for (let x = minX; x <= maxX; x++) {
        const nd = Math.hypot(x - gx, z - gz) / gr;
        if (nd > 1) continue;
        const wxw = (x * this.step) - this.size / 2;
        const wzw = (z * this.step) - this.size / 2;
        const i = z * this.res + x;
        if (kind === 'mountain') {
          const peak = Math.pow(1 - nd, 1.8);
          const ridges = this._fbm(wxw * 0.06, wzw * 0.06) * 0.5 + 0.5;
          this.heights[i] += amp * peak * (0.6 + ridges * 0.8);
        } else if (kind === 'hills') {
          const roll = (this._fbm(wxw * 0.045, wzw * 0.045) * 0.5 + 0.5);
          this.heights[i] += amp * 0.35 * roll * Math.pow(1 - nd, 0.7);
        } else if (kind === 'canyon') {
          const carve = Math.pow(1 - nd, 1.4);
          const winding = this._fbm(wxw * 0.05, wzw * 0.05);
          this.heights[i] -= amp * 0.6 * carve * (0.6 + Math.abs(winding) * 0.8);
        } else if (kind === 'plateau') {
          const top = nd < 0.7 ? 1 : Math.pow((1 - nd) / 0.3, 1.5);
          this.heights[i] += amp * 0.5 * Math.max(0, top);
        }
      }
    }
  }

  // ── Mesh build (interleaved pos+normal, stride 24) ─────────────
  buildMesh(): { vertices: Float32Array; indices: Uint32Array } {
    const r = this.res, h = this.heights, st = this.step, half = this.size / 2;
    const verts = new Float32Array(r * r * 6);
    for (let z = 0; z < r; z++) {
      for (let x = 0; x < r; x++) {
        const i = z * r + x;
        const px = x * st - half, pz = z * st - half, py = h[i];
        // Normal from finite differences
        const hl = x > 0     ? h[i - 1] : py;
        const hr = x < r - 1 ? h[i + 1] : py;
        const hd = z > 0     ? h[i - r] : py;
        const hu = z < r - 1 ? h[i + r] : py;
        let nx = hl - hr, ny = 2 * st, nz = hd - hu;
        const len = Math.hypot(nx, ny, nz) || 1;
        nx /= len; ny /= len; nz /= len;
        const o = i * 6;
        verts[o] = px; verts[o + 1] = py; verts[o + 2] = pz;
        verts[o + 3] = nx; verts[o + 4] = ny; verts[o + 5] = nz;
      }
    }
    // Indices (needs Uint32 for >65k verts; 129² = 16641 fits u16, but keep u32 for larger res)
    const idx = new Uint32Array((r - 1) * (r - 1) * 6);
    let k = 0;
    for (let z = 0; z < r - 1; z++) {
      for (let x = 0; x < r - 1; x++) {
        const a = z * r + x, b = a + r;
        idx[k++] = a; idx[k++] = b; idx[k++] = a + 1;
        idx[k++] = b; idx[k++] = b + 1; idx[k++] = a + 1;
      }
    }
    return { vertices: verts, indices: idx };
  }

  // Raycast a ray (origin+dir) against the heightfield. Returns world hit or null.
  raycast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number): { x: number; y: number; z: number } | null {
    // March along the ray sampling terrain height; detect crossing.
    let t = 0;
    const maxT = this.size * 2;
    const stepT = this.step * 0.5;
    let prevDiff = oy - this.heightAt(ox, oz);
    while (t < maxT) {
      t += stepT;
      const x = ox + dx * t, y = oy + dy * t, z = oz + dz * t;
      const diff = y - this.heightAt(x, z);
      if (diff <= 0 && prevDiff > 0) {
        // crossed surface — refine with bisection
        let lo = t - stepT, hi = t;
        for (let i = 0; i < 12; i++) {
          const mid = (lo + hi) / 2;
          const mx = ox + dx * mid, my = oy + dy * mid, mz = oz + dz * mid;
          if (my - this.heightAt(mx, mz) > 0) lo = mid; else hi = mid;
        }
        const ft = (lo + hi) / 2;
        return { x: ox + dx * ft, y: oy + dy * ft, z: oz + dz * ft };
      }
      prevDiff = diff;
    }
    return null;
  }

  // ── Internal noise ─────────────────────────────────────────────
  private _hash(x: number, z: number): number {
    const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
    return s - Math.floor(s);
  }
  private _vnoise(x: number, z: number): number {
    const ix = Math.floor(x), iz = Math.floor(z);
    const fx = x - ix, fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
    const a = this._hash(ix, iz), b = this._hash(ix + 1, iz);
    const c = this._hash(ix, iz + 1), d = this._hash(ix + 1, iz + 1);
    return (a * (1 - ux) + b * ux) * (1 - uz) + (c * (1 - ux) + d * ux) * uz;
  }
  private _fbm(x: number, z: number): number {
    let v = 0, amp = 0.5, fx = x, fz = z;
    for (let i = 0; i < 4; i++) { v += amp * this._vnoise(fx, fz); fx *= 2.03; fz *= 2.03; amp *= 0.5; }
    return v;
  }
}
