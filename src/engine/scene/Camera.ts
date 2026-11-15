import { Vec3 } from '../math/Vec3';
import { Mat4 } from '../math/Mat4';

export class Camera {
  position = new Vec3(0, 18, 35);
  target   = Vec3.zero();
  fov = Math.PI / 4; near = 0.1; far = 2000;

  private yaw   = 0.4;
  private pitch = 0.42;
  private dist  = 35;

  private lmbDown = false;
  private mmbDown = false;
  private lastX = 0; private lastY = 0;

  // WASD fly state
  private keys = new Set<string>();
  private flyMode = false;
  locked = false;     // true during Play mode — editor orbit/pan disabled

  getViewMatrix() { return Mat4.lookAt(this.position, this.target, Vec3.up()); }
  getProj(aspect: number) { return Mat4.perspective(this.fov, aspect, this.near, this.far); }
  getVP(aspect: number) { return this.getProj(aspect).multiply(this.getViewMatrix()); }

  getRay(nx: number, ny: number, aspect: number): { origin: Vec3; dir: Vec3 } {
    const tanH = Math.tan(this.fov / 2);
    const dx = nx * tanH * aspect, dy = ny * tanH;
    const f = this.target.sub(this.position).normalize();
    const r = f.cross(Vec3.up()).normalize();
    const u = r.cross(f);
    const dir = f.add(r.mul(dx)).sub(u.mul(dy)).normalize();
    return { origin: this.position.clone(), dir };
  }

  attachControls(canvas: HTMLCanvasElement) {
    // LMB = orbit
    canvas.addEventListener('mousedown', e => {
      if (this.locked) return;
      if (e.button === 0) { this.lmbDown = true; this.lastX = e.clientX; this.lastY = e.clientY; }
      if (e.button === 1) { this.mmbDown = true; this.lastX = e.clientX; this.lastY = e.clientY; e.preventDefault(); }
    });
    window.addEventListener('mouseup', e => {
      if (e.button === 0) this.lmbDown = false;
      if (e.button === 1) this.mmbDown = false;
    });
    window.addEventListener('mousemove', e => {
      const dx = e.clientX - this.lastX, dy = e.clientY - this.lastY;
      this.lastX = e.clientX; this.lastY = e.clientY;
      if (this.locked) return;
      if (this.lmbDown && !this.flyMode) {
        this.yaw   -= dx * 0.005;
        this.pitch  = Math.max(-1.4, Math.min(1.4, this.pitch - dy * 0.005));
        this._sync();
      }
      if (this.mmbDown) {
        const r = new Vec3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
        const u = new Vec3(0, 1, 0);
        const speed = this.dist * 0.001;
        this.target = this.target.sub(r.mul(dx * speed)).add(u.mul(dy * speed));
        this._sync();
      }
    });

    canvas.addEventListener('wheel', e => {
      if (this.locked) return;
      this.dist = Math.max(2, Math.min(600, this.dist * (1 + e.deltaY * 0.001)));
      this._sync();
    }, { passive: true });

    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // WASD + F for fly mode
    window.addEventListener('keydown', e => { this.keys.add(e.key.toLowerCase()); });
    window.addEventListener('keyup',   e => { this.keys.delete(e.key.toLowerCase()); });

    this._sync();
  }

  update(dt: number) {
    if (!this.flyMode) return;
    const speed = 15 * dt;
    const f = this.target.sub(this.position).normalize();
    const r = f.cross(Vec3.up()).normalize();
    if (this.keys.has('w')) { this.position = this.position.add(f.mul(speed)); this.target = this.target.add(f.mul(speed)); }
    if (this.keys.has('s')) { this.position = this.position.sub(f.mul(speed)); this.target = this.target.sub(f.mul(speed)); }
    if (this.keys.has('a')) { this.position = this.position.sub(r.mul(speed)); this.target = this.target.sub(r.mul(speed)); }
    if (this.keys.has('d')) { this.position = this.position.add(r.mul(speed)); this.target = this.target.add(r.mul(speed)); }
    if (this.keys.has('q')) { this.position.y -= speed; this.target.y -= speed; }
    if (this.keys.has('e')) { this.position.y += speed; this.target.y += speed; }
    this.dist = this.position.sub(this.target).length();
  }

  setFlyMode(v: boolean) {
    this.flyMode = v;
    if (!v) this._sync();
  }

  focusOn(pos: Vec3) {
    this.target = pos.clone();
    this._sync();
  }

  // Direct view override (used by the 3rd-person player camera in Play mode).
  setView(eye: Vec3, target: Vec3) {
    this.position = eye.clone();
    this.target = target.clone();
  }

  _sync() {
    const x = this.dist * Math.cos(this.pitch) * Math.sin(this.yaw);
    const y = this.dist * Math.sin(this.pitch);
    const z = this.dist * Math.cos(this.pitch) * Math.cos(this.yaw);
    this.position = this.target.add(new Vec3(x, y, z));
  }
}
