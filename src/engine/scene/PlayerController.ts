// 3rd-person player controller for Play mode.
// WASD movement relative to camera yaw, Shift to sprint, Space to jump with
// gravity, and ground-follow over the editable terrain. A spring-arm camera
// trails behind the player; the mouse orbits the view.

import { Vec3 } from '../math/Vec3';

export interface PlayerInput {
  forward: boolean; back: boolean; left: boolean; right: boolean;
  sprint: boolean; jumpPressed: boolean;
}

export class PlayerController {
  position = new Vec3(0, 0, 0);
  velocityY = 0;
  yaw = 0;            // facing direction (radians)
  grounded = true;

  // camera orbit (set by mouse)
  camYaw = 0;
  camPitch = 0.35;
  camDist = 7;

  // tuning
  walkSpeed = 6;
  sprintSpeed = 11;
  jumpVelocity = 8;
  gravity = -22;
  turnLerp = 0.18;

  // ground height provider (terrain)
  groundAt: (x: number, z: number) => number;
  // optional collision resolver (from→to, radius, height) → corrected position
  collide?: (from: Vec3, to: Vec3, r: number, h: number) => Vec3;
  // optional "stand on object tops" support height
  supportAt?: (x: number, z: number, r: number) => number;
  radius = 0.4;
  height = 1.8;

  constructor(groundAt: (x: number, z: number) => number, start = new Vec3(0, 0, 0)) {
    this.groundAt = groundAt;
    this.position = start.clone();
    this.position.y = this.groundAt(start.x, start.z);
  }

  // Mouse orbit input (called from a pointer-lock / drag handler)
  addMouseDelta(dx: number, dy: number) {
    this.camYaw   -= dx * 0.0035;
    this.camPitch = Math.max(-0.2, Math.min(1.2, this.camPitch + dy * 0.003));
  }
  zoom(deltaY: number) {
    this.camDist = Math.max(3, Math.min(16, this.camDist * (1 + deltaY * 0.001)));
  }

  update(dt: number, input: PlayerInput) {
    // Movement direction in world space, relative to camera yaw
    let mx = 0, mz = 0;
    if (input.forward) mz += 1;
    if (input.back)    mz -= 1;
    if (input.left)    mx -= 1;
    if (input.right)   mx += 1;

    const moving = mx !== 0 || mz !== 0;
    const speed = input.sprint ? this.sprintSpeed : this.walkSpeed;

    if (moving) {
      // camera-forward on the ground plane
      const cf = new Vec3(Math.sin(this.camYaw), 0, Math.cos(this.camYaw));
      const cr = new Vec3(Math.cos(this.camYaw), 0, -Math.sin(this.camYaw));
      let dir = cf.mul(mz).add(cr.mul(mx));
      const len = dir.length() || 1;
      dir = dir.mul(1 / len);
      const desired = this.position.add(dir.mul(speed * dt));
      // resolve against solid objects (wall-sliding)
      this.position = this.collide
        ? this.collide(this.position, desired, this.radius, this.height)
        : desired;
      // smoothly face movement direction
      const targetYaw = Math.atan2(dir.x, dir.z);
      this.yaw = lerpAngle(this.yaw, targetYaw, this.turnLerp);
    }

    // Ground = max(terrain, top of any object we're standing on)
    let ground = this.groundAt(this.position.x, this.position.z);
    if (this.supportAt) {
      const sup = this.supportAt(this.position.x, this.position.z, this.radius);
      // only step up onto objects close to our feet (avoid teleporting onto tall roofs)
      if (sup > ground && sup <= this.position.y + 0.6) ground = sup;
    }
    if (this.grounded && input.jumpPressed) {
      this.velocityY = this.jumpVelocity;
      this.grounded = false;
    }
    this.velocityY += this.gravity * dt;
    this.position.y += this.velocityY * dt;

    if (this.position.y <= ground) {
      this.position.y = ground;
      this.velocityY = 0;
      this.grounded = true;
    }
  }

  // Compute the spring-arm camera position + look target.
  cameraTransform(): { eye: Vec3; target: Vec3 } {
    const headY = this.position.y + 1.6;
    const target = new Vec3(this.position.x, headY, this.position.z);
    const cp = Math.cos(this.camPitch), sp = Math.sin(this.camPitch);
    const offset = new Vec3(
      Math.sin(this.camYaw) * cp,
      sp,
      Math.cos(this.camYaw) * cp,
    ).mul(this.camDist);
    let eye = target.sub(offset);
    // keep camera above terrain
    const minY = this.groundAt(eye.x, eye.z) + 0.6;
    if (eye.y < minY) eye.y = minY;
    return { eye, target };
  }
}

function lerpAngle(a: number, b: number, t: number): number {
  let diff = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (diff < -Math.PI) diff += Math.PI * 2;
  return a + diff * t;
}
