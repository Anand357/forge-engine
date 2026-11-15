// Semi-realistic vehicle controller for Play mode.
// Bicycle-ish car model: engine/brake force along facing, speed-dependent
// steering, lateral grip that allows drift when pushed, rolling resistance,
// and terrain ground-follow. A chase camera trails behind the car.

import { Vec3 } from '../math/Vec3';

export interface VehicleInput {
  throttle: boolean;   // W
  brake: boolean;      // S (brake / reverse)
  steerLeft: boolean;  // A
  steerRight: boolean; // D
  handbrake: boolean;  // Space
}

export class VehicleController {
  position = new Vec3(0, 0, 0);
  yaw = 0;                       // heading (radians)
  velocity = new Vec3(0, 0, 0);  // world-space planar velocity
  velY = 0;                      // vertical (for hops/landing)
  grounded = true;

  // chase camera
  camYaw = 0;
  camPitch = 0.28;
  camDist = 9;
  private camYawManual = false;

  // ── Tuning (semi-realistic) ──
  enginePower = 26;       // accel force
  reversePower = 14;
  brakeForce = 38;
  maxSpeed = 38;
  maxReverse = 12;
  steerRate = 2.2;        // rad/s at low speed
  grip = 7.5;             // lateral grip (higher = less slide)
  handbrakeGrip = 1.6;    // reduced grip → drift
  rollResist = 1.4;       // rolling resistance
  drag = 0.45;            // air drag (∝ v²)
  gravity = -24;

  groundAt: (x: number, z: number) => number;
  collide?: (from: Vec3, to: Vec3, r: number, h: number) => Vec3;
  radius = 1.1;

  constructor(groundAt: (x: number, z: number) => number, start = new Vec3(0, 0, 0), yaw = 0) {
    this.groundAt = groundAt;
    this.position = start.clone();
    this.position.y = this.groundAt(start.x, start.z);
    this.yaw = yaw;
    this.camYaw = yaw;
  }

  addMouseDelta(dx: number, dy: number) {
    this.camYaw -= dx * 0.0035;
    this.camPitch = Math.max(-0.1, Math.min(1.0, this.camPitch + dy * 0.003));
    this.camYawManual = true;
  }
  zoom(deltaY: number) {
    this.camDist = Math.max(5, Math.min(22, this.camDist * (1 + deltaY * 0.001)));
  }

  get speed(): number { return Math.hypot(this.velocity.x, this.velocity.z); }

  update(dt: number, input: VehicleInput) {
    // Forward & right basis on ground plane
    const fwd = new Vec3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
    const right = new Vec3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));

    // Decompose velocity into longitudinal (along fwd) and lateral (along right)
    let vLong = this.velocity.x * fwd.x + this.velocity.z * fwd.z;
    let vLat  = this.velocity.x * right.x + this.velocity.z * right.z;

    // ── Engine / brake ──
    if (input.throttle) {
      vLong += this.enginePower * dt;
    } else if (input.brake) {
      if (vLong > 0.2) vLong -= this.brakeForce * dt;       // braking
      else vLong -= this.reversePower * dt;                 // reverse
    }
    // Rolling resistance + air drag
    vLong -= vLong * this.rollResist * dt;
    vLong -= vLong * Math.abs(vLong) * this.drag * 0.01 * dt;
    vLong = Math.max(-this.maxReverse, Math.min(this.maxSpeed, vLong));

    // ── Steering (speed-sensitive; needs motion to turn) ──
    let steer = 0;
    if (input.steerLeft)  steer += 1;
    if (input.steerRight) steer -= 1;
    const speedFactor = Math.min(1, Math.abs(vLong) / 8);   // hard to steer when slow/stopped
    const dirSign = vLong >= 0 ? 1 : -1;
    this.yaw += steer * this.steerRate * speedFactor * dirSign * dt;

    // ── Lateral grip / drift ──
    const grip = input.handbrake ? this.handbrakeGrip : this.grip;
    vLat -= vLat * Math.min(1, grip * dt);                  // bleed sideways velocity
    if (input.handbrake) vLong -= vLong * 0.8 * dt;         // handbrake slows too

    // Recompose world velocity
    this.velocity = new Vec3(
      fwd.x * vLong + right.x * vLat,
      0,
      fwd.z * vLong + right.z * vLat,
    );

    // Integrate position (with solid-object collision)
    const desired = this.position.add(this.velocity.mul(dt));
    if (this.collide) {
      const corrected = this.collide(this.position, desired, this.radius, 1.0);
      // if we got blocked, bleed off speed (impact)
      const blockedX = Math.abs(corrected.x - desired.x) > 1e-4;
      const blockedZ = Math.abs(corrected.z - desired.z) > 1e-4;
      if (blockedX || blockedZ) this.velocity = this.velocity.mul(0.4);
      this.position = corrected;
    } else {
      this.position = desired;
    }

    // ── Terrain follow / gravity ──
    const ground = this.groundAt(this.position.x, this.position.z);
    this.velY += this.gravity * dt;
    this.position.y += this.velY * dt;
    if (this.position.y <= ground) {
      this.position.y = ground;
      this.velY = 0;
      this.grounded = true;
    } else {
      this.grounded = false;
    }
  }

  // Chase camera that auto-follows heading (unless the player dragged it).
  cameraTransform(): { eye: Vec3; target: Vec3 } {
    if (!this.camYawManual) {
      // ease camera yaw toward car heading
      const diff = wrapAngle(this.yaw - this.camYaw);
      this.camYaw += diff * 0.08;
    }
    const target = new Vec3(this.position.x, this.position.y + 1.5, this.position.z);
    const cp = Math.cos(this.camPitch), sp = Math.sin(this.camPitch);
    const offset = new Vec3(Math.sin(this.camYaw) * cp, sp, Math.cos(this.camYaw) * cp).mul(this.camDist);
    let eye = target.sub(offset);
    const minY = this.groundAt(eye.x, eye.z) + 1.0;
    if (eye.y < minY) eye.y = minY;
    return { eye, target };
  }
}

function wrapAngle(a: number): number {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}
