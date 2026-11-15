// Lightweight AABB collision for kinematic player/vehicle controllers.
// Objects register an axis-aligned box; the controller resolves its desired
// move against them with wall-sliding, so you can't walk/drive through props.

import { Vec3 } from '../math/Vec3';

export interface Box {
  id: string;
  min: Vec3;
  max: Vec3;
}

export class CollisionWorld {
  boxes: Box[] = [];

  clear() { this.boxes.length = 0; }

  // Rebuild from a list of {id, position, scale} (call when play starts / scene changes).
  rebuild(items: { id: string; position: Vec3; scale: Vec3; skip?: boolean }[]) {
    this.boxes.length = 0;
    for (const it of items) {
      if (it.skip) continue;
      // half-extents from scale (objects are ~unit sized, scaled)
      const hx = Math.max(0.2, it.scale.x * 0.5);
      const hy = Math.max(0.2, it.scale.y * 0.5);
      const hz = Math.max(0.2, it.scale.z * 0.5);
      this.boxes.push({
        id: it.id,
        min: new Vec3(it.position.x - hx, it.position.y - hy, it.position.z - hz),
        max: new Vec3(it.position.x + hx, it.position.y + hy, it.position.z + hz),
      });
    }
  }

  // Resolve a horizontal move for a vertical capsule (radius r, height h) from
  // `from` to the desired `to`. Returns the corrected position (wall-sliding).
  // Vertical (y) is handled by the caller via terrain; we only block XZ.
  resolveMove(from: Vec3, to: Vec3, r: number, h: number): Vec3 {
    let px = to.x, pz = to.z;
    const footY = to.y;          // base of capsule
    const headY = to.y + h;

    // Resolve each axis independently for stable sliding.
    // X axis first
    for (const b of this.boxes) {
      if (headY < b.min.y || footY > b.max.y) continue;   // no vertical overlap
      // expand box by radius
      const minX = b.min.x - r, maxX = b.max.x + r;
      const minZ = b.min.z - r, maxZ = b.max.z + r;
      if (pz > minZ && pz < maxZ && px > minX && px < maxX) {
        // pushed out along X toward whichever side is nearer to `from`
        if (from.x <= minX) px = minX;
        else if (from.x >= maxX) px = maxX;
      }
    }
    // Z axis (using already-corrected px)
    for (const b of this.boxes) {
      if (headY < b.min.y || footY > b.max.y) continue;
      const minX = b.min.x - r, maxX = b.max.x + r;
      const minZ = b.min.z - r, maxZ = b.max.z + r;
      if (px > minX && px < maxX && pz > minZ && pz < maxZ) {
        if (from.z <= minZ) pz = minZ;
        else if (from.z >= maxZ) pz = maxZ;
      }
    }
    return new Vec3(px, to.y, pz);
  }

  // Highest box top under a point (for standing on objects). Returns -Infinity if none.
  supportHeightAt(x: number, z: number, r: number): number {
    let top = -Infinity;
    for (const b of this.boxes) {
      if (x >= b.min.x - r && x <= b.max.x + r && z >= b.min.z - r && z <= b.max.z + r) {
        if (b.max.y > top) top = b.max.y;
      }
    }
    return top;
  }
}
