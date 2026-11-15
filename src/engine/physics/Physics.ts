import RAPIER from '@dimforge/rapier3d-compat';
import { Vec3 } from '../math/Vec3';
import { SceneNode } from '../scene/Scene';

export class PhysicsWorld {
  private world!: RAPIER.World;
  private bodies = new Map<string, RAPIER.RigidBody>();
  private ready = false;

  async init() {
    await RAPIER.init();
    // Use object form to avoid deprecated positional params
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 }); // ✅ correct
    this.ready = true;
  }

  isReady() { return this.ready; }

  setGravity(y: number) {
    if (this.ready) this.world.gravity = { x: 0, y, z: 0 };
  }

  addBody(node: SceneNode) {
    if (!this.ready) return;
    // Remove existing body first to avoid duplicates
    this.removeBody(node.id);

    const desc = node.isStatic
      ? RAPIER.RigidBodyDesc.fixed()
      : RAPIER.RigidBodyDesc.dynamic();
    desc.setTranslation(node.position.x, node.position.y, node.position.z);
    const body = this.world.createRigidBody(desc);

    let col: RAPIER.ColliderDesc;
    switch (node.type) {
      case 'sphere':
        col = RAPIER.ColliderDesc.ball(Math.max(node.scale.x, node.scale.y, node.scale.z) * 0.5);
        break;
      case 'cylinder':
        col = RAPIER.ColliderDesc.cylinder(node.scale.y * 0.5, node.scale.x * 0.5);
        break;
      case 'terrain':
        col = RAPIER.ColliderDesc.cuboid(40, 0.5, 40);
        break;
      default:
        col = RAPIER.ColliderDesc.cuboid(node.scale.x * 0.5, node.scale.y * 0.5, node.scale.z * 0.5);
    }
    col.setFriction(0.7).setRestitution(0.3);
    this.world.createCollider(col, body);
    this.bodies.set(node.id, body);
    node.physicsHandle = body;
  }

  removeBody(id: string) {
    const b = this.bodies.get(id);
    if (b) { this.world.removeRigidBody(b); this.bodies.delete(id); }
  }

  step(dt: number) {
    if (!this.ready) return;
    this.world.timestep = Math.min(dt, 0.05);
    this.world.step();
  }

  syncToScene(nodes: SceneNode[]) {
    for (const n of nodes) {
      const b = this.bodies.get(n.id);
      if (!b || n.isStatic) continue;
      const t = b.translation();
      n.position.x = t.x; n.position.y = t.y; n.position.z = t.z;
      const r = b.rotation();
      n.rotation.y = Math.atan2(2 * (r.w * r.y + r.x * r.z), 1 - 2 * (r.y * r.y + r.z * r.z));
      n.rotation.x = Math.asin(Math.max(-1, Math.min(1, 2 * (r.w * r.x - r.y * r.z))));
    }
  }

  applyImpulse(id: string, force: Vec3) {
    const b = this.bodies.get(id);
    if (b) b.applyImpulse({ x: force.x, y: force.y, z: force.z }, true);
  }
}
