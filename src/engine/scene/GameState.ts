// Lightweight gameplay layer that runs during Play mode.
// Reads tagged scene nodes (gameRole) and tracks score / health / objectives.
// - collectible : touch to gain score & remove
// - hazard      : touch to lose health
// - checkpoint  : touch in sequence to progress
// - goal        : touch when all collectibles gathered → win

import { SceneNode } from './Scene';
import { Vec3 } from '../math/Vec3';

export interface GameSnapshot {
  active: boolean;
  score: number;
  health: number;
  maxHealth: number;
  collected: number;
  totalCollectibles: number;
  checkpoints: number;
  totalCheckpoints: number;
  time: number;          // seconds elapsed
  status: 'playing' | 'won' | 'lost';
  message: string;
}

export class GameState {
  active = false;
  score = 0;
  health = 100;
  maxHealth = 100;
  time = 0;
  status: 'playing' | 'won' | 'lost' = 'playing';
  message = '';

  private collected = new Set<string>();
  private checkpointsHit = new Set<string>();
  private hazardCooldown = 0;

  // Snapshot used for the HUD; recomputed each frame.
  private totalCollectibles = 0;
  private totalCheckpoints = 0;

  start(nodes: SceneNode[]) {
    this.active = true;
    this.score = 0;
    this.health = this.maxHealth;
    this.time = 0;
    this.status = 'playing';
    this.message = '';
    this.collected.clear();
    this.checkpointsHit.clear();
    this.hazardCooldown = 0;
    this.totalCollectibles = nodes.filter(n => n.gameRole === 'collectible').length;
    this.totalCheckpoints  = nodes.filter(n => n.gameRole === 'checkpoint').length;
  }

  stop() { this.active = false; }

  // Called each play frame. `onRemove` lets the engine despawn collected nodes.
  update(dt: number, playerPos: Vec3, nodes: SceneNode[], onRemove: (id: string) => void): boolean {
    if (!this.active || this.status !== 'playing') return false;
    this.time += dt;
    this.hazardCooldown = Math.max(0, this.hazardCooldown - dt);
    let changed = false;

    for (const n of nodes) {
      if (n.gameRole === 'none') continue;
      const dx = n.position.x - playerPos.x;
      const dz = n.position.z - playerPos.z;
      const dy = n.position.y - playerPos.y;
      const reach = Math.max(1.0, Math.max(n.scale.x, n.scale.z) * 0.6) + 0.8;
      const near = (dx*dx + dz*dz) < reach*reach && Math.abs(dy) < 3.0;
      if (!near) continue;

      if (n.gameRole === 'collectible' && !this.collected.has(n.id)) {
        this.collected.add(n.id);
        this.score += 10;
        onRemove(n.id);
        changed = true;
      } else if (n.gameRole === 'hazard' && this.hazardCooldown <= 0) {
        this.health -= 20;
        this.hazardCooldown = 1.0;
        changed = true;
        if (this.health <= 0) { this.health = 0; this.status = 'lost'; this.message = '💀 You died!'; }
      } else if (n.gameRole === 'checkpoint' && !this.checkpointsHit.has(n.id)) {
        this.checkpointsHit.add(n.id);
        this.score += 25;
        changed = true;
      } else if (n.gameRole === 'goal') {
        if (this.collected.size >= this.totalCollectibles) {
          this.status = 'won';
          this.message = '🏆 You win!';
          changed = true;
        }
      }
    }

    // Win condition: all collectibles + checkpoints with no explicit goal
    if (this.status === 'playing' && this.totalCollectibles > 0 &&
        this.collected.size >= this.totalCollectibles &&
        this.checkpointsHit.size >= this.totalCheckpoints &&
        nodes.every(n => n.gameRole !== 'goal')) {
      this.status = 'won';
      this.message = '🏆 All collected — you win!';
      changed = true;
    }

    return changed;
  }

  snapshot(): GameSnapshot {
    return {
      active: this.active,
      score: this.score,
      health: this.health,
      maxHealth: this.maxHealth,
      collected: this.collected.size,
      totalCollectibles: this.totalCollectibles,
      checkpoints: this.checkpointsHit.size,
      totalCheckpoints: this.totalCheckpoints,
      time: this.time,
      status: this.status,
      message: this.message,
    };
  }
}
