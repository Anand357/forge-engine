// Blueprint node editor — an Unreal-style visual scripting overlay.
// You drag nodes around and "stretch strings" (wires) from an output port to an
// input port to connect logic. Renders to its own 2D canvas above the viewport;
// fully independent of the WebGPU renderer.

export type PortDir = 'in' | 'out';
export type PortKind = 'exec' | 'data';

export interface Port {
  id: string;
  name: string;
  dir: PortDir;
  kind: PortKind;
}

export interface BPNode {
  id: string;
  title: string;
  x: number; y: number;
  w: number; h: number;
  color: string;          // header color
  inputs: Port[];
  outputs: Port[];
  // optional binding to a scene object id (set via inspector)
  targetId?: string;
}

export interface Wire {
  id: string;
  fromNode: string; fromPort: string;
  toNode:   string; toPort:   string;
  kind: PortKind;
}

interface NodeTemplate {
  title: string; color: string;
  inputs: Omit<Port, 'id'>[];
  outputs: Omit<Port, 'id'>[];
}

// Built-in node palette
export const NODE_TEMPLATES: Record<string, NodeTemplate> = {
  'event-begin': {
    title: 'On Begin Play', color: '#c0392b',
    inputs: [],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
  'event-click': {
    title: 'On Clicked', color: '#c0392b',
    inputs: [],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }, { name: 'Target', dir: 'out', kind: 'data' }],
  },
  'event-tick': {
    title: 'On Tick', color: '#c0392b',
    inputs: [],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }, { name: 'DeltaTime', dir: 'out', kind: 'data' }],
  },
  'act-move': {
    title: 'Move Object', color: '#2471a3',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Target', dir: 'in', kind: 'data' }, { name: 'Offset', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
  'act-rotate': {
    title: 'Rotate Object', color: '#2471a3',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Target', dir: 'in', kind: 'data' }, { name: 'Speed', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
  'act-spawn': {
    title: 'Spawn Object', color: '#2471a3',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'At', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }, { name: 'Spawned', dir: 'out', kind: 'data' }],
  },
  'act-destroy': {
    title: 'Destroy Object', color: '#2471a3',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Target', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
  'act-light': {
    title: 'Set Light', color: '#b7950b',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Intensity', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
  'flow-branch': {
    title: 'Branch', color: '#7d3c98',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Condition', dir: 'in', kind: 'data' }],
    outputs: [{ name: 'True', dir: 'out', kind: 'exec' }, { name: 'False', dir: 'out', kind: 'exec' }],
  },
  'flow-delay': {
    title: 'Delay', color: '#7d3c98',
    inputs: [{ name: '', dir: 'in', kind: 'exec' }, { name: 'Seconds', dir: 'in', kind: 'data' }],
    outputs: [{ name: '', dir: 'out', kind: 'exec' }],
  },
};

const HEADER_H = 26;
const PORT_GAP = 20;
const PORT_R = 5;
const NODE_W = 168;

export class Blueprint {
  canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  nodes: BPNode[] = [];
  wires: Wire[] = [];

  // view transform (pan + zoom)
  private panX = 0; private panY = 0; private zoom = 1;

  // interaction state
  private draggingNode: BPNode | null = null;
  private dragDX = 0; private dragDY = 0;
  private panning = false; private panSX = 0; private panSY = 0;
  private linking: { node: BPNode; port: Port; x: number; y: number } | null = null;
  private mouseX = 0; private mouseY = 0;

  onSelectNode?: (node: BPNode | null) => void;
  selectedNode: BPNode | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this._wire();
  }

  // ── Node sizing ──────────────────────────────────────────────────
  private nodeHeight(n: BPNode): number {
    const rows = Math.max(n.inputs.length, n.outputs.length);
    return HEADER_H + rows * PORT_GAP + 12;
  }

  addNode(templateKey: string, x: number, y: number): BPNode {
    const t = NODE_TEMPLATES[templateKey];
    if (!t) throw new Error('unknown node template ' + templateKey);
    const node: BPNode = {
      id: crypto.randomUUID(),
      title: t.title, color: t.color,
      x, y, w: NODE_W, h: 0,
      inputs:  t.inputs.map(p => ({ ...p, id: crypto.randomUUID() })),
      outputs: t.outputs.map(p => ({ ...p, id: crypto.randomUUID() })),
    };
    node.h = this.nodeHeight(node);
    this.nodes.push(node);
    return node;
  }

  removeNode(id: string) {
    this.nodes = this.nodes.filter(n => n.id !== id);
    this.wires = this.wires.filter(w => w.fromNode !== id && w.toNode !== id);
    if (this.selectedNode?.id === id) { this.selectedNode = null; this.onSelectNode?.(null); }
  }

  clear() { this.nodes = []; this.wires = []; this.selectedNode = null; }

  // ── Port world positions ─────────────────────────────────────────
  private portPos(n: BPNode, p: Port): { x: number; y: number } {
    const list = p.dir === 'in' ? n.inputs : n.outputs;
    const idx = list.indexOf(p);
    const y = n.y + HEADER_H + 12 + idx * PORT_GAP;
    const x = p.dir === 'in' ? n.x : n.x + n.w;
    return { x, y };
  }

  private screenToWorld(sx: number, sy: number): { x: number; y: number } {
    return { x: (sx - this.panX) / this.zoom, y: (sy - this.panY) / this.zoom };
  }

  private hitPort(wx: number, wy: number): { node: BPNode; port: Port } | null {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      for (const p of [...n.inputs, ...n.outputs]) {
        const pp = this.portPos(n, p);
        if (Math.hypot(pp.x - wx, pp.y - wy) <= PORT_R + 4) return { node: n, port: p };
      }
    }
    return null;
  }

  private hitNode(wx: number, wy: number): BPNode | null {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      if (wx >= n.x && wx <= n.x + n.w && wy >= n.y && wy <= n.y + n.h) return n;
    }
    return null;
  }

  // ── Events ───────────────────────────────────────────────────────
  private _wire() {
    const c = this.canvas;

    c.addEventListener('mousedown', e => {
      const r = c.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const w = this.screenToWorld(sx, sy);

      // Middle / space-less right-button → pan
      if (e.button === 1 || e.button === 2) {
        this.panning = true; this.panSX = sx - this.panX; this.panSY = sy - this.panY;
        e.preventDefault(); return;
      }

      const port = this.hitPort(w.x, w.y);
      if (port && port.port.dir === 'out') {
        // Start stretching a wire from an output port
        this.linking = { node: port.node, port: port.port, x: w.x, y: w.y };
        return;
      }
      if (port && port.port.dir === 'in') {
        // Clicking an input that has a wire → grab and re-stretch it
        const existing = this.wires.find(wr => wr.toNode === port.node.id && wr.toPort === port.port.id);
        if (existing) {
          const fromNode = this.nodes.find(n => n.id === existing.fromNode)!;
          const fromPort = fromNode.outputs.find(p => p.id === existing.fromPort)!;
          this.wires = this.wires.filter(wr => wr.id !== existing.id);
          this.linking = { node: fromNode, port: fromPort, x: w.x, y: w.y };
          return;
        }
      }

      const node = this.hitNode(w.x, w.y);
      if (node) {
        // bring to front + select + drag
        this.nodes = this.nodes.filter(n => n !== node); this.nodes.push(node);
        this.selectedNode = node; this.onSelectNode?.(node);
        this.draggingNode = node; this.dragDX = w.x - node.x; this.dragDY = w.y - node.y;
      } else {
        this.selectedNode = null; this.onSelectNode?.(null);
        this.panning = true; this.panSX = sx - this.panX; this.panSY = sy - this.panY;
      }
    });

    window.addEventListener('mousemove', e => {
      if (!this.canvas.offsetParent) return; // overlay hidden
      const r = c.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      this.mouseX = sx; this.mouseY = sy;
      const w = this.screenToWorld(sx, sy);

      if (this.draggingNode) {
        this.draggingNode.x = w.x - this.dragDX;
        this.draggingNode.y = w.y - this.dragDY;
      } else if (this.linking) {
        this.linking.x = w.x; this.linking.y = w.y;
      } else if (this.panning) {
        this.panX = sx - this.panSX; this.panY = sy - this.panSY;
      }
    });

    window.addEventListener('mouseup', e => {
      if (this.linking) {
        const r = c.getBoundingClientRect();
        const sx = e.clientX - r.left, sy = e.clientY - r.top;
        const w = this.screenToWorld(sx, sy);
        const target = this.hitPort(w.x, w.y);
        if (target && target.port.dir === 'in' &&
            target.port.kind === this.linking.port.kind &&
            target.node !== this.linking.node) {
          // Remove any existing wire into that input (one connection per input)
          this.wires = this.wires.filter(wr => !(wr.toNode === target.node.id && wr.toPort === target.port.id));
          this.wires.push({
            id: crypto.randomUUID(),
            fromNode: this.linking.node.id, fromPort: this.linking.port.id,
            toNode: target.node.id, toPort: target.port.id,
            kind: this.linking.port.kind,
          });
        }
      }
      this.draggingNode = null; this.linking = null; this.panning = false;
    });

    c.addEventListener('contextmenu', e => e.preventDefault());

    c.addEventListener('wheel', e => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const before = this.screenToWorld(sx, sy);
      this.zoom = Math.max(0.35, Math.min(2.2, this.zoom * (1 - e.deltaY * 0.0012)));
      // keep cursor anchored
      this.panX = sx - before.x * this.zoom;
      this.panY = sy - before.y * this.zoom;
    }, { passive: false });
  }

  resize(w: number, h: number) {
    this.canvas.width = w; this.canvas.height = h;
  }

  // ── Render ───────────────────────────────────────────────────────
  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0d0d18';
    ctx.fillRect(0, 0, W, H);
    this._drawGrid();

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Wires (under nodes)
    for (const wire of this.wires) {
      const fn = this.nodes.find(n => n.id === wire.fromNode);
      const tn = this.nodes.find(n => n.id === wire.toNode);
      if (!fn || !tn) continue;
      const fp = fn.outputs.find(p => p.id === wire.fromPort);
      const tp = tn.inputs.find(p => p.id === wire.toPort);
      if (!fp || !tp) continue;
      const a = this.portPos(fn, fp), b = this.portPos(tn, tp);
      this._drawWire(a.x, a.y, b.x, b.y, wire.kind === 'exec' ? '#e8e8f0' : '#3dc8ff');
    }

    // In-progress link
    if (this.linking) {
      const a = this.portPos(this.linking.node, this.linking.port);
      this._drawWire(a.x, a.y, this.linking.x, this.linking.y,
        this.linking.port.kind === 'exec' ? '#ffffff' : '#3dffc0');
    }

    // Nodes
    for (const n of this.nodes) this._drawNode(n);

    ctx.restore();
  }

  private _drawGrid() {
    const ctx = this.ctx;
    const step = 28 * this.zoom;
    if (step < 6) return;
    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 1;
    const ox = this.panX % step, oy = this.panY % step;
    ctx.beginPath();
    for (let x = ox; x < this.canvas.width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, this.canvas.height); }
    for (let y = oy; y < this.canvas.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(this.canvas.width, y); }
    ctx.stroke();
  }

  private _drawWire(x1: number, y1: number, x2: number, y2: number, color: string) {
    const ctx = this.ctx;
    const dx = Math.max(40, Math.abs(x2 - x1) * 0.5);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2);
    ctx.stroke();
  }

  private _drawNode(n: BPNode) {
    const ctx = this.ctx;
    const sel = this.selectedNode === n;

    // Body
    ctx.fillStyle = 'rgba(22,22,38,0.96)';
    this._roundRect(n.x, n.y, n.w, n.h, 8);
    ctx.fill();

    // Header
    ctx.fillStyle = n.color;
    this._roundRectTop(n.x, n.y, n.w, HEADER_H, 8);
    ctx.fill();

    // Border
    ctx.strokeStyle = sel ? '#ffc83d' : 'rgba(255,255,255,0.12)';
    ctx.lineWidth = sel ? 2 : 1;
    this._roundRect(n.x, n.y, n.w, n.h, 8);
    ctx.stroke();

    // Title
    ctx.fillStyle = '#fff';
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(n.title, n.x + 10, n.y + HEADER_H / 2);
    if (n.targetId) {
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '9px Inter, system-ui, sans-serif';
      ctx.fillText('▸ bound', n.x + n.w - 44, n.y + HEADER_H / 2);
    }

    // Ports
    ctx.font = '11px Inter, system-ui, sans-serif';
    for (const p of n.inputs)  this._drawPort(n, p);
    for (const p of n.outputs) this._drawPort(n, p);
  }

  private _drawPort(n: BPNode, p: Port) {
    const ctx = this.ctx;
    const pos = this.portPos(n, p);
    const color = p.kind === 'exec' ? '#e8e8f0' : '#3dc8ff';

    if (p.kind === 'exec') {
      // triangle (exec pin)
      ctx.fillStyle = color;
      ctx.beginPath();
      const s = PORT_R;
      ctx.moveTo(pos.x - s, pos.y - s);
      ctx.lineTo(pos.x + s, pos.y);
      ctx.lineTo(pos.x - s, pos.y + s);
      ctx.closePath(); ctx.fill();
    } else {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(pos.x, pos.y, PORT_R, 0, Math.PI * 2); ctx.fill();
    }

    if (p.name) {
      ctx.fillStyle = 'rgba(230,230,245,0.8)';
      ctx.textBaseline = 'middle';
      if (p.dir === 'in') { ctx.textAlign = 'left';  ctx.fillText(p.name, pos.x + 10, pos.y); }
      else                { ctx.textAlign = 'right'; ctx.fillText(p.name, pos.x - 10, pos.y); }
      ctx.textAlign = 'left';
    }
  }

  private _roundRect(x: number, y: number, w: number, h: number, r: number) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  private _roundRectTop(x: number, y: number, w: number, h: number, r: number) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ── Serialization (for save/export) ──────────────────────────────
  toJSON() {
    return { nodes: this.nodes, wires: this.wires };
  }
}
