import { Mat4 } from '../math/Mat4';
import { Vec3 } from '../math/Vec3';
import { Camera } from '../scene/Camera';

export type GizmoAxis = 'x'|'y'|'z'|'none';

const UBO_SIZE = 144; // viewProj(64) + model(64) + color(16)

// Arrow: thin box shaft + pyramid tip, all local-space along +X, then rotated
function buildArrow(): { verts: Float32Array; idx: Uint16Array } {
  const verts: number[] = [];
  const idx:   number[] = [];
  const seg = 8;
  const shaftR = 0.025, shaftL = 0.72, tipR = 0.07, tipL = 0.28;

  // Shaft cylinder
  const base0 = 0;
  for (let i=0; i<=seg; i++) {
    const a = (i/seg)*Math.PI*2;
    const x=Math.cos(a)*shaftR, z=Math.sin(a)*shaftR;
    verts.push(x,0,z, x,0,z);     // bottom
    verts.push(x,shaftL,z, x,0,z);// top
  }
  for (let i=0; i<seg; i++) {
    const b=base0+i*2;
    idx.push(b,b+2,b+1, b+1,b+2,b+3);
  }

  // Cone tip
  const base1 = (seg+1)*2;
  verts.push(0,shaftL,0, 0,-1,0); // tip center bottom
  verts.push(0,1,0, 0,1,0);        // tip apex
  for (let i=0; i<=seg; i++) {
    const a = (i/seg)*Math.PI*2;
    const x=Math.cos(a)*tipR, z=Math.sin(a)*tipR;
    verts.push(x,shaftL,z, x,-0.5,z);
    verts.push(x,shaftL,z, x,0.5,z);
  }
  const cBase = base1+2;
  for (let i=0; i<seg; i++) {
    const b = cBase+i*2;
    // bottom cap
    idx.push(base1, b+2, b);
    // side
    idx.push(b,b+2,b+1, b+1,b+2,b+3);
    // apex
    idx.push(base1+1, b+1, b+3);
  }

  return { verts: new Float32Array(verts), idx: new Uint16Array(idx) };
}

// Small cube for center handle
function buildCube(): { verts: Float32Array; idx: Uint16Array } {
  const h = 0.055;
  const faces=[
    [-h,-h, h,0,0,1],[ h,-h, h,0,0,1],[ h, h, h,0,0,1],[-h, h, h,0,0,1],
    [ h,-h,-h,0,0,-1],[-h,-h,-h,0,0,-1],[-h, h,-h,0,0,-1],[ h, h,-h,0,0,-1],
    [-h, h, h,0,1,0],[ h, h, h,0,1,0],[ h, h,-h,0,1,0],[-h, h,-h,0,1,0],
    [-h,-h,-h,0,-1,0],[ h,-h,-h,0,-1,0],[ h,-h, h,0,-1,0],[-h,-h, h,0,-1,0],
    [ h,-h, h,1,0,0],[ h,-h,-h,1,0,0],[ h, h,-h,1,0,0],[ h, h, h,1,0,0],
    [-h,-h,-h,-1,0,0],[-h,-h, h,-1,0,0],[-h, h, h,-1,0,0],[-h, h,-h,-1,0,0],
  ];
  const i: number[]=[]; for(let f=0;f<6;f++){const b=f*4;i.push(b,b+1,b+2,b,b+2,b+3);}
  return { verts:new Float32Array(faces.flat()), idx:new Uint16Array(i) };
}

export class GizmoRenderer {
  private device:  GPUDevice;
  private pipeline!:  GPURenderPipeline;
  private bgl!:    GPUBindGroupLayout;

  // Arrow geometry (shared, rotated per axis via model matrix)
  private arrowVBuf!: GPUBuffer;
  private arrowIBuf!: GPUBuffer;
  private arrowICount = 0;

  private cubeVBuf!: GPUBuffer;
  private cubeIBuf!: GPUBuffer;
  private cubeICount = 0;

  // Per-axis UBOs + bind groups
  private axisUBO: GPUBuffer[] = [];
  private axisBG:  GPUBindGroup[] = [];
  private centerUBO!: GPUBuffer;
  private centerBG!:  GPUBindGroup;

  constructor(device: GPUDevice) { this.device = device; }

  build(canvasFmt: GPUTextureFormat) {
    const dev = this.device;

    const code = dev.createShaderModule({ code:`
struct U { viewProj: mat4x4<f32>, model: mat4x4<f32>, color: vec4<f32> }
@group(0) @binding(0) var<uniform> u: U;
@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
  return u.viewProj * u.model * vec4<f32>(p, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return u.color; }
` });

    this.bgl = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT, buffer:{} }
    ]});
    this.pipeline = dev.createRenderPipeline({
      label:'gizmo',
      layout: dev.createPipelineLayout({ bindGroupLayouts:[this.bgl] }),
      vertex:{ module:code, entryPoint:'vs',
        buffers:[{ arrayStride:24, attributes:[
          { shaderLocation:0, offset:0, format:'float32x3' },
          // normal at offset 12 unused
        ]}] },
      fragment:{ module:code, entryPoint:'fs', targets:[{ format:canvasFmt }] },
      primitive:{ topology:'triangle-list', cullMode:'none' },
      // No depth state → always renders on top
    });

    // Build geometry
    const arrow = buildArrow();
    const cube  = buildCube();
    this.arrowICount = arrow.idx.length;
    this.cubeICount  = cube.idx.length;

    const mkVBuf = (d: Float32Array) => {
      const b = dev.createBuffer({ size:d.byteLength, usage:GPUBufferUsage.VERTEX, mappedAtCreation:true });
      new Float32Array(b.getMappedRange()).set(d); b.unmap(); return b;
    };
    const mkIBuf = (d: Uint16Array) => {
      const b = dev.createBuffer({ size:Math.max(d.byteLength,4), usage:GPUBufferUsage.INDEX, mappedAtCreation:true });
      new Uint16Array(b.getMappedRange()).set(d); b.unmap(); return b;
    };
    this.arrowVBuf = mkVBuf(arrow.verts);
    this.arrowIBuf = mkIBuf(arrow.idx);
    this.cubeVBuf  = mkVBuf(cube.verts);
    this.cubeIBuf  = mkIBuf(cube.idx);

    // Create UBOs for 3 axes + center
    for (let i=0; i<3; i++) {
      const ubo = dev.createBuffer({ size:UBO_SIZE, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
      const bg  = dev.createBindGroup({ layout:this.bgl, entries:[{ binding:0, resource:{ buffer:ubo } }] });
      this.axisUBO.push(ubo);
      this.axisBG.push(bg);
    }
    this.centerUBO = dev.createBuffer({ size:UBO_SIZE, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    this.centerBG  = dev.createBindGroup({ layout:this.bgl, entries:[{ binding:0, resource:{ buffer:this.centerUBO } }] });
  }

  // ── Hit testing ───────────────────────────────────────────────
  hitTest(
    mouseNX: number, mouseNY: number,
    objPos: Vec3, camera: Camera, aspect: number
  ): GizmoAxis {
    const vp = camera.getVP(aspect);
    const scale = this._gizmoScale(objPos, camera);
    const axes: { dir: Vec3; label: GizmoAxis }[] = [
      { dir:new Vec3(1,0,0), label:'x' },
      { dir:new Vec3(0,1,0), label:'y' },
      { dir:new Vec3(0,0,1), label:'z' },
    ];
    const project = (p: Vec3): [number,number] => {
      const clip = vp.data;
      const x = clip[0]*p.x+clip[4]*p.y+clip[8]*p.z+clip[12];
      const y = clip[1]*p.x+clip[5]*p.y+clip[9]*p.z+clip[13];
      const w = clip[3]*p.x+clip[7]*p.y+clip[11]*p.z+clip[15];
      return [x/w, y/w];
    };
    const [cx,cy] = project(objPos);
    let best: GizmoAxis = 'none'; let bestD = 0.08; // threshold in NDC
    for (const ax of axes) {
      const tip = objPos.add(ax.dir.mul(scale));
      const [tx,ty] = project(tip);
      // Distance from mouse to line segment (center→tip)
      const dx=tx-cx, dy=ty-cy;
      const len2 = dx*dx+dy*dy;
      if (len2<1e-8) continue;
      const t = Math.max(0,Math.min(1,((mouseNX-cx)*dx+(mouseNY-cy)*dy)/len2));
      const px=cx+t*dx-mouseNX, py=cy+t*dy-mouseNY;
      const d = Math.sqrt(px*px+py*py);
      if (d<bestD) { bestD=d; best=ax.label; }
    }
    return best;
  }

  private _gizmoScale(objPos: Vec3, camera: Camera): number {
    return camera.position.sub(objPos).length() * 0.14;
  }

  // ── Render into canvas pass (already begun) ───────────────────
  render(
    pass:       GPURenderPassEncoder,
    objPos:     Vec3,
    camera:     Camera,
    viewProj:   Mat4,
    hovered:    GizmoAxis,
  ) {
    const scale = this._gizmoScale(objPos, camera);
    const T = Mat4.translation(objPos);

    const axisData: { rot: Mat4; color: [number,number,number,number] }[] = [
      { rot: Mat4.rotationZ(-Math.PI/2), color:[1.0,0.18,0.18,1] }, // X → red
      { rot: new Mat4(),                 color:[0.18,1.0,0.18,1] }, // Y → green
      { rot: Mat4.rotationX( Math.PI/2), color:[0.18,0.45,1.0, 1] },// Z → blue
    ];
    const axisLabels: GizmoAxis[] = ['x','y','z'];

    const u = new Float32Array(UBO_SIZE/4);
    pass.setPipeline(this.pipeline);

    // 3 arrows
    for (let i=0; i<3; i++) {
      const { rot, color } = axisData[i];
      const S = Mat4.scale(new Vec3(scale,scale,scale));
      const M = T.multiply(rot).multiply(S);
      u.set(viewProj.data, 0);
      u.set(M.data, 16);
      const hover = hovered===axisLabels[i];
      u.set([hover?1.0:color[0], hover?0.95:color[1], hover?0.1:color[2], 1.0], 32);
      this.device.queue.writeBuffer(this.axisUBO[i], 0, u);
      pass.setBindGroup(0, this.axisBG[i]);
      pass.setVertexBuffer(0, this.arrowVBuf);
      pass.setIndexBuffer(this.arrowIBuf, 'uint16');
      pass.drawIndexed(this.arrowICount);
    }

    // Center handle
    {
      const S2 = Mat4.scale(new Vec3(scale,scale,scale));
      const M = T.multiply(S2);
      u.set(viewProj.data, 0);
      u.set(M.data, 16);
      u.set([0.9,0.9,0.9,1.0], 32);
      this.device.queue.writeBuffer(this.centerUBO, 0, u);
      pass.setBindGroup(0, this.centerBG);
      pass.setVertexBuffer(0, this.cubeVBuf);
      pass.setIndexBuffer(this.cubeIBuf, 'uint16');
      pass.drawIndexed(this.cubeICount);
    }
  }
}
