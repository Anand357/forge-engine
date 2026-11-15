import { Mat4 } from '../math/Mat4';
import { Vec3 } from '../math/Vec3';
import { PointLight } from '../scene/Light';

export interface RenderObject {
  vertexBuffer:       GPUBuffer;
  indexBuffer:        GPUBuffer;
  indexFormat?:       GPUIndexFormat;   // defaults to 'uint16'
  uniformBuffer:      GPUBuffer;   // main pass per-object UBO
  bindGroup:          GPUBindGroup;
  shadowUniformBuffer:GPUBuffer;   // shadow pass per-object UBO
  shadowBindGroup:    GPUBindGroup;
  prepassBindGroup?:  GPUBindGroup;// normal/depth prepass per-object (reuses uniformBuffer)
  indexCount:         number;
  modelMatrix:        Mat4;
  color:              [number,number,number];
  metalness:          number;
  roughness:          number;
  emissive:           number;
  opacity:            number;
  selected:           boolean;
  textured:           number;      // 0 = flat color, 1 = triplanar procedural texture
  material?:          number;      // 0 generic, 1 terrain (natural blend), 2 water
}

// ── UBO sizes ────────────────────────────────────────────────────
// Frame: 4×mat4(256) + 10×vec4(160) + plPos[4](64) + plColor[4](64) = 544
//   vec4s: camPos,camRight,camUp,camForward,sunDir,sunColor,ambient,fog,extras,quality
const FRAME_UBO  = 544;
// Object: model(64)+albedo(16)+pbr(16)+flags(16) = 112
const OBJ_UBO   = 112;
// Shadow object: model(64) = 64
const SHDW_OBJ  = 64;
// Shadow frame: lightVP(64) = 64
const SHDW_FRAME = 64;

const SHADOW_SIZE = 2048;

// ── WGSL shared snippets ─────────────────────────────────────────
const FRAME_STRUCT = /* wgsl */`
struct Frame {
  viewProj     : mat4x4<f32>,
  lightVP      : mat4x4<f32>,
  invViewProj  : mat4x4<f32>,
  view         : mat4x4<f32>,
  camPos       : vec4<f32>,
  camRight     : vec4<f32>,
  camUp        : vec4<f32>,
  camForward   : vec4<f32>,
  sunDir       : vec4<f32>,
  sunColor     : vec4<f32>,
  ambientColor : vec4<f32>,
  fog          : vec4<f32>,   // x=density y=heightFalloff z=startDist  w=enabled
  extras       : vec4<f32>,   // x=time y=numLights z=ssr w=ssao
  quality      : vec4<f32>,   // x=preset(0/1/2) y=clouds z=godrays w=water
  plPos        : array<vec4<f32>,4>,
  plColor      : array<vec4<f32>,4>,
}`;

const PBR_FUNCS = /* wgsl */`
const PI : f32 = 3.14159265;

fn D_GGX(NdotH: f32, alpha: f32) -> f32 {
  let a2 = alpha*alpha;
  let d  = NdotH*NdotH*(a2-1.0)+1.0;
  return a2 / (PI*d*d + 1e-6);
}
fn G_Schlick(NdotX: f32, rough: f32) -> f32 {
  let k = (rough+1.0)*(rough+1.0)*0.125;
  return NdotX / (NdotX*(1.0-k)+k+1e-6);
}
fn F_Schlick(cosT: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0-F0)*pow(clamp(1.0-cosT,0.0,1.0),5.0);
}
fn brdf(N:vec3<f32>,L:vec3<f32>,V:vec3<f32>,albedo:vec3<f32>,metal:f32,rough:f32) -> vec3<f32> {
  let H     = normalize(L+V);
  let NdotL = max(dot(N,L),0.0);
  let NdotV = max(dot(N,V),0.001);
  let NdotH = max(dot(N,H),0.0);
  let VdotH = max(dot(V,H),0.0);
  let F0 = mix(vec3<f32>(0.04),albedo,metal);
  let alpha = rough*rough;
  let D  = D_GGX(NdotH,alpha);
  let G  = G_Schlick(NdotV,rough)*G_Schlick(NdotL,rough);
  let F  = F_Schlick(VdotH,F0);
  let kD = (vec3<f32>(1.0)-F)*(1.0-metal);
  return (kD*albedo/PI + D*G*F/max(4.0*NdotV*NdotL,0.001))*NdotL;
}`;

// Procedural value-noise + triplanar texturing (no external files).
const NOISE_FUNCS = /* wgsl */`
fn hash2(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}
fn vnoise(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f*f*(3.0-2.0*f);
  let a = hash2(i); let b = hash2(i+vec2<f32>(1.0,0.0));
  let c = hash2(i+vec2<f32>(0.0,1.0)); let d = hash2(i+vec2<f32>(1.0,1.0));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
fn fbm(p: vec2<f32>) -> f32 {
  var v = 0.0; var amp = 0.5; var pp = p;
  for (var i=0; i<5; i++) { v += amp*vnoise(pp); pp *= 2.03; amp *= 0.5; }
  return v;
}
// Triplanar detail: returns a slightly varying albedo multiplier + bump on normal
fn triplanarDetail(wp: vec3<f32>, N: vec3<f32>, scale: f32) -> f32 {
  let bw = abs(N);
  let w  = bw / (bw.x + bw.y + bw.z + 1e-4);
  let nx = fbm(wp.zy * scale);
  let ny = fbm(wp.xz * scale);
  let nz = fbm(wp.xy * scale);
  return nx*w.x + ny*w.y + nz*w.z;
}`;

export class Renderer {
  device!:  GPUDevice;
  canvas:   HTMLCanvasElement;
  private ctx!:  GPUCanvasContext;
  private fmt!:  GPUTextureFormat;

  // Feature toggles (driven from engine)
  ssaoEnabled = true;
  ssrEnabled  = true;
  fogEnabled  = true;
  cloudsEnabled   = true;
  godraysEnabled  = true;
  waterEnabled    = true;
  quality = 2;          // 0 Low, 1 Med, 2 Ultra
  wetness = 0;          // 0 dry … 1 rain-soaked (boosts reflections)

  // Shadow
  private shadowDepthTex!: GPUTexture;
  private shadowDepthView!: GPUTextureView;
  private shadowSampler!:   GPUSampler;
  private shadowFrameUBO!:  GPUBuffer;
  private shadowFrameBGL!:  GPUBindGroupLayout;
  private shadowFrameBG!:   GPUBindGroup;
  private shadowObjBGL!:    GPUBindGroupLayout;
  private shadowPipeline!:  GPURenderPipeline;

  // G-buffer (normal-prepass): rgba16f = worldNormal.xyz + linearDepth
  private gbufTex!:   GPUTexture;
  private gbufView!:  GPUTextureView;
  private prepassDepthTex!:  GPUTexture;
  private prepassDepthView!: GPUTextureView;
  private prepassPipeline!:  GPURenderPipeline;
  private prepassFrameBGL!:  GPUBindGroupLayout;
  private prepassFrameBG!:   GPUBindGroup;
  private prepassObjBGL!:    GPUBindGroupLayout;

  // SSAO
  private ssaoTex!:   GPUTexture;
  private ssaoView!:  GPUTextureView;
  private ssaoBlurTex!:  GPUTexture;
  private ssaoBlurView!: GPUTextureView;
  private ssaoPipeline!:  GPURenderPipeline;
  private ssaoBGL!:    GPUBindGroupLayout;
  private ssaoBG!:     GPUBindGroup;
  private ssaoBlurPipeline!: GPURenderPipeline;
  private ssaoBlurBGL!:  GPUBindGroupLayout;
  private ssaoBlurBG!:   GPUBindGroup;

  // HDR
  private hdrTex!:    GPUTexture;
  private hdrView!:   GPUTextureView;
  private depthTex!:  GPUTexture;
  private depthView!: GPUTextureView;

  // SSR (reads hdr + gbuffer, writes to hdrTex2)
  private hdrTex2!:   GPUTexture;
  private hdrView2!:  GPUTextureView;
  private ssrPipeline!: GPURenderPipeline;
  private ssrBGL!:    GPUBindGroupLayout;
  private ssrBG!:     GPUBindGroup;

  // God rays (reads hdr2 + gbuffer, writes hdrTex3)
  private hdrTex3!:   GPUTexture;
  private hdrView3!:  GPUTextureView;
  private godrayPipeline!: GPURenderPipeline;
  private godrayBGL!: GPUBindGroupLayout;
  private godrayBG!:  GPUBindGroup;

  // LDR (tonemapped) → FXAA → canvas
  private ldrTex!:   GPUTexture;
  private ldrView!:  GPUTextureView;
  private fxaaPipeline!: GPURenderPipeline;
  private fxaaBGL!:  GPUBindGroupLayout;
  private fxaaBG!:   GPUBindGroup;

  // Bloom intermediates
  private bloomExtTex!:  GPUTexture;
  private bloomExtView!: GPUTextureView;
  private bloomBlrTex!:  GPUTexture;
  private bloomBlrView!: GPUTextureView;

  // Frame UBO (global, one per frame)
  private frameUBO!: GPUBuffer;
  private frameBGL!: GPUBindGroupLayout;
  private frameBG!:  GPUBindGroup;   // rebuilt on resize (texture bindings change)
  private objBGL!:   GPUBindGroupLayout;

  // Post-process UBO
  private postUBO!:  GPUBuffer;

  // Pipelines
  private mainPipeline!:    GPURenderPipeline;
  private skyPipeline!:     GPURenderPipeline;
  private gridPipeline!:    GPURenderPipeline;
  private bloomExtPipeline!:GPURenderPipeline;
  private bloomHPipeline!:  GPURenderPipeline;
  private bloomVPipeline!:  GPURenderPipeline;
  private tonemapPipeline!: GPURenderPipeline;

  // Bloom bind groups (rebuilt on resize)
  private bloomExtBGL!: GPUBindGroupLayout;
  private bloomExtBG!:  GPUBindGroup;
  private bloomHBG!:    GPUBindGroup;
  private bloomVBG!:    GPUBindGroup;
  private tonemapBGL!:  GPUBindGroupLayout;
  private tonemapBG!:   GPUBindGroup;

  // Grid
  private gridBuf!:   GPUBuffer;
  private gridCount = 0;

  // Samplers
  private linearSampler!: GPUSampler;

  private cW = 0; private cH = 0;

  constructor(canvas: HTMLCanvasElement) { this.canvas = canvas; }

  async init(): Promise<boolean> {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference:'high-performance' });
    if (!adapter) return false;
    this.device = await adapter.requestDevice();
    this.ctx = this.canvas.getContext('webgpu') as GPUCanvasContext;
    this.fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.fmt, alphaMode:'opaque' });

    this.linearSampler = this.device.createSampler({ minFilter:'linear', magFilter:'linear' });
    this.shadowSampler = this.device.createSampler({ compare:'less' });

    this._buildShadowTex();
    this._buildPipelines();
    this._buildGrid(160, 4);
    this._buildPostUBO();
    this._ensureHDR();
    return true;
  }

  // ── Shadow depth texture (fixed 2048×2048) ────────────────────
  private _buildShadowTex() {
    this.shadowDepthTex = this.device.createTexture({
      size:[SHADOW_SIZE,SHADOW_SIZE], format:'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowDepthView = this.shadowDepthTex.createView();
  }

  // ── All render pipelines ──────────────────────────────────────
  private _buildPipelines() {
    const dev = this.device;

    // ════════════════════════════════════════════════════════════
    // Shadow pipeline
    // ════════════════════════════════════════════════════════════
    const shdwVS = dev.createShaderModule({ code:`
struct SF { lightVP: mat4x4<f32> }
struct SO { model:   mat4x4<f32> }
@group(0) @binding(0) var<uniform> sf: SF;
@group(1) @binding(0) var<uniform> so: SO;
@vertex fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
  return sf.lightVP * so.model * vec4<f32>(pos, 1.0);
}` });

    this.shadowFrameBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX, buffer:{} }
    ]});
    this.shadowObjBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX, buffer:{} }
    ]});
    this.shadowFrameUBO = dev.createBuffer({ size:SHDW_FRAME, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    this.shadowFrameBG  = dev.createBindGroup({ layout:this.shadowFrameBGL,
      entries:[{ binding:0, resource:{ buffer:this.shadowFrameUBO } }] });

    this.shadowPipeline = dev.createRenderPipeline({
      label:'shadow', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.shadowFrameBGL,this.shadowObjBGL] }),
      vertex:{ module:shdwVS, entryPoint:'vs',
        buffers:[{ arrayStride:24, attributes:[{ shaderLocation:0, offset:0, format:'float32x3' }] }] },
      depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less' },
      primitive:{ topology:'triangle-list', cullMode:'front' },
    });

    // ════════════════════════════════════════════════════════════
    // Frame BGL (shared by main pass — has shadow + ssao bindings)
    // ════════════════════════════════════════════════════════════
    this.frameBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT, buffer:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, texture:{ sampleType:'depth' } },
      { binding:2, visibility:GPUShaderStage.FRAGMENT, sampler:{ type:'comparison' } },
      { binding:3, visibility:GPUShaderStage.FRAGMENT, texture:{} },          // ssao
      { binding:4, visibility:GPUShaderStage.FRAGMENT, sampler:{} },          // linear
    ]});
    this.objBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT, buffer:{} }
    ]});
    this.frameUBO = dev.createBuffer({ size:FRAME_UBO, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });

    // Single-UBO BGL reused by sky / grid / prepass-frame / ssao-style frame access
    const frameOnlyBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT, buffer:{} },
    ]});

    // ════════════════════════════════════════════════════════════
    // Normal/Depth prepass → G-buffer (worldNormal.xyz, linearDepth)
    // ════════════════════════════════════════════════════════════
    const prepassCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
struct Obj { model: mat4x4<f32>, albedo: vec4<f32>, pbr: vec4<f32>, flags: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> obj: Obj;
struct VO { @builtin(position) pos: vec4<f32>, @location(0) nor: vec3<f32>, @location(1) vz: f32 }
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VO {
  let wp = obj.model * vec4<f32>(p,1.0);
  let cp = frame.viewProj * wp;
  let vp = frame.view * wp;
  let nor = normalize((obj.model * vec4<f32>(n,0.0)).xyz);
  return VO(cp, nor, -vp.z);
}
@fragment fn fs(in: VO) -> @location(0) vec4<f32> {
  return vec4<f32>(normalize(in.nor), in.vz);
}` });
    this.prepassFrameBGL = frameOnlyBGL;
    this.prepassObjBGL   = this.objBGL;
    this.prepassFrameBG  = dev.createBindGroup({ layout:this.prepassFrameBGL,
      entries:[{ binding:0, resource:{ buffer:this.frameUBO } }] });
    this.prepassPipeline = dev.createRenderPipeline({
      label:'prepass', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.prepassFrameBGL, this.prepassObjBGL] }),
      vertex:{ module:prepassCode, entryPoint:'vs',
        buffers:[{ arrayStride:24, attributes:[
          { shaderLocation:0, offset:0,  format:'float32x3' },
          { shaderLocation:1, offset:12, format:'float32x3' },
        ]}] },
      fragment:{ module:prepassCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list', cullMode:'back' },
      depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less' },
    });

    // ════════════════════════════════════════════════════════════
    // SSAO pass (samples G-buffer normal+depth)
    // ════════════════════════════════════════════════════════════
    this.ssaoBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, buffer:{} },   // frame
      { binding:1, visibility:GPUShaderStage.FRAGMENT, texture:{} },  // gbuffer
      { binding:2, visibility:GPUShaderStage.FRAGMENT, sampler:{} },  // linear
    ]});
    const ssaoCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var gbuf: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(p[vi],0.0,1.0);
}
fn hash3(p: vec2<f32>) -> f32 { return fract(sin(dot(p,vec2<f32>(12.9898,78.233)))*43758.5453); }
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(gbuf));
  let uv  = pos.xy / dim;
  // textureSampleLevel (explicit LOD) is allowed in non-uniform control flow
  let g   = textureSampleLevel(gbuf, samp, uv, 0.0);
  let depth = g.w;
  let N = normalize(g.xyz);
  let radius = 1.4;
  var occ = 0.0;
  let rnd = hash3(pos.xy)*6.2831;
  for (var i=0; i<12; i++) {
    let ang = rnd + f32(i)*0.5236;
    let r   = (f32(i)+1.0)/12.0;
    let off = vec2<f32>(cos(ang), sin(ang)) * (radius / max(depth,1.0)) * r * 0.05;
    let sg  = textureSampleLevel(gbuf, samp, uv + off, 0.0);
    let sd  = sg.w;
    let valid = select(0.0, 1.0, sd > 0.0);
    let diff = depth - sd;                              // positive => sample closer to camera
    let rangeCheck = smoothstep(0.0, 1.0, radius / max(abs(diff),1e-3));
    occ += valid * step(0.025, diff) * rangeCheck;
  }
  occ = 1.0 - (occ / 12.0) * 1.4;
  // Background (no geometry) = fully lit, no occlusion
  return vec4<f32>(select(clamp(occ, 0.0, 1.0), 1.0, depth <= 0.0));
}` });
    this.ssaoPipeline = dev.createRenderPipeline({
      label:'ssao', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.ssaoBGL] }),
      vertex:{ module:ssaoCode, entryPoint:'vs' },
      fragment:{ module:ssaoCode, entryPoint:'fs', targets:[{ format:'r8unorm' }] },
      primitive:{ topology:'triangle-list' },
    });

    // SSAO blur (4x4 box) → r8
    this.ssaoBlurBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, sampler:{} },
    ]});
    const ssaoBlurCode = dev.createShaderModule({ code:`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(p[vi],0.0,1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(src));
  let uv  = pos.xy / dim;
  let ts  = 1.0/dim;
  var s = 0.0;
  for (var x=-2;x<=2;x++) { for (var y=-2;y<=2;y++) {
    s += textureSample(src, samp, uv + vec2<f32>(f32(x),f32(y))*ts).x;
  }}
  return vec4<f32>(s/25.0);
}` });
    this.ssaoBlurPipeline = dev.createRenderPipeline({
      label:'ssao-blur', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.ssaoBlurBGL] }),
      vertex:{ module:ssaoBlurCode, entryPoint:'vs' },
      fragment:{ module:ssaoBlurCode, entryPoint:'fs', targets:[{ format:'r8unorm' }] },
      primitive:{ topology:'triangle-list' },
    });

    // ════════════════════════════════════════════════════════════
    // Sky pipeline
    // ════════════════════════════════════════════════════════════
    const skyCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var<uniform> frame: Frame;
struct VO { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  let p = array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return VO(vec4<f32>(p[vi], 0.9999, 1.0), p[vi]);
}

// ── 3D value noise + fbm for volumetric clouds ──
fn h3(p: vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}
fn vn3(p: vec3<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f*f*(3.0-2.0*f);
  let n000 = h3(i+vec3<f32>(0,0,0)); let n100 = h3(i+vec3<f32>(1,0,0));
  let n010 = h3(i+vec3<f32>(0,1,0)); let n110 = h3(i+vec3<f32>(1,1,0));
  let n001 = h3(i+vec3<f32>(0,0,1)); let n101 = h3(i+vec3<f32>(1,0,1));
  let n011 = h3(i+vec3<f32>(0,1,1)); let n111 = h3(i+vec3<f32>(1,1,1));
  let x00 = mix(n000,n100,u.x); let x10 = mix(n010,n110,u.x);
  let x01 = mix(n001,n101,u.x); let x11 = mix(n011,n111,u.x);
  return mix(mix(x00,x10,u.y), mix(x01,x11,u.y), u.z);
}
fn cloudFbm(p: vec3<f32>) -> f32 {
  var v = 0.0; var a = 0.5; var pp = p;
  for (var i=0; i<5; i++) { v += a*vn3(pp); pp = pp*2.02; a *= 0.5; }
  return v;
}
// Cloud density at a world point inside the cloud slab.
fn cloudDensity(p: vec3<f32>) -> f32 {
  let t = frame.extras.x * 0.012;
  let wp = p * 0.0025 + vec3<f32>(t, 0.0, t*0.6);
  var d = cloudFbm(wp);
  d = d * 1.4 - 0.62;                 // coverage threshold
  // erode with higher-freq detail
  d -= cloudFbm(wp*3.1) * 0.18;
  return clamp(d, 0.0, 1.0);
}

@fragment fn fs(in: VO) -> @location(0) vec4<f32> {
  let dir = normalize(frame.camForward.xyz + frame.camRight.xyz*in.ndc.x + frame.camUp.xyz*in.ndc.y);
  let sun = normalize(frame.sunDir.xyz);
  let up  = clamp(dir.y, -1.0, 1.0);

  // Rayleigh-ish gradient
  let zenith  = vec3<f32>(0.18, 0.34, 0.64);
  let horizon = vec3<f32>(0.72, 0.82, 0.92);
  let t = pow(clamp(1.0 - up, 0.0, 1.0), 0.55);
  var sky = mix(zenith, horizon, t);

  let sunUp = clamp(sun.y, 0.0, 1.0);
  let warm  = vec3<f32>(1.0, 0.78, 0.55);
  sky = mix(sky * warm, sky, smoothstep(0.0, 0.35, sunUp));

  if (up < 0.0) {
    sky = mix(vec3<f32>(0.20,0.17,0.14), horizon, exp(up*6.0));
  }

  // sun disk + glow
  let sd   = dot(dir, sun);
  let disk = smoothstep(0.9996, 0.99995, sd);
  let glow = pow(max(sd,0.0), 8.0) * 0.6 + pow(max(sd,0.0), 280.0) * 3.0;
  sky += disk * frame.sunColor.xyz * 12.0 + glow * frame.sunColor.xyz;

  // ── Volumetric clouds (raymarch a slab at altitude) ──
  if (frame.quality.y > 0.5 && dir.y > 0.02) {
    let slabLo = 280.0; let slabHi = 620.0;
    let tEnter = slabLo / dir.y;
    let tExit  = slabHi / dir.y;
    // step count scales with quality preset
    let steps  = select(select(18, 30, frame.quality.x > 0.5), 48, frame.quality.x > 1.5);
    let dt = (tExit - tEnter) / f32(steps);
    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    let camPos = frame.camPos.xyz;
    for (var i=0; i<48; i++) {
      if (i >= steps) { break; }
      let tt = tEnter + (f32(i)+0.5)*dt;
      let pos = camPos + dir*tt;
      let dens = cloudDensity(pos);
      if (dens > 0.01) {
        // light march toward sun (cheap 3-tap)
        var shadow = 0.0;
        for (var j=1; j<=3; j++) { shadow += cloudDensity(pos + sun*f32(j)*40.0); }
        let light = exp(-shadow * 0.9);
        let powder = 1.0 - exp(-dens * 2.0);
        let lit = mix(vec3<f32>(0.55,0.6,0.7), frame.sunColor.xyz*1.3, light) * powder;
        let a = dens * 0.5;
        scattered += transmittance * lit * a;
        transmittance *= (1.0 - a);
        if (transmittance < 0.02) { break; }
      }
    }
    let cloudFade = smoothstep(0.02, 0.18, dir.y);  // fade out near horizon
    // composite clouds over sky: sky attenuated by transmittance + in-scattered light
    let blended = sky * transmittance + scattered;
    sky = mix(sky, blended, cloudFade);
  }

  // horizon haze
  let haze = frame.fog.w * pow(clamp(1.0 - abs(up), 0.0, 1.0), 4.0) * 0.22;
  sky = mix(sky, vec3<f32>(0.66, 0.74, 0.84), haze);

  return vec4<f32>(sky, 1.0);
}` });
    this.skyPipeline = dev.createRenderPipeline({
      label:'sky', layout: dev.createPipelineLayout({ bindGroupLayouts:[frameOnlyBGL] }),
      vertex:{ module:skyCode, entryPoint:'vs' },
      fragment:{ module:skyCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
      depthStencil:{ format:'depth24plus', depthWriteEnabled:false, depthCompare:'always' },
    });

    // ════════════════════════════════════════════════════════════
    // Main PBR pipeline (shadow + SSAO + triplanar texture + fog)
    // ════════════════════════════════════════════════════════════
    const mainCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
${PBR_FUNCS}
${NOISE_FUNCS}
struct Obj { model: mat4x4<f32>, albedo: vec4<f32>, pbr: vec4<f32>, flags: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(0) @binding(3) var ssaoTex: texture_2d<f32>;
@group(0) @binding(4) var linSamp: sampler;
@group(1) @binding(0) var<uniform> obj: Obj;

struct VO {
  @builtin(position) pos: vec4<f32>,
  @location(0) wp: vec3<f32>,
  @location(1) nor: vec3<f32>,
}
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VO {
  let wp = obj.model * vec4<f32>(p,1.0);
  return VO(frame.viewProj * wp, wp.xyz, normalize((obj.model*vec4<f32>(n,0.0)).xyz));
}

fn sampleShadow(wp: vec3<f32>) -> f32 {
  let lc    = frame.lightVP * vec4<f32>(wp, 1.0);
  let uv    = lc.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let depth = lc.z - 0.003;
  let ts    = 1.0 / 2048.0;
  var s     = 0.0;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      s += textureSampleCompare(shadowMap, shadowSampler,
             uv + vec2<f32>(f32(x)*ts, f32(y)*ts), depth);
    }
  }
  let inBounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && lc.z >= 0.0 && lc.z <= 1.0;
  return select(0.0, s / 9.0, inBounds);
}

@fragment fn fs(in: VO) -> @location(0) vec4<f32> {
  var N       = normalize(in.nor);
  let V       = normalize(frame.camPos.xyz - in.wp);
  var albedo  = obj.albedo.xyz;
  let metal   = obj.pbr.x;
  var rough   = obj.pbr.y;
  let emStr   = obj.pbr.z;
  let sel     = obj.pbr.w;
  let opacity = obj.albedo.w;
  let textured = obj.flags.x;
  let matKind  = obj.flags.y;     // 0 generic, 1 terrain natural, 2 water
  var waterFresnel = 0.0;

  // ── Animated water surface ──
  if (matKind > 1.5 && frame.quality.w > 0.5) {
    let tw = frame.extras.x;
    // sum of moving noise octaves → wave height field, then derive normal
    let p = in.wp.xz;
    let w1 = fbm(p*0.12 + vec2<f32>(tw*0.35, tw*0.18));
    let w2 = fbm(p*0.34 - vec2<f32>(tw*0.22, tw*0.30));
    let wave = (w1*0.7 + w2*0.3);
    // normal from finite differences of the wave field
    let e = 0.6;
    let hX = fbm((p+vec2<f32>(e,0.0))*0.12 + vec2<f32>(tw*0.35, tw*0.18))*0.7
           + fbm((p+vec2<f32>(e,0.0))*0.34 - vec2<f32>(tw*0.22, tw*0.30))*0.3;
    let hZ = fbm((p+vec2<f32>(0.0,e))*0.12 + vec2<f32>(tw*0.35, tw*0.18))*0.7
           + fbm((p+vec2<f32>(0.0,e))*0.34 - vec2<f32>(tw*0.22, tw*0.30))*0.3;
    let bump = vec3<f32>((wave-hX), 1.0, (wave-hZ)) * vec3<f32>(2.2, 1.0, 2.2);
    N = normalize(vec3<f32>(0.0,1.0,0.0) + vec3<f32>(bump.x, 0.0, bump.z));
    // deep vs shallow tint
    let deep    = vec3<f32>(0.02, 0.10, 0.18);
    let shallow = vec3<f32>(0.10, 0.34, 0.42);
    albedo = mix(deep, shallow, clamp(wave*1.4, 0.0, 1.0));
    rough  = 0.04;
    // fresnel: more reflective at grazing angles
    waterFresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    // foam on wave crests
    let foam = smoothstep(0.62, 0.78, wave);
    albedo = mix(albedo, vec3<f32>(0.85,0.9,0.95), foam*0.6);
  } else if (matKind > 0.5 && matKind < 1.5) {
    // ── Natural terrain material: blend grass / rock / sand / snow ──
    let slope  = clamp(1.0 - N.y, 0.0, 1.0);            // 0 flat, 1 vertical
    let height = in.wp.y;
    // base palette
    let grass = vec3<f32>(0.26, 0.40, 0.18);
    let rock  = vec3<f32>(0.34, 0.32, 0.30);
    let sand  = vec3<f32>(0.62, 0.55, 0.38);
    let snow  = vec3<f32>(0.92, 0.94, 0.98);
    // procedural breakup
    let n1 = triplanarDetail(in.wp, N, 0.6);
    let n2 = triplanarDetail(in.wp, N, 3.0);
    let grassV = grass * (0.75 + n1 * 0.5);
    let rockV  = rock  * (0.7 + n2 * 0.6);
    var c = mix(grassV, rockV, smoothstep(0.28, 0.62, slope + n2 * 0.2));
    // sand near/below water line
    c = mix(sand, c, smoothstep(1.0, 3.2, height));
    // snow on high flat peaks
    let snowMask = smoothstep(16.0, 26.0, height) * (1.0 - smoothstep(0.4, 0.75, slope));
    c = mix(c, snow, snowMask);
    albedo = c;
    rough  = clamp(mix(0.95, 0.6, snowMask), 0.4, 1.0);
    // micro-normal bump
    let e = 0.2;
    let gx = triplanarDetail(in.wp + vec3<f32>(e,0,0), N, 3.0) - n2;
    let gz = triplanarDetail(in.wp + vec3<f32>(0,0,e), N, 3.0) - n2;
    N = normalize(N - vec3<f32>(gx, 0.0, gz) * 2.0);
  } else if (textured > 0.5) {
    // rich multi-octave procedural surface for textured objects
    let d1 = triplanarDetail(in.wp, N, 1.4);
    let d2 = triplanarDetail(in.wp, N, 5.5);
    let d3 = triplanarDetail(in.wp, N, 16.0);
    let detail = d1*0.5 + d2*0.32 + d3*0.18;
    albedo = albedo * (0.78 + detail*0.5);
    rough  = clamp(rough + (detail-0.5)*0.35, 0.04, 1.0);
    // layered normal bump (coarse + fine)
    let e = 0.12;
    let gx = (triplanarDetail(in.wp + vec3<f32>(e,0,0), N, 5.5) - d2)
           + (triplanarDetail(in.wp + vec3<f32>(e,0,0), N, 16.0) - d3)*0.5;
    let gz = (triplanarDetail(in.wp + vec3<f32>(0,0,e), N, 5.5) - d2)
           + (triplanarDetail(in.wp + vec3<f32>(0,0,e), N, 16.0) - d3)*0.5;
    N = normalize(N - vec3<f32>(gx, 0.0, gz) * 1.8);
  } else if (matKind < 0.5) {
    // subtle micro-surface on plain flat-color objects so they read as real materials
    let m1 = triplanarDetail(in.wp, N, 9.0);
    rough  = clamp(rough + (m1-0.5)*0.18, 0.03, 1.0);
    let e = 0.1;
    let mx = triplanarDetail(in.wp + vec3<f32>(e,0,0), N, 9.0) - m1;
    let mz = triplanarDetail(in.wp + vec3<f32>(0,0,e), N, 9.0) - m1;
    N = normalize(N - vec3<f32>(mx, 0.0, mz) * 0.5);
  }

  // ── Rain wetness: darker, smoother, more reflective (water pools) ──
  let wet = frame.ambientColor.w;
  if (wet > 0.01 && matKind < 1.5) {
    // upward-facing surfaces collect more water
    let pool = clamp(N.y, 0.0, 1.0) * wet;
    albedo = albedo * (1.0 - pool*0.45);          // wet darkens
    rough  = mix(rough, 0.08, pool);              // wet smooths → reflective
  }

  // Sun (lit fraction from PCF). Warm key light.
  let L = normalize(frame.sunDir.xyz);
  let lit = sampleShadow(in.wp);
  let sunWarm = frame.sunColor.xyz * vec3<f32>(1.05, 1.0, 0.92);
  let sunC = brdf(N,L,V,albedo,metal,rough) * sunWarm * lit * 1.15;

  // SSAO (screen-space ambient occlusion) — sample unconditionally (uniform flow)
  let suv = in.pos.xy / vec2<f32>(textureDimensions(ssaoTex));
  let aoSample = textureSampleLevel(ssaoTex, linSamp, suv, 0.0).x;
  let ao = select(1.0, aoSample, frame.extras.w > 0.5);

  // ── Hemispheric ambient: cool sky from above, warm ground bounce below ──
  let skyAmb = vec3<f32>(0.34, 0.43, 0.58);
  let gndAmb = vec3<f32>(0.26, 0.22, 0.17);
  let hemi   = mix(gndAmb, skyAmb, N.y * 0.5 + 0.5);
  let ambC   = hemi * frame.ambientColor.xyz * 2.4 * albedo * (1.0 - metal*0.7) * ao;

  // Point lights
  var ptC = vec3<f32>(0.0);
  let nL = i32(frame.extras.y);
  for (var i=0;i<4;i++) {
    if (i>=nL) { break; }
    let lp  = frame.plPos[i].xyz;
    let lc2 = frame.plColor[i].xyz;
    let lr  = frame.plColor[i].w;
    let lv  = lp - in.wp;
    let dd  = length(lv);
    if (dd>=lr) { continue; }
    let Lp  = normalize(lv);
    let att = pow(1.0 - dd/lr, 2.0);
    ptC += brdf(N,Lp,V,albedo,metal,rough) * lc2 * att;
  }

  let emissive = albedo * emStr;
  var col = ambC + sunC + ptC + emissive;

  // ── Water sky reflection (cheap): blend toward a sky tint by fresnel ──
  // (full screen-space reflection is added later in the SSR pass)
  if (matKind > 1.5 && frame.quality.w > 0.5) {
    let R = reflect(-V, N);
    let skyRefl = mix(vec3<f32>(0.30,0.45,0.62), vec3<f32>(0.75,0.85,0.95), clamp(R.y,0.0,1.0));
    col = mix(col, skyRefl, clamp(waterFresnel*0.7 + 0.18, 0.0, 0.85));
    // sharp sun glint on the water
    let spec = pow(max(dot(R, L), 0.0), 220.0);
    col += frame.sunColor.xyz * spec * 3.0 * lit;
  }

  // Selection pulse
  if (sel > 0.5) {
    col = mix(col, vec3<f32>(1.0,0.75,0.1), 0.35 + sin(frame.extras.x*5.0)*0.1);
  }

  // ── Height/distance fog ──
  if (frame.fog.w > 0.5) {
    let dist   = length(frame.camPos.xyz - in.wp);
    let start  = frame.fog.z;
    let density= frame.fog.x;
    let hf     = exp(-max(in.wp.y,0.0) * frame.fog.y);     // less fog at altitude
    let fogAmt = (1.0 - exp(-max(dist-start,0.0) * density)) * hf;
    let fogCol = mix(vec3<f32>(0.62,0.70,0.82), frame.sunColor.xyz, 0.15);
    col = mix(col, fogCol, clamp(fogAmt,0.0,1.0));
  }

  return vec4<f32>(col, opacity);
}` });

    this.mainPipeline = dev.createRenderPipeline({
      label:'main', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.frameBGL,this.objBGL] }),
      vertex:{ module:mainCode, entryPoint:'vs',
        buffers:[{ arrayStride:24, attributes:[
          { shaderLocation:0, offset:0,  format:'float32x3' },
          { shaderLocation:1, offset:12, format:'float32x3' },
        ]}] },
      fragment:{ module:mainCode, entryPoint:'fs', targets:[{ format:'rgba16float',
        blend:{ color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha',operation:'add'},
                alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'} } }] },
      primitive:{ topology:'triangle-list', cullMode:'back' },
      depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less' },
    });

    // ════════════════════════════════════════════════════════════
    // Grid pipeline
    // ════════════════════════════════════════════════════════════
    const gridCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var<uniform> frame: Frame;
@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
  return frame.viewProj * vec4<f32>(p,1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.3,0.32,0.45,0.5);
}` });
    this.gridPipeline = dev.createRenderPipeline({
      label:'grid', layout: dev.createPipelineLayout({ bindGroupLayouts:[frameOnlyBGL] }),
      vertex:{ module:gridCode, entryPoint:'vs',
        buffers:[{ arrayStride:12, attributes:[{ shaderLocation:0, offset:0, format:'float32x3' }] }] },
      fragment:{ module:gridCode, entryPoint:'fs', targets:[{ format:'rgba16float',
        blend:{ color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha',operation:'add'},
                alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'} } }] },
      primitive:{ topology:'line-list' },
      depthStencil:{ format:'depth24plus', depthWriteEnabled:false, depthCompare:'less-equal' },
    });

    // ════════════════════════════════════════════════════════════
    // SSR pass (screen-space reflections): hdr + gbuffer → hdr2
    // ════════════════════════════════════════════════════════════
    this.ssrBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, buffer:{} },   // frame
      { binding:1, visibility:GPUShaderStage.FRAGMENT, texture:{} },  // hdr color
      { binding:2, visibility:GPUShaderStage.FRAGMENT, texture:{} },  // gbuffer
      { binding:3, visibility:GPUShaderStage.FRAGMENT, sampler:{} },  // linear
    ]});
    const ssrCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr:  texture_2d<f32>;
@group(0) @binding(2) var gbuf: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(p[vi],0.0,1.0);
}
// reconstruct world position from gbuffer linear view depth + ray through pixel
fn worldFromUV(uv: vec2<f32>, viewZ: f32) -> vec3<f32> {
  let ndc = vec2<f32>(uv.x*2.0-1.0, (1.0-uv.y)*2.0-1.0);
  // direction in world space across the frustum
  let dir = normalize(frame.camForward.xyz + frame.camRight.xyz*ndc.x + frame.camUp.xyz*ndc.y);
  let cosA = dot(dir, normalize(frame.camForward.xyz));
  return frame.camPos.xyz + dir * (viewZ / max(cosA, 1e-3));
}
fn projectToUV(wp: vec3<f32>) -> vec3<f32> {
  let cp = frame.viewProj * vec4<f32>(wp, 1.0);
  if (cp.w <= 0.0) { return vec3<f32>(-1.0); }
  let ndc = cp.xyz / cp.w;
  let uv  = vec2<f32>(ndc.x*0.5+0.5, 0.5-ndc.y*0.5);
  return vec3<f32>(uv, cp.w);   // z carries clip w as a proxy for "in front"
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(hdr));
  let uv  = pos.xy / dim;
  // All samples use explicit-LOD (textureSampleLevel) so they're legal in the
  // marching loop's non-uniform control flow.
  let base = textureSampleLevel(hdr, samp, uv, 0.0).xyz;

  let g = textureSampleLevel(gbuf, samp, uv, 0.0);
  let viewZ = g.w;
  // Sky (no geometry) → pass through untouched
  if (viewZ <= 0.0) { return vec4<f32>(base, 1.0); }
  let N = normalize(g.xyz);

  // ── Screen-space GI: gather bounced light from nearby surfaces ──
  // quality.x toggles sample count; modest bounce keeps it cheap.
  var indirect = vec3<f32>(0.0);
  // Low preset (0) → no GI; Med (1) → 6 taps; Ultra (2) → 12 taps
  let giSamples = select(select(0, 6, frame.quality.x > 0.5), 12, frame.quality.x > 1.5);
  if (giSamples > 0) {
    let radius = 0.045 / max(viewZ * 0.06, 0.4);
    var wsum = 0.0;
    for (var i=0; i<12; i++) {
      if (i >= giSamples) { break; }
      let ang = f32(i) * 2.39996 + viewZ;       // golden-angle spiral
      let r = radius * sqrt((f32(i)+0.5)/f32(giSamples));
      let suv2 = uv + vec2<f32>(cos(ang), sin(ang)) * r;
      let sg = textureSampleLevel(gbuf, samp, suv2, 0.0);
      if (sg.w <= 0.0) { continue; }
      let sN = normalize(sg.xyz);
      // weight: facing toward our normal (light bounces from surfaces we see)
      let wN = max(dot(N, sN), 0.0);
      let depthW = 1.0 - clamp(abs(sg.w - viewZ)*0.25, 0.0, 1.0);
      let sc = textureSampleLevel(hdr, samp, suv2, 0.0).xyz;
      indirect += sc * wN * depthW;
      wsum += 1.0;
    }
    indirect = indirect / max(wsum, 1.0);
  }
  // add a fraction of gathered light as colored indirect bounce
  let giStrength = select(0.35, 0.0, giSamples == 0);
  var lit = base + indirect * giStrength;

  // SSR disabled → return GI-enhanced color
  if (frame.extras.z < 0.5) { return vec4<f32>(lit, 1.0); }

  let wp = worldFromUV(uv, viewZ);
  let V  = normalize(wp - frame.camPos.xyz);
  let R  = reflect(V, N);

  // March in world space
  var p = wp + N*0.05;
  let stepLen = 0.35;
  var hitColor = vec3<f32>(-1.0);
  var fade = 0.0;
  for (var i=0; i<32; i++) {
    p += R*stepLen;
    let pr = projectToUV(p);
    if (pr.x < 0.0 || pr.x > 1.0 || pr.y < 0.0 || pr.y > 1.0 || pr.z <= 0.0) { break; }
    let sceneZ = textureSampleLevel(gbuf, samp, pr.xy, 0.0).w;
    if (sceneZ <= 0.0) { continue; }
    // depth of marched point in view space
    let marchViewZ = -(frame.view * vec4<f32>(p,1.0)).z;
    let diff = marchViewZ - sceneZ;
    if (diff > 0.02 && diff < 1.2) {
      hitColor = textureSampleLevel(hdr, samp, pr.xy, 0.0).xyz;
      fade = (1.0 - f32(i)/32.0) * smoothstep(1.2,0.0,diff);
      break;
    }
  }
  if (hitColor.x < 0.0) { return vec4<f32>(lit, 1.0); }
  // Fresnel-weighted, capped strength
  let fres = pow(1.0 - max(dot(-V, N), 0.0), 4.0);
  let strength = clamp(fade * (0.25 + fres*0.75), 0.0, 0.6);
  return vec4<f32>(mix(lit, hitColor, strength), 1.0);
}` });
    this.ssrPipeline = dev.createRenderPipeline({
      label:'ssr', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.ssrBGL] }),
      vertex:{ module:ssrCode, entryPoint:'vs' },
      fragment:{ module:ssrCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
    });

    // ════════════════════════════════════════════════════════════
    // God rays (radial sun scattering) — hdr2 + gbuffer → hdr3
    // ════════════════════════════════════════════════════════════
    this.godrayBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, buffer:{} },   // frame
      { binding:1, visibility:GPUShaderStage.FRAGMENT, texture:{} },  // hdr color
      { binding:2, visibility:GPUShaderStage.FRAGMENT, texture:{} },  // gbuffer (depth in .w)
      { binding:3, visibility:GPUShaderStage.FRAGMENT, sampler:{} },  // linear
    ]});
    const godrayCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr:  texture_2d<f32>;
@group(0) @binding(2) var gbuf: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(p[vi],0.0,1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(hdr));
  let uv  = pos.xy / dim;
  let base = textureSampleLevel(hdr, samp, uv, 0.0).xyz;
  if (frame.quality.z < 0.5) { return vec4<f32>(base, 1.0); }

  // project sun to screen
  let sunDirW = normalize(frame.sunDir.xyz);
  let sunClip = frame.viewProj * vec4<f32>(frame.camPos.xyz + sunDirW*1000.0, 1.0);
  if (sunClip.w <= 0.0) { return vec4<f32>(base, 1.0); }  // sun behind camera
  let sunUV = vec2<f32>(sunClip.x/sunClip.w*0.5+0.5, 0.5 - sunClip.y/sunClip.w*0.5);

  // radial march from pixel toward the sun, accumulating bright (sky) samples
  let steps = select(select(24, 40, frame.quality.x > 0.5), 64, frame.quality.x > 1.5);
  let delta = (sunUV - uv) / f32(steps);
  var coord = uv;
  var illum = 0.0;
  var decay = 1.0;
  for (var i=0; i<64; i++) {
    if (i >= steps) { break; }
    coord += delta;
    if (coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) { break; }
    let g = textureSampleLevel(gbuf, samp, coord, 0.0);
    let isSky = select(0.0, 1.0, g.w <= 0.0);   // only sky contributes (rays come from sky)
    let s = textureSampleLevel(hdr, samp, coord, 0.0).xyz;
    let lum = dot(s, vec3<f32>(0.2126,0.7152,0.0722));
    illum += lum * isSky * decay;
    decay *= 0.96;
  }
  illum = illum / f32(steps);
  // weight by closeness to the sun & sun being in view
  let toSun = 1.0 - clamp(length(uv - sunUV)*0.8, 0.0, 1.0);
  let shafts = illum * toSun * 1.6;
  return vec4<f32>(base + frame.sunColor.xyz * shafts, 1.0);
}` });
    this.godrayPipeline = dev.createRenderPipeline({
      label:'godrays', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.godrayBGL] }),
      vertex:{ module:godrayCode, entryPoint:'vs' },
      fragment:{ module:godrayCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
    });

    // ════════════════════════════════════════════════════════════
    // Bloom extract
    // ════════════════════════════════════════════════════════════
    this.bloomExtBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, sampler:{} },
    ]});
    const bloomExtCode = dev.createShaderModule({ code:`
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var s:   sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(p[vi],0.0,1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / vec2<f32>(textureDimensions(hdr));
  let c  = textureSample(hdr, s, uv).xyz;
  let lum = dot(c, vec3<f32>(0.2126,0.7152,0.0722));
  let bright = max(lum - 0.9, 0.0) * 1.5;
  return vec4<f32>(c * bright, 1.0);
}` });
    this.bloomExtPipeline = dev.createRenderPipeline({
      label:'bloom-ext', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.bloomExtBGL] }),
      vertex:{ module:bloomExtCode, entryPoint:'vs' },
      fragment:{ module:bloomExtCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
    });

    // Bloom blur (separable Gaussian, direction via UBO)
    const bloomBlrBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, sampler:{} },
      { binding:2, visibility:GPUShaderStage.FRAGMENT, buffer:{} },
    ]});
    const blurCode = dev.createShaderModule({ code:`
struct P { dir: vec2<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var s:   sampler;
@group(0) @binding(2) var<uniform> p: P;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let q=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(q[vi],0.0,1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let sz = vec2<f32>(textureDimensions(src));
  let uv = pos.xy / sz;
  let step = p.dir / sz;
  let w = array<f32,7>(0.0625,0.125,0.1875,0.25,0.1875,0.125,0.0625);
  var c = vec3<f32>(0.0);
  for (var i=0;i<7;i++) {
    c += textureSample(src, s, uv + step*f32(i-3)).xyz * w[i];
  }
  return vec4<f32>(c, 1.0);
}` });
    const blurHDir = dev.createBuffer({ size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const blurVDir = dev.createBuffer({ size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(blurHDir, 0, new Float32Array([1,0,0,0]));
    dev.queue.writeBuffer(blurVDir, 0, new Float32Array([0,1,0,0]));
    (this as any)._blurHDir = blurHDir;
    (this as any)._blurVDir = blurVDir;
    (this as any)._bloomBlrBGL = bloomBlrBGL;

    const blurPL = dev.createPipelineLayout({ bindGroupLayouts:[bloomBlrBGL] });
    this.bloomHPipeline = dev.createRenderPipeline({
      label:'blur-h', layout:blurPL,
      vertex:{ module:blurCode, entryPoint:'vs' },
      fragment:{ module:blurCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
    });
    this.bloomVPipeline = dev.createRenderPipeline({
      label:'blur-v', layout:blurPL,
      vertex:{ module:blurCode, entryPoint:'vs' },
      fragment:{ module:blurCode, entryPoint:'fs', targets:[{ format:'rgba16float' }] },
      primitive:{ topology:'triangle-list' },
    });

    // ════════════════════════════════════════════════════════════
    // Tonemap (reads hdr2 + bloom)
    // ════════════════════════════════════════════════════════════
    this.tonemapBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:2, visibility:GPUShaderStage.FRAGMENT, sampler:{} },
      { binding:3, visibility:GPUShaderStage.FRAGMENT, buffer:{} },
    ]});
    const tonemapCode = dev.createShaderModule({ code:`
struct P { exposure: f32, bloomStr: f32, _p: vec2<f32> }
@group(0) @binding(0) var hdr:   texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var s:     sampler;
@group(0) @binding(3) var<uniform> p: P;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let q=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(q[vi],0.0,1.0);
}
fn aces(x: vec3<f32>) -> vec3<f32> {
  return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3(0.0), vec3(1.0));
}
fn vibrance(c: vec3<f32>, amt: f32) -> vec3<f32> {
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  let mx = max(c.r, max(c.g, c.b));
  let sat = mx - min(c.r, min(c.g, c.b));
  return mix(vec3<f32>(l), c, 1.0 + amt * (1.0 - sat));
}
// Filmic contrast S-curve in linear space
fn filmicContrast(c: vec3<f32>, k: f32) -> vec3<f32> {
  return clamp((c - 0.5) * k + 0.5, vec3(0.0), vec3(1.0));
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(hdr));
  let uv  = pos.xy / dim;
  let col = textureSample(hdr,   s, uv).xyz * p.exposure;
  let blm = textureSample(bloom, s, uv).xyz * p.bloomStr;
  var tm  = aces(col + blm);
  // livelier color
  tm = vibrance(tm, 0.26);
  // cinematic warm/cool split-tone (cool shadows, warm highlights)
  let lum0 = dot(tm, vec3<f32>(0.2126, 0.7152, 0.0722));
  tm = mix(tm * vec3<f32>(0.94, 0.97, 1.06), tm * vec3<f32>(1.06, 1.02, 0.95), smoothstep(0.15, 0.85, lum0));
  // gentle filmic contrast
  tm = filmicContrast(tm, 1.08);
  // vignette
  let v = uv - 0.5;
  let vig = 1.0 - dot(v, v) * 0.52;
  tm *= vig;
  let outc = pow(clamp(tm, vec3(0.0), vec3(1.0)), vec3<f32>(1.0/2.2));
  // store perceptual luma in alpha for FXAA
  let luma = sqrt(dot(outc, vec3<f32>(0.299, 0.587, 0.114)));
  return vec4<f32>(outc, luma);
}` });
    this.tonemapPipeline = dev.createRenderPipeline({
      label:'tonemap', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.tonemapBGL] }),
      vertex:{ module:tonemapCode, entryPoint:'vs' },
      fragment:{ module:tonemapCode, entryPoint:'fs', targets:[{ format:'rgba8unorm' }] },
      primitive:{ topology:'triangle-list' },
    });

    // ════════════════════════════════════════════════════════════
    // FXAA (anti-aliasing) — ldr → canvas. Luma is in ldr alpha.
    // ════════════════════════════════════════════════════════════
    this.fxaaBGL = dev.createBindGroupLayout({ entries:[
      { binding:0, visibility:GPUShaderStage.FRAGMENT, texture:{} },
      { binding:1, visibility:GPUShaderStage.FRAGMENT, sampler:{} },
      { binding:2, visibility:GPUShaderStage.FRAGMENT, buffer:{} },   // frame (for quality)
    ]});
    const fxaaCode = dev.createShaderModule({ code:`
${FRAME_STRUCT}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> frame: Frame;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let q=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
  return vec4<f32>(q[vi],0.0,1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(tex));
  let uv  = pos.xy / dim;
  let rcp = 1.0 / dim;
  let center = textureSampleLevel(tex, samp, uv, 0.0);
  // FXAA disabled on Low preset → passthrough
  if (frame.quality.x < 0.5) { return vec4<f32>(center.rgb, 1.0); }

  let lM  = center.a;
  let lNW = textureSampleLevel(tex, samp, uv + vec2<f32>(-1.0,-1.0)*rcp, 0.0).a;
  let lNE = textureSampleLevel(tex, samp, uv + vec2<f32>( 1.0,-1.0)*rcp, 0.0).a;
  let lSW = textureSampleLevel(tex, samp, uv + vec2<f32>(-1.0, 1.0)*rcp, 0.0).a;
  let lSE = textureSampleLevel(tex, samp, uv + vec2<f32>( 1.0, 1.0)*rcp, 0.0).a;
  let lMin = min(lM, min(min(lNW,lNE), min(lSW,lSE)));
  let lMax = max(lM, max(max(lNW,lNE), max(lSW,lSE)));
  let range = lMax - lMin;
  // skip flat areas
  if (range < max(0.045, lMax*0.125)) { return vec4<f32>(center.rgb, 1.0); }

  var dir = vec2<f32>(
    -((lNW + lNE) - (lSW + lSE)),
     ((lNW + lSW) - (lNE + lSE)),
  );
  let reduce = max((lNW+lNE+lSW+lSE)*0.03125, 0.0078);
  let rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
  dir = clamp(dir * rcpDir, vec2<f32>(-8.0), vec2<f32>(8.0)) * rcp;

  let rgbA = 0.5 * (
    textureSampleLevel(tex, samp, uv + dir*(1.0/3.0 - 0.5), 0.0).rgb +
    textureSampleLevel(tex, samp, uv + dir*(2.0/3.0 - 0.5), 0.0).rgb);
  let rgbB = rgbA*0.5 + 0.25 * (
    textureSampleLevel(tex, samp, uv + dir*(-0.5), 0.0).rgb +
    textureSampleLevel(tex, samp, uv + dir*( 0.5), 0.0).rgb);
  let lB = sqrt(dot(rgbB, vec3<f32>(0.299,0.587,0.114)));
  var outc = rgbB;
  if (lB < lMin || lB > lMax) { outc = rgbA; }
  return vec4<f32>(outc, 1.0);
}` });
    this.fxaaPipeline = dev.createRenderPipeline({
      label:'fxaa', layout: dev.createPipelineLayout({ bindGroupLayouts:[this.fxaaBGL] }),
      vertex:{ module:fxaaCode, entryPoint:'vs' },
      fragment:{ module:fxaaCode, entryPoint:'fs', targets:[{ format:this.fmt }] },
      primitive:{ topology:'triangle-list' },
    });
  }

  private _buildPostUBO() {
    this.postUBO = this.device.createBuffer({ size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.postUBO, 0, new Float32Array([1.2, 0.6, 0, 0]));
  }

  private _buildGrid(size: number, step: number) {
    const lines: number[] = [];
    for (let i=-size; i<=size; i+=step) {
      lines.push(i,0,-size, i,0,size);
      lines.push(-size,0,i, size,0,i);
    }
    this.gridCount = lines.length/3;
    this.gridBuf = this.device.createBuffer({
      size:lines.length*4, usage:GPUBufferUsage.VERTEX, mappedAtCreation:true,
    });
    new Float32Array(this.gridBuf.getMappedRange()).set(lines);
    this.gridBuf.unmap();
  }

  // ── HDR + bloom + gbuffer + ssao textures (recreated on resize) ─
  private _ensureHDR() {
    const w = this.canvas.width, h = this.canvas.height;
    if (w<1||h<1||(w===this.cW&&h===this.cH)) return;
    this.hdrTex?.destroy(); this.hdrTex2?.destroy(); this.hdrTex3?.destroy();
    this.ldrTex?.destroy(); this.depthTex?.destroy();
    this.bloomExtTex?.destroy(); this.bloomBlrTex?.destroy();
    this.gbufTex?.destroy(); this.prepassDepthTex?.destroy();
    this.ssaoTex?.destroy(); this.ssaoBlurTex?.destroy();
    this.cW=w; this.cH=h;

    const mk = (format: GPUTextureFormat) => this.device.createTexture({
      size:[w,h], format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrTex      = mk('rgba16float');   this.hdrView      = this.hdrTex.createView();
    this.hdrTex2     = mk('rgba16float');   this.hdrView2     = this.hdrTex2.createView();
    this.hdrTex3     = mk('rgba16float');   this.hdrView3     = this.hdrTex3.createView();
    this.ldrTex      = mk('rgba8unorm');    this.ldrView      = this.ldrTex.createView();
    this.depthTex    = mk('depth24plus');   this.depthView    = this.depthTex.createView();
    this.bloomExtTex = mk('rgba16float');   this.bloomExtView = this.bloomExtTex.createView();
    this.bloomBlrTex = mk('rgba16float');   this.bloomBlrView = this.bloomBlrTex.createView();
    this.gbufTex     = mk('rgba16float');   this.gbufView     = this.gbufTex.createView();
    this.prepassDepthTex = mk('depth24plus'); this.prepassDepthView = this.prepassDepthTex.createView();
    this.ssaoTex     = mk('r8unorm');       this.ssaoView     = this.ssaoTex.createView();
    this.ssaoBlurTex = mk('r8unorm');       this.ssaoBlurView = this.ssaoBlurTex.createView();

    // Main frame bind group (shadow map + blurred ssao)
    this.frameBG = this.device.createBindGroup({ layout:this.frameBGL, entries:[
      { binding:0, resource:{ buffer:this.frameUBO } },
      { binding:1, resource:this.shadowDepthView },
      { binding:2, resource:this.shadowSampler },
      { binding:3, resource:this.ssaoBlurView },
      { binding:4, resource:this.linearSampler },
    ]});

    // Prepass frame BG (just the UBO)
    this.prepassFrameBG = this.device.createBindGroup({ layout:this.prepassFrameBGL,
      entries:[{ binding:0, resource:{ buffer:this.frameUBO } }] });

    // Sky + grid single-UBO BGs
    (this as any)._skyBG = this.device.createBindGroup({ layout:this.skyPipeline.getBindGroupLayout(0),
      entries:[{ binding:0, resource:{ buffer:this.frameUBO } }] });
    (this as any)._gridBG = this.device.createBindGroup({ layout:this.gridPipeline.getBindGroupLayout(0),
      entries:[{ binding:0, resource:{ buffer:this.frameUBO } }] });

    // SSAO bind group
    this.ssaoBG = this.device.createBindGroup({ layout:this.ssaoBGL, entries:[
      { binding:0, resource:{ buffer:this.frameUBO } },
      { binding:1, resource:this.gbufView },
      { binding:2, resource:this.linearSampler },
    ]});
    this.ssaoBlurBG = this.device.createBindGroup({ layout:this.ssaoBlurBGL, entries:[
      { binding:0, resource:this.ssaoView },
      { binding:1, resource:this.linearSampler },
    ]});

    // SSR bind group (hdr → hdr2)
    this.ssrBG = this.device.createBindGroup({ layout:this.ssrBGL, entries:[
      { binding:0, resource:{ buffer:this.frameUBO } },
      { binding:1, resource:this.hdrView },
      { binding:2, resource:this.gbufView },
      { binding:3, resource:this.linearSampler },
    ]});

    // God-ray bind group (hdr2 → hdr3)
    this.godrayBG = this.device.createBindGroup({ layout:this.godrayBGL, entries:[
      { binding:0, resource:{ buffer:this.frameUBO } },
      { binding:1, resource:this.hdrView2 },
      { binding:2, resource:this.gbufView },
      { binding:3, resource:this.linearSampler },
    ]});

    // Bloom extract reads hdr3 (post-godrays)
    this.bloomExtBG = this.device.createBindGroup({ layout:this.bloomExtBGL, entries:[
      { binding:0, resource:this.hdrView3 },
      { binding:1, resource:this.linearSampler },
    ]});

    const blrBGL = (this as any)._bloomBlrBGL as GPUBindGroupLayout;
    this.bloomHBG = this.device.createBindGroup({ layout:blrBGL, entries:[
      { binding:0, resource:this.bloomExtView },
      { binding:1, resource:this.linearSampler },
      { binding:2, resource:{ buffer:(this as any)._blurHDir } },
    ]});
    this.bloomVBG = this.device.createBindGroup({ layout:blrBGL, entries:[
      { binding:0, resource:this.bloomBlrView },
      { binding:1, resource:this.linearSampler },
      { binding:2, resource:{ buffer:(this as any)._blurVDir } },
    ]});

    // Tonemap reads hdr3 (post-godrays) + bloom result → writes ldrTex
    this.tonemapBG = this.device.createBindGroup({ layout:this.tonemapBGL, entries:[
      { binding:0, resource:this.hdrView3 },
      { binding:1, resource:this.bloomExtView },
      { binding:2, resource:this.linearSampler },
      { binding:3, resource:{ buffer:this.postUBO } },
    ]});

    // FXAA reads ldrTex → canvas
    this.fxaaBG = this.device.createBindGroup({ layout:this.fxaaBGL, entries:[
      { binding:0, resource:this.ldrView },
      { binding:1, resource:this.linearSampler },
      { binding:2, resource:{ buffer:this.frameUBO } },
    ]});
  }

  resize(w: number, h: number) {
    if (w<1||h<1) return;
    this.canvas.width=w; this.canvas.height=h;
    this._ensureHDR();
  }

  // ── GPU resource helpers ──────────────────────────────────────
  mkVBuf(data: Float32Array): GPUBuffer {
    const b = this.device.createBuffer({ size:data.byteLength, usage:GPUBufferUsage.VERTEX, mappedAtCreation:true });
    new Float32Array(b.getMappedRange()).set(data); b.unmap(); return b;
  }
  mkIBuf(data: Uint16Array): GPUBuffer {
    const b = this.device.createBuffer({ size:data.byteLength, usage:GPUBufferUsage.INDEX, mappedAtCreation:true });
    new Uint16Array(b.getMappedRange()).set(data); b.unmap(); return b;
  }
  mkIBuf32(data: Uint32Array): GPUBuffer {
    const b = this.device.createBuffer({ size:data.byteLength, usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST, mappedAtCreation:true });
    new Uint32Array(b.getMappedRange()).set(data); b.unmap(); return b;
  }
  mkUniformResources(): {
    uniformBuffer: GPUBuffer; bindGroup: GPUBindGroup;
    shadowUniformBuffer: GPUBuffer; shadowBindGroup: GPUBindGroup;
    prepassBindGroup: GPUBindGroup;
  } {
    const uniformBuffer = this.device.createBuffer({ size:OBJ_UBO, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const bindGroup = this.device.createBindGroup({ layout:this.objBGL,
      entries:[{ binding:0, resource:{ buffer:uniformBuffer } }] });
    // prepass reuses the same per-object UBO (model+albedo+pbr+flags), only needs binding 0
    const prepassBindGroup = this.device.createBindGroup({ layout:this.prepassObjBGL,
      entries:[{ binding:0, resource:{ buffer:uniformBuffer } }] });
    const shadowUniformBuffer = this.device.createBuffer({ size:SHDW_OBJ, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const shadowBindGroup = this.device.createBindGroup({ layout:this.shadowObjBGL,
      entries:[{ binding:0, resource:{ buffer:shadowUniformBuffer } }] });
    return { uniformBuffer, bindGroup, shadowUniformBuffer, shadowBindGroup, prepassBindGroup };
  }

  // ── Main render method ────────────────────────────────────────
  render(
    objects:     RenderObject[],
    viewProj:    Mat4,
    lightVP:     Mat4,
    view:        Mat4,
    camPos:      Vec3,
    camRight:    Vec3,
    camUp:       Vec3,
    camForward:  Vec3,
    sunDir:      Vec3,
    sunColor:    Vec3,
    pointLights: PointLight[],
    time:        number,
    showGrid:    boolean,
  ) {
    this._ensureHDR();
    const { width: W, height: H } = this.canvas;
    if (W<1||H<1) return;
    const dev = this.device;

    const invVP = viewProj.invert();

    // ── Update frame UBO ─────────────────────────────────────────
    const fu = new Float32Array(FRAME_UBO/4);
    fu.set(viewProj.data,    0);   // 0..15
    fu.set(lightVP.data,    16);   // 16..31
    fu.set(invVP.data,      32);   // 32..47
    fu.set(view.data,       48);   // 48..63
    fu.set([camPos.x,    camPos.y,    camPos.z,    1], 64);
    fu.set([camRight.x,  camRight.y,  camRight.z,  0], 68);
    fu.set([camUp.x,     camUp.y,     camUp.z,     0], 72);
    fu.set([camForward.x,camForward.y,camForward.z,0], 76);
    fu.set([sunDir.x,    sunDir.y,    sunDir.z,    0], 80);
    fu.set([sunColor.x,  sunColor.y,  sunColor.z,  1], 84);
    fu.set([0.10, 0.14, 0.20, this.wetness],           88); // ambient.xyz + wetness in .w
    fu.set([0.012, 0.06, 18.0, this.fogEnabled?1:0],   92); // fog: density, heightFalloff, start, enabled
    fu.set([time, Math.min(pointLights.length,4),
            this.ssrEnabled?1:0, this.ssaoEnabled?1:0], 96); // extras
    fu.set([this.quality, this.cloudsEnabled?1:0,
            this.godraysEnabled?1:0, this.waterEnabled?1:0], 100); // quality
    const nLights = Math.min(pointLights.length, 4);
    for (let i=0; i<nLights; i++) {
      const pl = pointLights[i];
      fu.set([pl.position.x, pl.position.y, pl.position.z, 1], 104+i*4);
      fu.set([pl.color.x*pl.intensity, pl.color.y*pl.intensity, pl.color.z*pl.intensity, pl.radius], 120+i*4);
    }
    dev.queue.writeBuffer(this.frameUBO, 0, fu);
    dev.queue.writeBuffer(this.shadowFrameUBO, 0, lightVP.data);

    const enc = dev.createCommandEncoder({ label:'frame' });

    // ── Pass 1: Shadow map ────────────────────────────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[],
        depthStencilAttachment:{
          view:this.shadowDepthView,
          depthClearValue:1, depthLoadOp:'clear', depthStoreOp:'store',
        }
      });
      pass.setPipeline(this.shadowPipeline);
      pass.setBindGroup(0, this.shadowFrameBG);
      for (const obj of objects) {
        dev.queue.writeBuffer(obj.shadowUniformBuffer, 0, obj.modelMatrix.data);
        pass.setBindGroup(1, obj.shadowBindGroup);
        pass.setVertexBuffer(0, obj.vertexBuffer);
        pass.setIndexBuffer(obj.indexBuffer, obj.indexFormat ?? 'uint16');
        pass.drawIndexed(obj.indexCount);
      }
      pass.end();
    }

    // ── Write per-object UBOs once (used by prepass + main) ───────
    const ou = new Float32Array(OBJ_UBO/4);
    for (const obj of objects) {
      ou.set(obj.modelMatrix.data, 0);
      ou.set([obj.color[0], obj.color[1], obj.color[2], obj.opacity], 16);
      ou.set([obj.metalness, obj.roughness, obj.emissive, obj.selected?1:0], 20);
      ou.set([obj.textured, obj.material ?? 0, 0, 0], 24);
      dev.queue.writeBuffer(obj.uniformBuffer, 0, ou);
    }

    // ── Pass 2: Normal/Depth prepass → G-buffer ───────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{
          view:this.gbufView,
          clearValue:{r:0,g:0,b:0,a:0}, loadOp:'clear', storeOp:'store',
        }],
        depthStencilAttachment:{
          view:this.prepassDepthView,
          depthClearValue:1, depthLoadOp:'clear', depthStoreOp:'store',
        }
      });
      pass.setPipeline(this.prepassPipeline);
      pass.setBindGroup(0, this.prepassFrameBG);
      for (const obj of objects) {
        if (!obj.prepassBindGroup) continue;
        pass.setBindGroup(1, obj.prepassBindGroup);
        pass.setVertexBuffer(0, obj.vertexBuffer);
        pass.setIndexBuffer(obj.indexBuffer, obj.indexFormat ?? 'uint16');
        pass.drawIndexed(obj.indexCount);
      }
      pass.end();
    }

    // ── Pass 3: SSAO → ssaoTex, then blur → ssaoBlurTex ───────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.ssaoView, loadOp:'clear', storeOp:'store', clearValue:{r:1,g:1,b:1,a:1} }]
      });
      pass.setPipeline(this.ssaoPipeline);
      pass.setBindGroup(0, this.ssaoBG);
      pass.draw(3);
      pass.end();
    }
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.ssaoBlurView, loadOp:'clear', storeOp:'store', clearValue:{r:1,g:1,b:1,a:1} }]
      });
      pass.setPipeline(this.ssaoBlurPipeline);
      pass.setBindGroup(0, this.ssaoBlurBG);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 4: Sky + Grid + Scene (PBR) → HDR ────────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{
          view:this.hdrView,
          clearValue:{r:0,g:0,b:0,a:1}, loadOp:'clear', storeOp:'store',
        }],
        depthStencilAttachment:{
          view:this.depthView,
          depthClearValue:1, depthLoadOp:'clear', depthStoreOp:'store',
        }
      });

      pass.setPipeline(this.skyPipeline);
      pass.setBindGroup(0, (this as any)._skyBG);
      pass.draw(3);

      if (showGrid) {
        pass.setPipeline(this.gridPipeline);
        pass.setBindGroup(0, (this as any)._gridBG);
        pass.setVertexBuffer(0, this.gridBuf);
        pass.draw(this.gridCount);
      }

      pass.setPipeline(this.mainPipeline);
      pass.setBindGroup(0, this.frameBG);
      for (const obj of objects) {
        pass.setBindGroup(1, obj.bindGroup);
        pass.setVertexBuffer(0, obj.vertexBuffer);
        pass.setIndexBuffer(obj.indexBuffer, obj.indexFormat ?? 'uint16');
        pass.drawIndexed(obj.indexCount);
      }
      pass.end();
    }

    // ── Pass 5: SSR (hdr → hdr2). If disabled, shader passes through. ─
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.hdrView2, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]
      });
      pass.setPipeline(this.ssrPipeline);
      pass.setBindGroup(0, this.ssrBG);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 5b: God rays (hdr2 → hdr3). Shader passes through if disabled. ─
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.hdrView3, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]
      });
      pass.setPipeline(this.godrayPipeline);
      pass.setBindGroup(0, this.godrayBG);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 6: Bloom extract (from hdr3) ─────────────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.bloomExtView, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]
      });
      pass.setPipeline(this.bloomExtPipeline);
      pass.setBindGroup(0, this.bloomExtBG);
      pass.draw(3);
      pass.end();
    }
    // Pass 7: blur H (ext → blr)
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.bloomBlrView, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]
      });
      pass.setPipeline(this.bloomHPipeline);
      pass.setBindGroup(0, this.bloomHBG);
      pass.draw(3);
      pass.end();
    }
    // Pass 8: blur V (blr → ext)
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{ view:this.bloomExtView, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]
      });
      pass.setPipeline(this.bloomVPipeline);
      pass.setBindGroup(0, this.bloomVBG);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 9: Tonemap → LDR texture ─────────────────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{
          view: this.ldrView,
          loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1}
        }]
      });
      pass.setPipeline(this.tonemapPipeline);
      pass.setBindGroup(0, this.tonemapBG);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 10: FXAA → canvas ────────────────────────────────────
    {
      const pass = enc.beginRenderPass({
        colorAttachments:[{
          view: this.ctx.getCurrentTexture().createView(),
          loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1}
        }]
      });
      pass.setPipeline(this.fxaaPipeline);
      pass.setBindGroup(0, this.fxaaBG);
      pass.draw(3);
      pass.end();
    }

    dev.queue.submit([enc.finish()]);
  }
}
