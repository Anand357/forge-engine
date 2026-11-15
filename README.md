# Forge Engine

**A 3D game engine and world editor built from scratch in TypeScript on WebGPU, running entirely in the browser.**

![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![WebGPU](https://img.shields.io/badge/WebGPU-005A9C)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Screenshots

> _Screenshots coming soon._
>
> `![Forge Engine editor](docs/screenshot-editor.png)`

## Features

### Rendering (custom WebGPU renderer, hand-written WGSL)
- **PBR materials** with per-object albedo, roughness and metalness
- **Shadow mapping** (2048×2048 directional shadow map)
- **SSAO** from a normal/depth G-buffer prepass, with blur
- **HDR pipeline** with bloom, screen-space reflections, god rays, tone mapping and FXAA
- **Procedural sky** with raymarched volumetric clouds, sun disk and horizon haze
- **Animated water** surface built from layered moving noise
- **Triplanar procedural texturing** and distance fog with no external texture files
- **Time of day and weather** controls with an auto-moving sun

### World and terrain
- Setup screen to start a **finite, infinite or custom** world with Rolling Hills, Mountains, Flat Plain or Empty presets
- Gravity presets for Earth, Moon, Mars and Zero-G
- **Editable heightfield terrain**: sculpt (raise, lower, smooth) and stamp mountains, hills, canyons and plateaus
- Procedural value-noise / fBm terrain generation

### Editor
- **Hierarchy, Layers and Inspector** panels for organising and editing scene objects
- Translate **gizmos** with ray-axis picking
- **Undo**, duplicate and delete
- **Asset browser** with a procedural library of 100+ ready-to-place models built from primitives
- **glTF 2.0 / .glb import** through a small self-written loader
- **Blueprint editor**, a node-based visual scripting overlay where you wire output ports to input ports
- **Save/load worlds** as JSON and share them through links
- Guided first-run tutorial

### Play mode
- **Rapier 3D physics** (rigid bodies and colliders) through `@dimforge/rapier3d-compat`
- **Third-person player controller**: WASD, sprint, jump, a spring-arm camera and terrain ground-follow
- **Vehicle controller** with speed-dependent steering, lateral grip and drift, plus a chase camera
- AABB collision with wall-sliding so players and vehicles can't pass through props
- Gameplay layer with collectibles, hazards, checkpoints, goals, score and health, plus ready-made game templates

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | TypeScript |
| Graphics API | WebGPU + WGSL shaders |
| Physics | Rapier 3D (`@dimforge/rapier3d-compat`, WASM) |
| Build tool | Vite |
| Math | Custom `Vec3` / column-major `Mat4` library |

No rendering framework is used (no Three.js or Babylon.js). The renderer, math, loaders and editor UI are all written from scratch.

## Getting Started

**Requirements:** Node.js 18+ and a browser with WebGPU support (recent Chrome or Edge, or Safari 18+).

```bash
git clone https://github.com/Anand357/forge-engine.git
cd forge-engine
npm install
npm run dev
```

Open the local URL Vite prints, usually http://localhost:5173.

This project needs no environment variables or API keys.

Build for production:

```bash
npm run build
npm run preview
```

## Project Structure

```
forge-engine/
├── index.html                  # Editor UI layout and styles
├── public/
└── src/
    ├── main.ts                 # Editor wiring: setup screen, panels, tools, play mode
    └── engine/
        ├── ForgeEngine.ts      # Engine core: scene, renderer, physics, game loop
        ├── blueprint/
        │   └── Blueprint.ts    # Node-based visual scripting editor
        ├── math/
        │   ├── Vec3.ts
        │   └── Mat4.ts
        ├── physics/
        │   └── Physics.ts      # Rapier world and rigid-body sync
        ├── renderer/
        │   ├── Renderer.ts     # WebGPU pipelines + WGSL (PBR, shadows, SSAO, post FX)
        │   ├── Geometry.ts     # Primitive mesh generators
        │   ├── AssetLibrary.ts # 100+ procedural models
        │   ├── GLTFLoader.ts   # glTF 2.0 / .glb loader
        │   └── GizmoRenderer.ts
        └── scene/
            ├── Scene.ts        # Scene graph, nodes and layers
            ├── Camera.ts
            ├── Terrain.ts      # Sculptable heightfield terrain
            ├── Collision.ts    # AABB collision
            ├── PlayerController.ts
            ├── VehicleController.ts
            ├── GameState.ts    # Score, health, objectives
            └── Light.ts
```

## What I Learned

- How a modern GPU pipeline works end to end: buffers, bind groups, uniform layouts, render passes and writing shaders in WGSL
- Building a multi-pass renderer (G-buffer prepass, shadow pass, SSAO, HDR, bloom, SSR, god rays, tone mapping and FXAA) and managing the textures between passes
- The math behind 3D graphics: matrices, camera projection, ray picking and procedural noise
- Integrating a WASM physics engine (Rapier) and keeping physics bodies in sync with the scene graph
- Structuring a larger TypeScript codebase that cleanly separates engine, editor and gameplay code

## License

This project is licensed under the [MIT License](LICENSE).
