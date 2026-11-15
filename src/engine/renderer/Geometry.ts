export type GeoData={vertices:Float32Array;indices:Uint16Array};

export function createCube(s=1):GeoData{
  const h=s/2;
  const faces=[
    [-h,-h, h,0,0,1],[ h,-h, h,0,0,1],[ h, h, h,0,0,1],[-h, h, h,0,0,1],
    [ h,-h,-h,0,0,-1],[-h,-h,-h,0,0,-1],[-h, h,-h,0,0,-1],[ h, h,-h,0,0,-1],
    [-h, h, h,0,1,0],[ h, h, h,0,1,0],[ h, h,-h,0,1,0],[-h, h,-h,0,1,0],
    [-h,-h,-h,0,-1,0],[ h,-h,-h,0,-1,0],[ h,-h, h,0,-1,0],[-h,-h, h,0,-1,0],
    [ h,-h, h,1,0,0],[ h,-h,-h,1,0,0],[ h, h,-h,1,0,0],[ h, h, h,1,0,0],
    [-h,-h,-h,-1,0,0],[-h,-h, h,-1,0,0],[-h, h, h,-1,0,0],[-h, h,-h,-1,0,0],
  ];
  const idx:number[]=[]; for(let i=0;i<6;i++){const b=i*4;idx.push(b,b+1,b+2,b,b+2,b+3);}
  return{vertices:new Float32Array(faces.flat()),indices:new Uint16Array(idx)};
}

export function createSphere(r=1,seg=32):GeoData{
  const v:number[]=[],idx:number[]=[];
  for(let lat=0;lat<=seg;lat++){
    const t=(lat/seg)*Math.PI;
    for(let lon=0;lon<=seg;lon++){
      const p=(lon/seg)*2*Math.PI,x=Math.sin(t)*Math.cos(p),y=Math.cos(t),z=Math.sin(t)*Math.sin(p);
      v.push(x*r,y*r,z*r,x,y,z);
    }
  }
  for(let lat=0;lat<seg;lat++) for(let lon=0;lon<seg;lon++){
    const a=lat*(seg+1)+lon,b=a+seg+1; idx.push(a,b,a+1,b,b+1,a+1);
  }
  return{vertices:new Float32Array(v),indices:new Uint16Array(idx)};
}

export function createPlane(w=20,d=20,sw=32,sd=32):GeoData{
  const v:number[]=[],idx:number[]=[];
  for(let z=0;z<=sd;z++) for(let x=0;x<=sw;x++) v.push((x/sw-.5)*w,0,(z/sd-.5)*d,0,1,0);
  for(let z=0;z<sd;z++) for(let x=0;x<sw;x++){const a=z*(sw+1)+x,b=a+sw+1;idx.push(a,b,a+1,b,b+1,a+1);}
  return{vertices:new Float32Array(v),indices:new Uint16Array(idx)};
}

export function createCylinder(r=0.5,h=2,seg=24):GeoData{
  const v:number[]=[],idx:number[]=[];
  for(let i=0;i<=seg;i++){
    const a=(i/seg)*Math.PI*2,x=Math.cos(a),z=Math.sin(a);
    v.push(x*r,-h/2,z*r,x,0,z);v.push(x*r,h/2,z*r,x,0,z);
  }
  for(let i=0;i<seg;i++){const b=i*2;idx.push(b,b+2,b+1,b+1,b+2,b+3);}
  const base=(seg+1)*2;
  v.push(0,h/2,0,0,1,0);v.push(0,-h/2,0,0,-1,0);
  for(let i=0;i<seg;i++){
    const a0=(i/seg)*Math.PI*2,a1=((i+1)/seg)*Math.PI*2;
    v.push(Math.cos(a0)*r,h/2,Math.sin(a0)*r,0,1,0);
    v.push(Math.cos(a1)*r,h/2,Math.sin(a1)*r,0,1,0);
    v.push(Math.cos(a0)*r,-h/2,Math.sin(a0)*r,0,-1,0);
    v.push(Math.cos(a1)*r,-h/2,Math.sin(a1)*r,0,-1,0);
    const cb=base+2+i*4;
    idx.push(base,cb,cb+1);idx.push(base+1,cb+3,cb+2);
  }
  return{vertices:new Float32Array(v),indices:new Uint16Array(idx)};
}

export function createCone(r=0.7,h=2,seg=24):GeoData{
  const v:number[]=[],idx:number[]=[];
  const sl=Math.sqrt(h*h+r*r);
  for(let i=0;i<=seg;i++){
    const a=(i/seg)*Math.PI*2,x=Math.cos(a),z=Math.sin(a);
    v.push(x*r,-h/2,z*r,x*h/sl,r/sl,z*h/sl);
    v.push(0,h/2,0,x*h/sl,r/sl,z*h/sl);
  }
  for(let i=0;i<seg;i++){const b=i*2;idx.push(b,b+2,b+1);}
  const base=(seg+1)*2;v.push(0,-h/2,0,0,-1,0);
  for(let i=0;i<seg;i++){
    const a0=(i/seg)*Math.PI*2,a1=((i+1)/seg)*Math.PI*2;
    v.push(Math.cos(a0)*r,-h/2,Math.sin(a0)*r,0,-1,0);
    v.push(Math.cos(a1)*r,-h/2,Math.sin(a1)*r,0,-1,0);
    const cb=base+1+i*2;idx.push(base,cb+1,cb);
  }
  return{vertices:new Float32Array(v),indices:new Uint16Array(idx)};
}

export function createTerrain(w=80,d=80,segs=100):GeoData{
  const v:number[]=[],idx:number[]=[],heights:number[]=[];
  function noise(x:number,z:number){
    return Math.sin(x*0.15)*Math.cos(z*0.13)*5
          +Math.sin(x*0.4+1.3)*Math.cos(z*0.35+0.7)*2
          +Math.sin(x*0.9+2.1)*Math.cos(z*0.8+1.4)*0.8
          +Math.sin(x*2.0+0.5)*Math.cos(z*1.8+0.3)*0.3;
  }
  for(let z=0;z<=segs;z++) for(let x=0;x<=segs;x++)
    heights.push(noise((x/segs-.5)*w,(z/segs-.5)*d));
  for(let z=0;z<=segs;z++) for(let x=0;x<=segs;x++){
    const px=(x/segs-.5)*w,pz=(z/segs-.5)*d,py=heights[z*(segs+1)+x];
    const hL=x>0?heights[z*(segs+1)+x-1]:py;
    const hR=x<segs?heights[z*(segs+1)+x+1]:py;
    const hD=z>0?heights[(z-1)*(segs+1)+x]:py;
    const hU=z<segs?heights[(z+1)*(segs+1)+x]:py;
    const step=w/segs,nx2=(hL-hR)/(2*step),ny2=1,nz2=(hD-hU)/(2*step);
    const len=Math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2);
    v.push(px,py,pz,nx2/len,ny2/len,nz2/len);
  }
  for(let z=0;z<segs;z++) for(let x=0;x<segs;x++){
    const a=z*(segs+1)+x,b=a+segs+1;idx.push(a,b,a+1,b,b+1,a+1);
  }
  return{vertices:new Float32Array(v),indices:new Uint16Array(idx)};
}
