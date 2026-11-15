export class Vec3 {
  constructor(public x=0, public y=0, public z=0){}
  clone(){ return new Vec3(this.x,this.y,this.z) }
  set(x:number,y:number,z:number){ this.x=x;this.y=y;this.z=z;return this }
  add(v:Vec3){ return new Vec3(this.x+v.x,this.y+v.y,this.z+v.z) }
  sub(v:Vec3){ return new Vec3(this.x-v.x,this.y-v.y,this.z-v.z) }
  mul(s:number){ return new Vec3(this.x*s,this.y*s,this.z*s) }
  div(s:number){ return this.mul(1/s) }
  dot(v:Vec3){ return this.x*v.x+this.y*v.y+this.z*v.z }
  cross(v:Vec3){ return new Vec3(this.y*v.z-this.z*v.y,this.z*v.x-this.x*v.z,this.x*v.y-this.y*v.x) }
  lengthSq(){ return this.dot(this) }
  length(){ return Math.sqrt(this.lengthSq()) }
  normalize(){ const l=this.length(); return l>1e-5?this.div(l):new Vec3(0,1,0) }
  lerp(v:Vec3,t:number){ return new Vec3(this.x+(v.x-this.x)*t,this.y+(v.y-this.y)*t,this.z+(v.z-this.z)*t) }
  toArray():[number,number,number]{ return [this.x,this.y,this.z] }
  static fromArray(a:number[]){ return new Vec3(a[0],a[1],a[2]) }
  static zero(){ return new Vec3(0,0,0) }
  static up(){ return new Vec3(0,1,0) }
  static forward(){ return new Vec3(0,0,-1) }
  static right(){ return new Vec3(1,0,0) }
}
