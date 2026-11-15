import { Vec3 } from '../math/Vec3';

export interface PointLight {
  id:        string;
  name:      string;
  position:  Vec3;
  color:     Vec3;
  intensity: number;
  radius:    number;
}
