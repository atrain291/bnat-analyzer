import { useRef, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { SMPL_BONE_CONNECTIONS, FOOT_JOINT_INDICES } from "../constants/smpl";
import type { FrameData } from "../api/performances";

interface DancerSkeleton {
  joints3d: number[][];
  footContact: FrameData["foot_contact"];
  color: string;
}

interface Skeleton3DViewerProps {
  dancers: DancerSkeleton[];
}

const JOINT_RADIUS = 0.02;
const BONE_RADIUS = 0.008;

function Joint({ position, color, scale = 1 }: { position: [number, number, number]; color: string; scale?: number }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[JOINT_RADIUS * scale, 8, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

function Bone({ start, end, color }: { start: [number, number, number]; end: [number, number, number]; color: string }) {
  const ref = useRef<THREE.Mesh>(null);

  const { position, scale, quaternion } = useMemo(() => {
    const s = new THREE.Vector3(...start);
    const e = new THREE.Vector3(...end);
    const mid = new THREE.Vector3().lerpVectors(s, e, 0.5);
    const dist = s.distanceTo(e);
    const dir = new THREE.Vector3().subVectors(e, s).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    return { position: mid, scale: dist, quaternion: q };
  }, [start, end]);

  return (
    <mesh ref={ref} position={position} quaternion={quaternion}>
      <cylinderGeometry args={[BONE_RADIUS, BONE_RADIUS, scale, 6]} />
      <meshStandardMaterial color={color} opacity={0.8} transparent />
    </mesh>
  );
}

function DancerSkeleton3D({ dancer }: { dancer: DancerSkeleton }) {
  const joints = dancer.joints3d;
  const color = dancer.color;

  const footContactMap = useMemo(() => {
    if (!dancer.footContact) return new Set<number>();
    const grounded = new Set<number>();
    const contacts: [number, number][] = [
      [7, dancer.footContact.left_heel],
      [8, dancer.footContact.right_heel],
      [10, dancer.footContact.left_toe],
      [11, dancer.footContact.right_toe],
    ];
    for (const [idx, val] of contacts) {
      if (val > 0.5) grounded.add(idx);
    }
    return grounded;
  }, [dancer.footContact]);

  return (
    <group>
      {joints.map((j, i) => {
        if (!j || j.length < 3) return null;
        const isFoot = FOOT_JOINT_INDICES.includes(i as typeof FOOT_JOINT_INDICES[number]);
        const isGrounded = footContactMap.has(i);
        const jointColor = isFoot ? (isGrounded ? "#4ade80" : "#f87171") : color;
        const jointScale = isFoot ? 1.4 : 1;
        return (
          <Joint key={i} position={[j[0], j[1], j[2]]} color={jointColor} scale={jointScale} />
        );
      })}
      {SMPL_BONE_CONNECTIONS.map(([a, b], i) => {
        const ja = joints[a];
        const jb = joints[b];
        if (!ja || !jb || ja.length < 3 || jb.length < 3) return null;
        return (
          <Bone key={i} start={[ja[0], ja[1], ja[2]]} end={[jb[0], jb[1], jb[2]]} color={color} />
        );
      })}
    </group>
  );
}

function GroundPlane() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.1, 0]}>
      <planeGeometry args={[4, 4]} />
      <meshStandardMaterial color="#1f2937" transparent opacity={0.5} side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function Skeleton3DViewer({ dancers }: Skeleton3DViewerProps) {
  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <Canvas camera={{ fov: 50, near: 0.01, far: 50, position: [1.5, 0.5, 2.0] }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 5, 4]} intensity={0.8} />
        {dancers.map((dancer, i) => (
          <DancerSkeleton3D key={i} dancer={dancer} />
        ))}
        <GroundPlane />
        <OrbitControls target={[0, -0.3, 0]} enableDamping dampingFactor={0.1} />
      </Canvas>
    </div>
  );
}

export type { DancerSkeleton };
