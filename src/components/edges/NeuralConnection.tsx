/**
 * Neural Network Connection Visualization
 * Shows dense connections between layers like actual neural networks
 */

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import type { Position3D } from '@/schema/types';

export interface NeuralConnectionProps {
  sourcePosition: Position3D;
  targetPosition: Position3D;
  sourceNeurons?: number;
  targetNeurons?: number;
  color?: string;
  highlighted?: boolean;
  animated?: boolean;
  connectionDensity?: number; // 0-1, how many connections to show
  style?: 'single' | 'dense' | 'bundle';
}

/**
 * Single connection line between layers
 */
export function SingleConnection({
  sourcePosition,
  targetPosition,
  color = '#ffffff',
  highlighted = false,
}: NeuralConnectionProps) {
  const geometry = useMemo(() => {
    const points = [
      new THREE.Vector3(sourcePosition.x, sourcePosition.y, sourcePosition.z),
      new THREE.Vector3(targetPosition.x, targetPosition.y, targetPosition.z),
    ];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [sourcePosition, targetPosition]);
  
  return (
    <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color: highlighted ? '#ffffff' : color,
      transparent: true,
      opacity: highlighted ? 1 : 0.5,
    }))} />
  );
}

/**
 * Dense connection bundle showing multiple lines
 */
export function DenseConnection({
  sourcePosition,
  targetPosition,
  sourceNeurons = 16,
  targetNeurons = 16,
  color = '#4488ff',
  highlighted = false,
  animated = true,
  connectionDensity = 0.3,
}: NeuralConnectionProps) {
  const groupRef = useRef<THREE.Group>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  
  // Limit connections for performance
  const maxConnections = 100;
  const numConnections = Math.min(
    Math.floor(sourceNeurons * targetNeurons * connectionDensity),
    maxConnections
  );
  
  // Generate connection lines
  const lines = useMemo(() => {
    const connections: { start: THREE.Vector3; end: THREE.Vector3 }[] = [];
    
    // Calculate source and target layer bounds
    const sourceSpread = Math.min(Math.sqrt(sourceNeurons) * 0.1, 0.4);
    const targetSpread = Math.min(Math.sqrt(targetNeurons) * 0.1, 0.4);
    
    for (let i = 0; i < numConnections; i++) {
      // Random positions within layer bounds
      const srcOffset = {
        y: (Math.random() - 0.5) * sourceSpread,
        z: (Math.random() - 0.5) * sourceSpread * 0.5,
      };
      const tgtOffset = {
        y: (Math.random() - 0.5) * targetSpread,
        z: (Math.random() - 0.5) * targetSpread * 0.5,
      };
      
      connections.push({
        start: new THREE.Vector3(
          sourcePosition.x + 0.3, // Offset from layer center
          sourcePosition.y + srcOffset.y,
          sourcePosition.z + srcOffset.z
        ),
        end: new THREE.Vector3(
          targetPosition.x - 0.3, // Offset from layer center
          targetPosition.y + tgtOffset.y,
          targetPosition.z + tgtOffset.z
        ),
      });
    }
    
    return connections;
  }, [sourcePosition, targetPosition, sourceNeurons, targetNeurons, numConnections]);
  
  // Animate opacity
  useFrame((state) => {
    if (animated && materialRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 0.3;
      materialRef.current.opacity = highlighted ? 0.8 : pulse;
    }
  });
  
  // Create buffer geometry for all lines
  const geometry = useMemo(() => {
    const positions: number[] = [];
    
    lines.forEach(line => {
      positions.push(line.start.x, line.start.y, line.start.z);
      positions.push(line.end.x, line.end.y, line.end.z);
    });
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [lines]);
  
  return (
    <group ref={groupRef}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial
          ref={materialRef}
          color={highlighted ? '#ffffff' : color}
          transparent
          opacity={0.3}
          depthWrite={false}
        />
      </lineSegments>
    </group>
  );
}

/**
 * Bundled connection - shows as a tube/pipe
 */
export function BundledConnection({
  sourcePosition,
  targetPosition,
  sourceNeurons = 16,
  targetNeurons = 16,
  color = '#4488ff',
  highlighted = false,
  animated = true,
}: NeuralConnectionProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Calculate bundle thickness based on connection count
  const connectionStrength = Math.log2(Math.min(sourceNeurons, targetNeurons) + 1) * 0.02;
  const thickness = Math.max(0.02, Math.min(connectionStrength, 0.1));
  
  // Create tube path
  const curve = useMemo(() => {
    const start = new THREE.Vector3(sourcePosition.x, sourcePosition.y, sourcePosition.z);
    const end = new THREE.Vector3(targetPosition.x, targetPosition.y, targetPosition.z);
    
    // Bezier control points for smooth curve
    const midX = (start.x + end.x) / 2;
    const control1 = new THREE.Vector3(midX, start.y, start.z);
    const control2 = new THREE.Vector3(midX, end.y, end.z);
    
    return new THREE.CubicBezierCurve3(start, control1, control2, end);
  }, [sourcePosition, targetPosition]);
  
  const geometry = useMemo(() => {
    return new THREE.TubeGeometry(curve, 20, thickness, 8, false);
  }, [curve, thickness]);
  
  // Animate flow effect
  useFrame((state) => {
    if (animated && meshRef.current) {
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      if (material.emissiveIntensity !== undefined) {
        material.emissiveIntensity = Math.sin(state.clock.elapsedTime * 3) * 0.2 + 0.3;
      }
    }
  });
  
  const baseColor = new THREE.Color(color);
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color={highlighted ? '#ffffff' : baseColor}
        emissive={baseColor}
        emissiveIntensity={0.3}
        transparent
        opacity={highlighted ? 0.9 : 0.6}
        metalness={0.3}
        roughness={0.7}
      />
    </mesh>
  );
}

/**
 * Flow particles along connection
 */
export function FlowParticles({
  sourcePosition,
  targetPosition,
  color = '#ffffff',
  particleCount = 5,
  speed = 1,
}: {
  sourcePosition: Position3D;
  targetPosition: Position3D;
  color?: string;
  particleCount?: number;
  speed?: number;
}) {
  const particlesRef = useRef<THREE.Points>(null);
  
  // Create particle positions
  const { positions, offsets } = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const off = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
      off[i] = i / particleCount; // Spread particles along path
      
      // Initial positions will be updated in useFrame
      pos[i * 3] = sourcePosition.x;
      pos[i * 3 + 1] = sourcePosition.y;
      pos[i * 3 + 2] = sourcePosition.z;
    }
    
    return { positions: pos, offsets: off };
  }, [sourcePosition, particleCount]);
  
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);
  
  // Animate particles
  useFrame((state) => {
    if (particlesRef.current) {
      const posAttr = particlesRef.current.geometry.getAttribute('position') as THREE.BufferAttribute;
      
      for (let i = 0; i < particleCount; i++) {
        // Calculate t along path (0 to 1)
        const t = ((state.clock.elapsedTime * speed + offsets[i]) % 1);
        
        // Lerp position
        posAttr.setXYZ(
          i,
          sourcePosition.x + (targetPosition.x - sourcePosition.x) * t,
          sourcePosition.y + (targetPosition.y - sourcePosition.y) * t,
          sourcePosition.z + (targetPosition.z - sourcePosition.z) * t
        );
      }
      
      posAttr.needsUpdate = true;
    }
  });
  
  return (
    <points ref={particlesRef} geometry={geometry}>
      <pointsMaterial
        color={color}
        size={0.05}
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
}

/**
 * Smart connection that chooses visualization based on layer sizes
 */
export function SmartConnection({
  sourcePosition,
  targetPosition,
  sourceNeurons = 16,
  targetNeurons = 16,
  color = '#4488ff',
  highlighted = false,
  animated = true,
  style = 'bundle',
}: NeuralConnectionProps) {
  const totalConnections = sourceNeurons * targetNeurons;
  
  // Choose visualization based on connection count
  if (style === 'single' || totalConnections < 50) {
    return (
      <SingleConnection
        sourcePosition={sourcePosition}
        targetPosition={targetPosition}
        color={color}
        highlighted={highlighted}
      />
    );
  }
  
  if (style === 'dense' || totalConnections < 500) {
    return (
      <>
        <DenseConnection
          sourcePosition={sourcePosition}
          targetPosition={targetPosition}
          sourceNeurons={sourceNeurons}
          targetNeurons={targetNeurons}
          color={color}
          highlighted={highlighted}
          animated={animated}
          connectionDensity={0.2}
        />
        {animated && (
          <FlowParticles
            sourcePosition={sourcePosition}
            targetPosition={targetPosition}
            color={color}
            particleCount={3}
            speed={0.5}
          />
        )}
      </>
    );
  }
  
  // For very dense connections, use bundled representation
  return (
    <>
      <BundledConnection
        sourcePosition={sourcePosition}
        targetPosition={targetPosition}
        sourceNeurons={sourceNeurons}
        targetNeurons={targetNeurons}
        color={color}
        highlighted={highlighted}
        animated={animated}
      />
      {animated && (
        <FlowParticles
          sourcePosition={sourcePosition}
          targetPosition={targetPosition}
          color="#ffffff"
          particleCount={5}
          speed={0.8}
        />
      )}
    </>
  );
}
