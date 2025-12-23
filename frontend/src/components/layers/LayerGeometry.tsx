import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Text, RoundedBox } from '@react-three/drei';
import type { LayerType, LayerParams, TensorShape } from '@/schema/types';
import { LAYER_CATEGORIES, DEFAULT_CATEGORY_COLORS } from '@/schema/types';

/**
 * Props for layer geometry components
 */
export interface LayerGeometryProps {
  type: LayerType;
  params?: LayerParams;
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  color: string;
  scale?: { x: number; y: number; z: number };
  selected?: boolean;
  hovered?: boolean;
  lod?: number;
  showLabel?: boolean;
  label?: string;
  onClick?: () => void;
  onPointerOver?: () => void;
  onPointerOut?: () => void;
}

/**
 * Get geometry dimensions based on layer type and params
 */
function getLayerDimensions(
  type: LayerType,
  _params?: LayerParams,
  outputShape?: TensorShape
): { width: number; height: number; depth: number } {
  // Base dimensions
  let width = 1;
  let height = 1;
  let depth = 0.2;
  
  // Scale based on output shape
  if (outputShape && outputShape.length > 0) {
    const dims = outputShape.filter((d): d is number => typeof d === 'number' && d > 0);
    if (dims.length >= 3) {
      // 3D+ tensor (batch, channels, height, width, ...)
      const [c, h, w] = dims.slice(-3);
      width = Math.min(Math.log2(w + 1) * 0.5, 3);
      height = Math.min(Math.log2(h + 1) * 0.5, 3);
      depth = Math.min(Math.log2(c + 1) * 0.2, 1);
    } else if (dims.length >= 2) {
      // 2D tensor (batch, features)
      const [h, w] = dims.slice(-2);
      width = Math.min(Math.log2(w + 1) * 0.4, 2);
      height = Math.min(Math.log2(h + 1) * 0.4, 2);
    } else if (dims.length === 1) {
      // 1D tensor
      width = Math.min(Math.log2(dims[0] + 1) * 0.3, 2);
    }
  }
  
  // Adjust based on layer type
  const category = LAYER_CATEGORIES[type];
  switch (category) {
    case 'convolution':
      depth = Math.max(depth, 0.4);
      break;
    case 'pooling':
      depth = 0.15;
      break;
    case 'activation':
      depth = 0.1;
      width *= 0.8;
      height *= 0.8;
      break;
    case 'normalization':
      depth = 0.15;
      break;
    case 'attention':
      depth = 0.5;
      break;
    case 'recurrent':
      depth = 0.6;
      break;
  }
  
  return { width, height, depth };
}

/**
 * Main layer mesh component
 */
export function LayerMesh({
  type,
  params,
  outputShape,
  color,
  scale = { x: 1, y: 1, z: 1 },
  selected = false,
  hovered = false,
  lod = 0,
  showLabel = true,
  label,
  onClick,
  onPointerOver,
  onPointerOut,
}: LayerGeometryProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const dims = useMemo(() => getLayerDimensions(type, params, outputShape), [type, params, outputShape]);
  
  // Animate on hover/select
  useFrame(() => {
    if (meshRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      meshRef.current.scale.lerp(
        new THREE.Vector3(
          dims.width * scale.x * targetScale,
          dims.height * scale.y * targetScale,
          dims.depth * scale.z * targetScale
        ),
        0.1
      );
    }
  });
  
  // Color with hover/select modulation
  const displayColor = useMemo(() => {
    const baseColor = new THREE.Color(color);
    if (selected) {
      baseColor.multiplyScalar(1.3);
    } else if (hovered) {
      baseColor.multiplyScalar(1.15);
    }
    return baseColor;
  }, [color, selected, hovered]);
  
  // Choose geometry based on LOD
  const geometry = useMemo(() => {
    if (lod >= 2) {
      // Very low detail - simple box
      return <boxGeometry args={[1, 1, 1]} />;
    } else if (lod >= 1) {
      // Medium detail
      return <boxGeometry args={[1, 1, 1]} />;
    } else {
      // High detail - rounded box
      return null; // Use RoundedBox component
    }
  }, [lod]);
  
  return (
    <group>
      {lod < 1 ? (
        <RoundedBox
          ref={meshRef}
          args={[dims.width * scale.x, dims.height * scale.y, dims.depth * scale.z]}
          radius={0.05}
          smoothness={4}
          onClick={onClick}
          onPointerOver={onPointerOver}
          onPointerOut={onPointerOut}
        >
          <meshStandardMaterial
            color={displayColor}
            emissive={selected ? displayColor : undefined}
            emissiveIntensity={selected ? 0.2 : 0}
            metalness={0.1}
            roughness={0.6}
          />
        </RoundedBox>
      ) : (
        <mesh
          ref={meshRef}
          scale={[dims.width * scale.x, dims.height * scale.y, dims.depth * scale.z]}
          onClick={onClick}
          onPointerOver={onPointerOver}
          onPointerOut={onPointerOut}
        >
          {geometry}
          <meshStandardMaterial
            color={displayColor}
            emissive={selected ? displayColor : undefined}
            emissiveIntensity={selected ? 0.2 : 0}
          />
        </mesh>
      )}
      
      {/* Selection outline */}
      {selected && (
        <mesh scale={[dims.width * scale.x * 1.05, dims.height * scale.y * 1.05, dims.depth * scale.z * 1.05]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color="#ffffff" wireframe />
        </mesh>
      )}
      
      {/* Label */}
      {showLabel && label && lod < 2 && (
        <Text
          position={[0, dims.height * scale.y * 0.5 + 0.3, 0]}
          fontSize={0.2}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {label}
        </Text>
      )}
    </group>
  );
}

/**
 * Specialized geometry for convolutional layers
 */
export function ConvLayerMesh(props: LayerGeometryProps) {
  const { params } = props;
  const kernelSize = params?.kernelSize;
  
  // Show kernel visualization
  const showKernel = kernelSize && (typeof kernelSize === 'number' ? kernelSize > 1 : kernelSize[0] > 1);
  
  return (
    <group>
      <LayerMesh {...props} />
      {showKernel && (
        <mesh position={[0, 0, 0.5]}>
          <boxGeometry args={[0.3, 0.3, 0.1]} />
          <meshStandardMaterial color="#ffffff" opacity={0.5} transparent />
        </mesh>
      )}
    </group>
  );
}

/**
 * Specialized geometry for attention layers
 */
export function AttentionLayerMesh(props: LayerGeometryProps) {
  const { params, color, scale = { x: 1, y: 1, z: 1 } } = props;
  const numHeads = params?.numHeads || 8;
  
  // Create head indicators
  const headMarkers = useMemo(() => {
    const markers = [];
    const angleStep = (Math.PI * 2) / numHeads;
    const radius = 0.4;
    
    for (let i = 0; i < Math.min(numHeads, 12); i++) {
      const angle = i * angleStep;
      markers.push(
        <mesh
          key={i}
          position={[
            Math.cos(angle) * radius * scale.x,
            Math.sin(angle) * radius * scale.y,
            0.15 * scale.z,
          ]}
        >
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial color={color} />
        </mesh>
      );
    }
    return markers;
  }, [numHeads, color, scale]);
  
  return (
    <group>
      <LayerMesh {...props} />
      {headMarkers}
    </group>
  );
}

/**
 * Specialized geometry for pooling layers
 */
export function PoolingLayerMesh(props: LayerGeometryProps) {
  return (
    <group>
      <LayerMesh {...props} />
      {/* Grid pattern to indicate pooling */}
      <gridHelper
        args={[0.8, 4, '#ffffff', '#ffffff']}
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, 0.15]}
      />
    </group>
  );
}

/**
 * Factory function to get appropriate layer component
 */
export function getLayerComponent(type: LayerType): React.ComponentType<LayerGeometryProps> {
  const category = LAYER_CATEGORIES[type];
  
  switch (category) {
    case 'convolution':
      return ConvLayerMesh;
    case 'attention':
      return AttentionLayerMesh;
    case 'pooling':
      return PoolingLayerMesh;
    default:
      return LayerMesh;
  }
}

/**
 * Default color for a layer type
 */
export function getDefaultLayerColor(type: LayerType): string {
  const category = LAYER_CATEGORIES[type];
  return DEFAULT_CATEGORY_COLORS[category];
}
