/**
 * Neural Network Layer Visualization Components
 * Enhanced 3D representation for different layer types
 */

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Text, Sphere, Cylinder, Box, Torus } from '@react-three/drei';
import type { NN3DNode } from '@/schema/types';

export interface NeuralLayerProps {
  node: NN3DNode & { computedPosition: { x: number; y: number; z: number } };
  color: string;
  selected?: boolean;
  hovered?: boolean;
  showNeurons?: boolean;
  maxNeurons?: number;
  onClick?: () => void;
  onPointerOver?: () => void;
  onPointerOut?: () => void;
}

/**
 * Determine neuron count from layer attributes
 */
function getNeuronCount(node: NN3DNode): number {
  const attrs = node.attributes || {};
  
  // Check various attribute names
  const count = attrs.out_features || 
                attrs.outFeatures ||
                attrs.out_channels ||
                attrs.outChannels ||
                attrs.hidden_size ||
                attrs.hiddenSize ||
                attrs.units ||
                attrs.num_features ||
                attrs.numFeatures ||
                attrs.embed_dim ||
                attrs.embedDim ||
                16; // Default
  
  return typeof count === 'number' ? count : 16;
}

/**
 * Get input size from layer attributes
 */
function _getInputSize(node: NN3DNode): number {
  const attrs = node.attributes || {};
  
  const count = attrs.in_features ||
                attrs.inFeatures ||
                attrs.in_channels ||
                attrs.inChannels ||
                attrs.input_size ||
                attrs.inputSize ||
                16;
  
  return typeof count === 'number' ? count : 16;
}

// Export to prevent unused warning
void _getInputSize;

/**
 * Dense/Linear Layer - shows as a grid of neurons
 */
export function DenseLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  showNeurons = true,
  maxNeurons = 64,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const neuronCount = Math.min(getNeuronCount(node), maxNeurons);
  
  // Calculate grid dimensions
  const cols = Math.ceil(Math.sqrt(neuronCount));
  const rows = Math.ceil(neuronCount / cols);
  const spacing = 0.15;
  const neuronRadius = 0.06;
  
  // Animate on hover
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const neurons = useMemo(() => {
    const positions: [number, number, number][] = [];
    for (let i = 0; i < neuronCount; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const x = (col - (cols - 1) / 2) * spacing;
      const y = (row - (rows - 1) / 2) * spacing;
      positions.push([x, y, 0]);
    }
    return positions;
  }, [neuronCount, cols, rows]);
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Background plane */}
      <mesh position={[0, 0, -0.05]}>
        <planeGeometry args={[cols * spacing + 0.2, rows * spacing + 0.2]} />
        <meshStandardMaterial 
          color={displayColor} 
          transparent 
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Neurons */}
      {showNeurons && neurons.map((pos, i) => (
        <Sphere key={i} args={[neuronRadius, 8, 8]} position={pos}>
          <meshStandardMaterial
            color={displayColor}
            emissive={selected ? displayColor : undefined}
            emissiveIntensity={selected ? 0.3 : 0}
            metalness={0.3}
            roughness={0.5}
          />
        </Sphere>
      ))}
      
      {/* Selection indicator */}
      {selected && (
        <mesh position={[0, 0, -0.06]}>
          <planeGeometry args={[cols * spacing + 0.3, rows * spacing + 0.3]} />
          <meshBasicMaterial color="#ffffff" wireframe />
        </mesh>
      )}
      
      {/* Label */}
      <Text
        position={[0, rows * spacing / 2 + 0.25, 0]}
        fontSize={0.15}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
      
      {/* Neuron count */}
      <Text
        position={[0, -rows * spacing / 2 - 0.1, 0]}
        fontSize={0.08}
        color="#aaaaaa"
        anchorX="center"
        anchorY="top"
      >
        {getNeuronCount(node)} neurons
      </Text>
    </group>
  );
}

/**
 * Convolutional Layer - shows as a 3D block with feature maps
 */
export function ConvLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const attrs = node.attributes || {};
  
  const outChannels = Math.min(Number(attrs.out_channels || attrs.outChannels || 32), 64);
  const kernelSize = attrs.kernel_size || attrs.kernelSize || [3, 3];
  const kSize = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
  
  // Stack of feature maps
  const layers = Math.min(Math.ceil(outChannels / 8), 8);
  const layerSpacing = 0.08;
  const mapSize = 0.5 + Math.log2(kSize + 1) * 0.2;
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
      groupRef.current.rotation.y += 0.002;
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Stacked feature maps */}
      {Array.from({ length: layers }).map((_, i) => {
        const z = (i - (layers - 1) / 2) * layerSpacing;
        const alpha = 0.3 + (i / layers) * 0.5;
        return (
          <Box key={i} args={[mapSize, mapSize, 0.03]} position={[0, 0, z]}>
            <meshStandardMaterial
              color={displayColor}
              transparent
              opacity={alpha}
              metalness={0.2}
              roughness={0.6}
            />
          </Box>
        );
      })}
      
      {/* Kernel indicator */}
      <Box 
        args={[mapSize * 0.3, mapSize * 0.3, layers * layerSpacing + 0.1]} 
        position={[mapSize * 0.25, mapSize * 0.25, 0]}
      >
        <meshStandardMaterial
          color="#ffffff"
          transparent
          opacity={0.4}
          wireframe
        />
      </Box>
      
      {/* Selection indicator */}
      {selected && (
        <Box args={[mapSize + 0.1, mapSize + 0.1, layers * layerSpacing + 0.15]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      {/* Label */}
      <Text
        position={[0, mapSize / 2 + 0.2, 0]}
        fontSize={0.15}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
      
      <Text
        position={[0, -mapSize / 2 - 0.05, 0]}
        fontSize={0.08}
        color="#aaaaaa"
        anchorX="center"
        anchorY="top"
      >
        {outChannels}ch / {kSize}×{kSize}
      </Text>
    </group>
  );
}

/**
 * Pooling Layer - shows as a compressed block
 */
export function PoolingLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const attrs = node.attributes || {};
  const poolSize = attrs.kernel_size || attrs.kernelSize || [2, 2];
  const pSize = Array.isArray(poolSize) ? poolSize[0] : poolSize;
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Funnel shape representation */}
      <Cylinder args={[0.3, 0.2, 0.15, 8]} rotation={[Math.PI / 2, 0, 0]}>
        <meshStandardMaterial
          color={displayColor}
          metalness={0.2}
          roughness={0.6}
        />
      </Cylinder>
      
      {/* Grid overlay */}
      <mesh position={[0, 0, 0.08]}>
        <planeGeometry args={[0.5, 0.5, pSize, pSize]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.3} />
      </mesh>
      
      {selected && (
        <Box args={[0.7, 0.7, 0.25]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      <Text
        position={[0, 0.35, 0]}
        fontSize={0.15}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
    </group>
  );
}

/**
 * Normalization Layer - shows as a flat processing block
 */
export function NormLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      <Box args={[0.6, 0.4, 0.08]}>
        <meshStandardMaterial
          color={displayColor}
          metalness={0.3}
          roughness={0.5}
        />
      </Box>
      
      {/* Normalization symbol - horizontal lines */}
      {[-0.1, 0, 0.1].map((y, i) => (
        <mesh key={i} position={[0, y, 0.05]}>
          <planeGeometry args={[0.4, 0.02]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
        </mesh>
      ))}
      
      {selected && (
        <Box args={[0.7, 0.5, 0.15]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
    </group>
  );
}

/**
 * Activation Layer - shows as a small function block
 */
export function ActivationLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.2 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  // Draw activation function shape
  const activationType = node.type.toLowerCase();
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Sphere for activation */}
      <Sphere args={[0.2, 16, 16]}>
        <meshStandardMaterial
          color={displayColor}
          emissive={displayColor}
          emissiveIntensity={0.2}
          metalness={0.4}
          roughness={0.4}
        />
      </Sphere>
      
      {/* Function symbol */}
      <Text
        position={[0, 0, 0.22]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.01}
        outlineColor="#000000"
      >
        {activationType === 'relu' ? 'ƒ' : 
         activationType === 'sigmoid' ? 'σ' : 
         activationType === 'tanh' ? 'tanh' :
         activationType === 'softmax' ? 'soft' : 'ƒ'}
      </Text>
      
      {selected && (
        <Sphere args={[0.25, 8, 8]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Sphere>
      )}
      
      <Text
        position={[0, 0.35, 0]}
        fontSize={0.1}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.01}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
    </group>
  );
}

/**
 * LSTM/GRU/RNN Layer - shows as a recurrent block
 */
export function RecurrentLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const attrs = node.attributes || {};
  const hiddenSize = attrs.hidden_size || attrs.hiddenSize || 128;
  
  useFrame((_, delta) => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
      // Slow rotation to show recurrence
      groupRef.current.rotation.y += delta * 0.3;
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Main cell block */}
      <Box args={[0.5, 0.5, 0.3]}>
        <meshStandardMaterial
          color={displayColor}
          metalness={0.2}
          roughness={0.6}
        />
      </Box>
      
      {/* Recurrence loop */}
      <Torus args={[0.35, 0.03, 8, 32]} position={[0, 0, 0]} rotation={[0, 0, 0]}>
        <meshStandardMaterial
          color="#ffffff"
          transparent
          opacity={0.5}
        />
      </Torus>
      
      {/* Arrow indicator */}
      <mesh position={[0.35, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.05, 0.1, 8]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      
      {selected && (
        <Box args={[0.65, 0.65, 0.4]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      <Text
        position={[0, 0.4, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
      
      <Text
        position={[0, -0.35, 0]}
        fontSize={0.08}
        color="#aaaaaa"
        anchorX="center"
        anchorY="top"
      >
        {`h=${hiddenSize}`}
      </Text>
    </group>
  );
}

/**
 * Attention/Transformer Layer - shows as a multi-head attention block
 */
export function AttentionLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const attrs = node.attributes || {};
  const numHeads = Math.min(Number(attrs.num_heads || attrs.numHeads || 8), 12);
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  // Create attention head indicators
  const heads = useMemo(() => {
    const positions: [number, number, number][] = [];
    const radius = 0.25;
    for (let i = 0; i < numHeads; i++) {
      const angle = (i / numHeads) * Math.PI * 2;
      positions.push([Math.cos(angle) * radius, Math.sin(angle) * radius, 0]);
    }
    return positions;
  }, [numHeads]);
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Center query block */}
      <Box args={[0.15, 0.15, 0.15]} position={[0, 0, 0]}>
        <meshStandardMaterial color={displayColor} metalness={0.3} roughness={0.5} />
      </Box>
      
      {/* Attention heads */}
      {heads.map((pos, i) => (
        <group key={i}>
          <Sphere args={[0.06, 8, 8]} position={pos}>
            <meshStandardMaterial
              color={displayColor}
              transparent
              opacity={0.7}
            />
          </Sphere>
          {/* Connection to center */}
          <mesh>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, pos[0], pos[1], pos[2]])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#ffffff" transparent opacity={0.3} />
          </mesh>
        </group>
      ))}
      
      {selected && (
        <Sphere args={[0.4, 8, 8]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Sphere>
      )}
      
      <Text
        position={[0, 0.45, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
      
      <Text
        position={[0, -0.4, 0]}
        fontSize={0.08}
        color="#aaaaaa"
        anchorX="center"
        anchorY="top"
      >
        {numHeads} heads
      </Text>
    </group>
  );
}

/**
 * Embedding Layer - shows as a lookup table
 */
export function EmbeddingLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  const attrs = node.attributes || {};
  const vocabSize = attrs.num_embeddings || attrs.numEmbeddings || 10000;
  const embedDim = attrs.embedding_dim || attrs.embeddingDim || 256;
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      {/* Table representation - stacked rows */}
      {[0, 0.1, 0.2, 0.3].map((y, i) => (
        <Box key={i} args={[0.5, 0.08, 0.1]} position={[0, y - 0.15, i * 0.03]}>
          <meshStandardMaterial
            color={displayColor}
            transparent
            opacity={0.4 + i * 0.15}
          />
        </Box>
      ))}
      
      {/* Lookup arrow */}
      <mesh position={[-0.35, 0, 0.1]} rotation={[0, 0, Math.PI / 2]}>
        <coneGeometry args={[0.05, 0.15, 8]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      
      {selected && (
        <Box args={[0.7, 0.6, 0.3]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      <Text
        position={[0, 0.35, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
      
      <Text
        position={[0, -0.3, 0]}
        fontSize={0.07}
        color="#aaaaaa"
        anchorX="center"
        anchorY="top"
      >
        {`${vocabSize}→${embedDim}`}
      </Text>
    </group>
  );
}

/**
 * Generic/Other Layer - fallback
 */
export function GenericLayerMesh({
  node,
  color,
  selected = false,
  hovered = false,
  onClick,
  onPointerOver,
  onPointerOut,
}: NeuralLayerProps) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame(() => {
    if (groupRef.current) {
      const targetScale = hovered ? 1.1 : 1.0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const baseColor = new THREE.Color(color);
  const displayColor = selected ? baseColor.clone().multiplyScalar(1.3) : 
                       hovered ? baseColor.clone().multiplyScalar(1.15) : baseColor;
  
  return (
    <group 
      ref={groupRef}
      onClick={onClick}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      <Box args={[0.4, 0.4, 0.15]}>
        <meshStandardMaterial
          color={displayColor}
          metalness={0.2}
          roughness={0.6}
        />
      </Box>
      
      {selected && (
        <Box args={[0.5, 0.5, 0.2]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </Box>
      )}
      
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#000000"
      >
        {node.name}
      </Text>
    </group>
  );
}

/**
 * Get the appropriate layer mesh component based on node type
 */
export function getLayerMeshComponent(nodeType: string): React.ComponentType<NeuralLayerProps> {
  const type = nodeType.toLowerCase();
  
  if (type.includes('linear') || type.includes('dense') || type.includes('fc')) {
    return DenseLayerMesh;
  }
  if (type.includes('conv')) {
    return ConvLayerMesh;
  }
  if (type.includes('pool')) {
    return PoolingLayerMesh;
  }
  if (type.includes('norm') || type.includes('batch') || type.includes('layer')) {
    return NormLayerMesh;
  }
  if (type.includes('relu') || type.includes('sigmoid') || type.includes('tanh') || 
      type.includes('gelu') || type.includes('softmax') || type.includes('activation')) {
    return ActivationLayerMesh;
  }
  if (type.includes('lstm') || type.includes('gru') || type.includes('rnn')) {
    return RecurrentLayerMesh;
  }
  if (type.includes('attention') || type.includes('transformer')) {
    return AttentionLayerMesh;
  }
  if (type.includes('embed')) {
    return EmbeddingLayerMesh;
  }
  
  return GenericLayerMesh;
}
