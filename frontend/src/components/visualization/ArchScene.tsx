/**
 * Architecture Scene - VGG-style 3D Visualization
 * Main scene component that orchestrates layout and rendering.
 */

import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line as DreiLine } from '@react-three/drei';

import { type ArchitectureLayout, type LayerBlock } from '@/core/arch-layout';
import { useLayoutWorker } from '@/hooks/useLayoutWorker';
import styles from './NeuralVisualizer.module.css';

// ============================================================================
// Types
// ============================================================================

export interface ArchSceneProps {
  architecture: any | null;
  showLabels?: boolean;
  showConnections?: boolean;
  onLayerClick?: (layerId: string) => void;
  selectedLayerId?: string | null;
}

// ============================================================================
// Sub-Components
// ============================================================================

const LayerBlockMesh: React.FC<{ block: LayerBlock; isSelected: boolean; onClick: () => void }> = ({ block, isSelected, onClick }) => {
  const { width, height, depth } = block.dimensions;
  const color = isSelected ? '#b4ff39' : block.color;
  return (
    <mesh position={[block.position.x, block.position.y, block.position.z]} onClick={onClick}>
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial color={color} transparent={block.opacity < 1} opacity={block.opacity} />
    </mesh>
  );
};

const ConnectionLine: React.FC<{ from: any; to: any; isSkip: boolean }> = ({ from, to, isSkip }) => (
  <DreiLine
    points={[[from.x, from.y, from.z], [to.x, to.y, to.z]]}
    color={isSkip ? '#D08050' : '#607080'}
    lineWidth={isSkip ? 2.5 : 2}
    dashed={isSkip}
    dashSize={0.15}
    gapSize={0.08}
  />
);

// ============================================================================
// Camera Controller
// ============================================================================

import { useThree } from '@react-three/fiber';
import { useEffect } from 'react';

const CameraController: React.FC<{ 
  position: { x: number; y: number; z: number }; 
  target: { x: number; y: number; z: number };
}> = ({ position, target }) => {
  const { camera, controls } = useThree();
  
  useEffect(() => {
    if (camera) {
      camera.position.set(position.x, position.y, position.z);
      camera.lookAt(target.x, target.y, target.z);
    }
    // Update OrbitControls target if available
    // @ts-ignore
    if (controls) {
      // @ts-ignore
      controls.target.set(target.x, target.y, target.z);
      // @ts-ignore
      controls.update();
    }
  }, [camera, controls, position, target]);
  
  return null;
};

// ============================================================================
// Main Scene Content
// ============================================================================

const SceneContent: React.FC<{
  layout: ArchitectureLayout;
  onLayerClick: (id: string) => void;
  selectedLayerId: string | null;
  showConnections: boolean;
  showLabels: boolean;
}> = ({ layout, onLayerClick, selectedLayerId, showConnections, showLabels }) => {
  return (
    // Center the entire model group based on the computed layout center
    <group position={[-layout.center.x, -layout.center.y, -layout.center.z]}>
      {layout.blocks.map(block => (
        <React.Fragment key={block.id}>
          <LayerBlockMesh
            block={block}
            isSelected={selectedLayerId === block.id}
            onClick={() => onLayerClick(block.id)}
          />
          {showLabels && (
            <Text
              position={[block.position.x, block.position.y + block.dimensions.height / 2 + 0.3, block.position.z]}
              fontSize={0.2}
              color="white"
              anchorX="center"
            >
              {block.displayName}
            </Text>
          )}
        </React.Fragment>
      ))}

      {showConnections && layout.connections.map((conn, i) => (
        <ConnectionLine
          key={`conn-${i}`}
          from={conn.fromPos}
          to={conn.toPos}
          isSkip={conn.isSkipConnection}
        />
      ))}
    </group>
  );
};

// ============================================================================
// Exported Component
// ============================================================================

export const ArchScene: React.FC<ArchSceneProps> = ({
  architecture,
  showLabels = true,
  showConnections = true,
  onLayerClick = () => {},
  selectedLayerId = null,
}) => {
  const { layout, isLoading, error } = useLayoutWorker(architecture);

  if (isLoading) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <span>COMPUTING_LAYOUT...</span>
      </div>
    );
  }
  
  if (error) {
    return (
        <div className={styles.error}>
            <h3>Layout Error</h3>
            <p>{error}</p>
        </div>
    );
  }
  
  if (!layout || layout.blocks.length === 0) {
    return (
      <div className={styles.empty}>
          <h3>No Model Loaded</h3>
          <p>Drop a file to begin visualization</p>
      </div>
    );
  }
  
  const cameraSuggestion = layout.cameraSuggestion || {
    position: { x: 0, y: 5, z: 20 },
    target: { x: 0, y: 0, z: 0 }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Canvas
        camera={{ fov: 50, position: [0, 5, 20] }} // Initial fallback
        style={{ background: '#1a1a2e' }}
      >
        <CameraController 
          position={cameraSuggestion.position} 
          target={cameraSuggestion.target} 
        />
        <gridHelper args={[200, 200, '#333', '#222']} position={[0, -1, 0]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <OrbitControls makeDefault />
        
        <SceneContent
          layout={layout}
          showLabels={showLabels}
          showConnections={showConnections}
          selectedLayerId={selectedLayerId}
          onLayerClick={onLayerClick}
        />
      </Canvas>
    </div>
  );
};

export default ArchScene;

