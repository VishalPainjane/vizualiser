/**
 * Architecture Scene - VGG-style 3D Visualization
 * 
 * Renders neural network architecture as 3D blocks where:
 * - Block HEIGHT = spatial dimension (shrinks through network)
 * - Block DEPTH = channel count (grows through network)
 * - Position flows left-to-right showing the data transformation
 */

import React, { useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment,
  Text,
  Html,
  RoundedBox,
  Line,
} from '@react-three/drei';
import * as THREE from 'three';

import { 
  computeArchitectureLayout, 
  type ArchitectureLayout, 
  type LayerBlock 
} from '@/core/arch-layout';

// ============================================================================
// Types
// ============================================================================

export type CameraView = 'front' | 'top' | 'side' | 'isometric' | 'back' | 'bottom';

export interface ArchSceneProps {
  architecture: {
    name: string;
    framework: string;
    totalParameters: number;
    inputShape?: number[] | null;
    outputShape?: number[] | null;
    layers: Array<{
      id: string;
      name: string;
      type: string;
      category: string;
      inputShape: number[] | null;
      outputShape: number[] | null;
      params: Record<string, unknown>;
      numParameters: number;
    }>;
    connections: Array<{
      source: string;
      target: string;
    }>;
  } | null;
  
  showLabels?: boolean;
  showDimensions?: boolean;
  showConnections?: boolean;
  onLayerClick?: (layerId: string) => void;
  onLayerHover?: (layerId: string | null) => void;
  selectedLayerId?: string | null;
  
  // Camera control props
  cameraView?: CameraView;
  onCameraChange?: () => void;
}

// ============================================================================
// Layer Block Component
// ============================================================================

interface LayerBlockMeshProps {
  block: LayerBlock;
  isSelected: boolean;
  isHovered: boolean;
  showLabel: boolean;
  showDimension: boolean;
  onClick: () => void;
  onHover: (hovered: boolean) => void;
}

const LayerBlockMesh: React.FC<LayerBlockMeshProps> = ({
  block,
  isSelected,
  isHovered,
  showLabel,
  showDimension,
  onClick,
  onHover,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoverAnim, setHoverAnim] = useState(0);
  
  // Animate on hover
  useFrame((_, delta) => {
    if (isHovered || isSelected) {
      setHoverAnim(prev => Math.min(1, prev + delta * 5));
    } else {
      setHoverAnim(prev => Math.max(0, prev - delta * 5));
    }
    
    if (meshRef.current) {
      const scale = 1 + hoverAnim * 0.05;
      meshRef.current.scale.setScalar(scale);
    }
  });
  
  const { width, height, depth } = block.dimensions;
  const color = new THREE.Color(block.color);
  const edgeColor = new THREE.Color(block.color).multiplyScalar(1.4); // Brighter edges
  
  // Lighten color on hover
  if (hoverAnim > 0) {
    color.lerp(new THREE.Color('#ffffff'), hoverAnim * 0.3);
  }
  
  return (
    <group position={[block.position.x, block.position.y, block.position.z]}>
      {/* Glow/shadow base for depth */}
      <RoundedBox
        args={[width + 0.08, height + 0.08, depth + 0.08]}
        radius={0.04}
        smoothness={4}
      >
        <meshBasicMaterial
          color={block.color}
          transparent
          opacity={0.15}
        />
      </RoundedBox>
      
      {/* Main block */}
      <RoundedBox
        ref={meshRef}
        args={[width, height, depth]}
        radius={0.03}
        smoothness={4}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerEnter={(e) => { e.stopPropagation(); onHover(true); }}
        onPointerLeave={() => onHover(false)}
      >
        <meshStandardMaterial
          color={color}
          roughness={0.2}
          metalness={0.05}
        />
      </RoundedBox>
      
      {/* Edge highlight - subtle wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(width, height, depth)]} />
        <lineBasicMaterial color={edgeColor} transparent opacity={0.5} />
      </lineSegments>
      
      {/* Selection outline */}
      {isSelected && (
        <RoundedBox
          args={[width + 0.1, height + 0.1, depth + 0.1]}
          radius={0.04}
          smoothness={4}
        >
          <meshBasicMaterial
            color="#b4ff39"
            transparent
            opacity={0.4}
            side={THREE.BackSide}
          />
        </RoundedBox>
      )}
      
      {/* Top label (layer name) */}
      {showLabel && (
        <Text
          position={[0, height / 2 + 0.35, 0]}
          fontSize={0.18}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="#000000"
          maxWidth={2.0}
        >
          {block.label}
        </Text>
      )}
      
      {/* Bottom label (dimensions) */}
      {showDimension && block.dimensionLabel && (
        <Text
          position={[0, -height / 2 - 0.2, 0]}
          fontSize={0.14}
          color="#b4ff39"
          anchorX="center"
          anchorY="top"
          outlineWidth={0.015}
          outlineColor="#000000"
        >
          {block.dimensionLabel}
        </Text>
      )}
      
      {/* Hover tooltip */}
      {isHovered && (
        <Html
          position={[0, height / 2 + 0.5, 0]}
          center
          style={{ pointerEvents: 'none' }}
        >
          <div style={{
            background: 'rgba(0,0,0,0.9)',
            padding: '10px 14px',
            borderRadius: '6px',
            color: 'white',
            fontSize: '13px',
            whiteSpace: 'nowrap',
            border: `2px solid ${block.color}`,
            boxShadow: `0 0 15px ${block.color}40`,
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
              {block.name}
            </div>
            <div style={{ color: '#aaa', fontSize: '10px' }}>
              Type: {block.type}
            </div>
            {block.outputShape && (
              <div style={{ color: '#aaa', fontSize: '10px' }}>
                Output: {block.outputShape.height}×{block.outputShape.width}×{block.outputShape.channels}
              </div>
            )}
            {block.numParameters > 0 && (
              <div style={{ color: '#aaa', fontSize: '10px' }}>
                Params: {block.numParameters.toLocaleString()}
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
};

// ============================================================================
// Connection Lines (supports skip connections with curves)
// ============================================================================

interface ConnectionLineProps {
  from: { x: number; y: number; z: number };
  to: { x: number; y: number; z: number };
  isSkipConnection: boolean;
}

const ConnectionLine: React.FC<ConnectionLineProps> = ({ from, to, isSkipConnection }) => {
  // For skip connections or connections between different Y levels, use a curve
  const needsCurve = isSkipConnection || Math.abs(from.y - to.y) > 0.1;
  
  const points: [number, number, number][] = useMemo(() => {
    const distance = Math.sqrt(
      Math.pow(to.x - from.x, 2) + 
      Math.pow(to.y - from.y, 2) + 
      Math.pow(to.z - from.z, 2)
    );
    
    if (!needsCurve && distance < 3) {
      // Short direct connections - smooth bezier
      const midX = (from.x + to.x) / 2;
      return [
        [from.x, from.y, from.z],
        [from.x + 0.2, from.y, from.z],
        [midX, from.y, from.z],
        [to.x - 0.2, to.y, to.z],
        [to.x, to.y, to.z],
      ];
    }
    
    // Create a curved path for skip connections or long connections
    const midX = (from.x + to.x) / 2;
    const offsetZ = isSkipConnection ? 2.0 : 0.8; // Arc out in Z for visibility
    const offsetY = (to.y - from.y) / 2;
    const curveHeight = Math.min(distance * 0.15, 1.0); // Gentle curve based on distance
    
    // More control points for smoother curves
    return [
      [from.x, from.y, from.z],
      [from.x + 0.4, from.y + curveHeight * 0.3, from.z + offsetZ * 0.2],
      [from.x + (midX - from.x) * 0.4, from.y + offsetY * 0.5 + curveHeight, from.z + offsetZ * 0.6],
      [midX, from.y + offsetY, from.z + offsetZ],
      [to.x - (to.x - midX) * 0.4, to.y - offsetY * 0.5 + curveHeight, to.z + offsetZ * 0.6],
      [to.x - 0.4, to.y + curveHeight * 0.3, to.z + offsetZ * 0.2],
      [to.x, to.y, to.z],
    ];
  }, [from, to, needsCurve, isSkipConnection]);
  
  // Connection colors - subtle gray for normal, warm for skip
  const normalColor = '#607080'; // Muted slate
  const skipColor = '#D08050';   // Soft orange for residual/skip
  const glowColor = isSkipConnection ? '#E0A070' : '#708090';
  
  return (
    <group>
      {/* Glow effect - thicker background line */}
      <Line
        points={points}
        color={glowColor}
        lineWidth={isSkipConnection ? 4 : 3}
        transparent
        opacity={isSkipConnection ? 0.2 : 0.12}
      />
      {/* Main connection line */}
      <Line
        points={points}
        color={isSkipConnection ? skipColor : normalColor}
        lineWidth={isSkipConnection ? 2.5 : 2}
        transparent
        opacity={isSkipConnection ? 0.9 : 0.7}
        dashed={isSkipConnection}
        dashSize={0.15}
        gapSize={0.08}
      />
      {/* Arrow head at destination */}
      <mesh position={[to.x - 0.08, to.y, to.z]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.08, 0.2, 8]} />
        <meshBasicMaterial 
          color={isSkipConnection ? skipColor : normalColor} 
          transparent 
          opacity={isSkipConnection ? 0.9 : 0.7} 
        />
      </mesh>
    </group>
  );
};

// ============================================================================
// Camera Controller - Full user control with view presets
// ============================================================================

interface CameraControllerProps {
  bounds: ArchitectureLayout['bounds'];
  center: { x: number; y: number; z: number };
  isLinear: boolean;
  cameraView?: CameraView;
}

const CameraController: React.FC<CameraControllerProps> = ({ bounds, center, isLinear, cameraView }) => {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  
  // Calculate optimal distance based on model size
  const getOptimalDistance = React.useCallback(() => {
    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;
    const depth = bounds.maxZ - bounds.minZ;
    const maxDim = Math.max(width, height, depth);
    
    const fov = (camera as THREE.PerspectiveCamera).fov || 50;
    const fovRad = (fov / 2) * Math.PI / 180;
    return Math.max((maxDim / 2) / Math.tan(fovRad) * 1.8, 3);
  }, [bounds, camera]);
  
  // Apply camera view preset
  React.useEffect(() => {
    const distance = getOptimalDistance();
    let pos: [number, number, number];
    
    switch (cameraView) {
      case 'front':
        pos = [center.x, center.y, center.z + distance];
        break;
      case 'back':
        pos = [center.x, center.y, center.z - distance];
        break;
      case 'top':
        pos = [center.x, center.y + distance, center.z + 0.01];
        break;
      case 'bottom':
        pos = [center.x, center.y - distance, center.z + 0.01];
        break;
      case 'side':
        pos = [center.x + distance, center.y, center.z];
        break;
      case 'isometric':
        pos = [
          center.x + distance * 0.6,
          center.y + distance * 0.5,
          center.z + distance * 0.6
        ];
        break;
      default:
        // Default: auto based on architecture type
        if (isLinear) {
          const height = bounds.maxY - bounds.minY;
          pos = [center.x, center.y + height * 0.3, center.z + distance];
        } else {
          pos = [
            center.x + distance * 0.3,
            center.y + distance * 0.5,
            center.z + distance * 0.8
          ];
        }
    }
    
    camera.position.set(...pos);
    camera.lookAt(center.x, center.y, center.z);
    camera.updateProjectionMatrix();
    
    if (controlsRef.current) {
      controlsRef.current.target.set(center.x, center.y, center.z);
      controlsRef.current.update();
    }
  }, [bounds, center, camera, isLinear, cameraView, getOptimalDistance]);
  
  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enablePan
      enableZoom
      enableRotate
      // Extreme zoom range
      minDistance={0.1}
      maxDistance={1000}
      // Fast controls
      zoomSpeed={1.5}
      rotateSpeed={1.0}
      panSpeed={1.5}
      // Smooth damping
      enableDamping
      dampingFactor={0.1}
      // Full rotation freedom
      minPolarAngle={0}
      maxPolarAngle={Math.PI}
      minAzimuthAngle={-Infinity}
      maxAzimuthAngle={Infinity}
      // Mouse controls: LEFT = PAN (drag to move), MIDDLE = ZOOM, RIGHT = ROTATE
      // Hold SPACE for rotation mode (handled by screenSpacePanning)
      mouseButtons={{
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.ROTATE,
      }}
      // Pan in screen space (left/right = left/right, up/down = up/down)
      screenSpacePanning={true}
      // Touch controls: one finger = pan, two fingers = zoom/rotate
      touches={{
        ONE: THREE.TOUCH.PAN,
        TWO: THREE.TOUCH.DOLLY_ROTATE,
      }}
      // Keyboard controls enabled
      keyPanSpeed={25}
    />
  );
};

// ============================================================================
// Legend (fixed position HTML overlay - not in 3D space)
// ============================================================================

const LegendOverlay: React.FC = () => {
  const items = [
    { color: '#5B8BD9', label: 'Conv' },
    { color: '#E07070', label: 'Pool' },
    { color: '#6BAF6B', label: 'FC/Linear' },
    { color: '#D9A740', label: 'Activation' },
    { color: '#50A8A0', label: 'Norm' },
    { color: '#9070C0', label: 'Attention' },
    { color: '#708090', label: 'Dropout' },
  ];
  
  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      left: '20px',
      background: 'rgba(10, 10, 26, 0.9)',
      padding: '12px 16px',
      borderRadius: '6px',
      color: 'white',
      fontSize: '11px',
      fontFamily: 'JetBrains Mono, monospace',
      backdropFilter: 'blur(8px)',
      border: '1px dashed rgba(180, 255, 57, 0.3)',
      zIndex: 100,
      pointerEvents: 'none',
    }}>
      <div style={{ 
        fontWeight: 'bold', 
        marginBottom: '8px', 
        fontSize: '11px',
        color: '#b4ff39',
        letterSpacing: '0.05em',
      }}>
        LAYER_TYPES
      </div>
      {items.map(item => (
        <div key={item.label} style={{ display: 'flex', alignItems: 'center', marginBottom: '3px' }}>
          <div style={{
            width: '10px',
            height: '10px',
            backgroundColor: item.color,
            marginRight: '8px',
            borderRadius: '2px',
          }} />
          <span style={{ opacity: 0.85, fontSize: '10px' }}>{item.label}</span>
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// Main Scene
// ============================================================================

const SceneContent: React.FC<{
  layout: ArchitectureLayout;
  showLabels: boolean;
  showDimensions: boolean;
  showConnections: boolean;
  selectedLayerId: string | null;
  onLayerClick: (id: string) => void;
  onLayerHover: (id: string | null) => void;
  cameraView?: CameraView;
}> = ({
  layout,
  showLabels,
  showDimensions,
  showConnections,
  selectedLayerId,
  onLayerClick,
  onLayerHover,
  cameraView,
}) => {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  
  const handleHover = (id: string, hovered: boolean) => {
    const newId = hovered ? id : null;
    setHoveredId(newId);
    onLayerHover(newId);
  };
  
  // Calculate grid size based on model
  const gridSize = Math.max(
    layout.bounds.maxX - layout.bounds.minX,
    layout.bounds.maxY - layout.bounds.minY,
    10
  );
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={0.7} />
      <directionalLight position={[-10, 5, -5]} intensity={0.3} />
      <pointLight position={[layout.center.x, layout.center.y + 5, 5]} intensity={0.3} />
      
      {/* Environment */}
      <Environment preset="city" />
      
      {/* Camera setup with integrated controls */}
      <CameraController 
        bounds={layout.bounds} 
        center={layout.center}
        isLinear={layout.isLinear}
        cameraView={cameraView}
      />
      
      {/* Grid - centered below model */}
      <group position={[layout.center.x, layout.bounds.minY - 0.5, layout.center.z]}>
        <gridHelper
          args={[gridSize * 1.5, 20, '#333333', '#222222']}
          rotation={[0, 0, 0]}
        />
      </group>
      
      {/* Connections */}
      {showConnections && layout.connections.map((conn, i) => (
        <ConnectionLine
          key={`conn-${i}`}
          from={conn.fromPos}
          to={conn.toPos}
          isSkipConnection={conn.isSkipConnection}
        />
      ))}
      
      {/* Layer blocks */}
      {layout.blocks.map(block => (
        <LayerBlockMesh
          key={block.id}
          block={block}
          isSelected={selectedLayerId === block.id}
          isHovered={hoveredId === block.id}
          showLabel={showLabels}
          showDimension={showDimensions}
          onClick={() => onLayerClick(block.id)}
          onHover={(h) => handleHover(block.id, h)}
        />
      ))}
    </>
  );
};

// ============================================================================
// Exported Component
// ============================================================================

export const ArchScene: React.FC<ArchSceneProps> = ({
  architecture,
  showLabels = true,
  showDimensions = true,
  showConnections = true,
  onLayerClick,
  onLayerHover,
  selectedLayerId = null,
  cameraView,
}) => {
  // Compute layout from architecture
  const layout = useMemo(() => {
    if (!architecture) return null;
    return computeArchitectureLayout(architecture);
  }, [architecture]);
  
  if (!layout) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#1a1a2e',
        color: '#888',
      }}>
        Drop a model file to visualize
      </div>
    );
  }
  
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Canvas
        camera={{
          fov: 50,
          near: 0.01,
          far: 5000,
          position: [0, 5, 20],
        }}
        style={{ background: 'linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%)' }}
      >
        <SceneContent
          layout={layout}
          showLabels={showLabels}
          showDimensions={showDimensions}
          showConnections={showConnections}
          selectedLayerId={selectedLayerId}
          onLayerClick={onLayerClick || (() => {})}
          onLayerHover={onLayerHover || (() => {})}
          cameraView={cameraView}
        />
      </Canvas>
      <LegendOverlay />
    </div>
  );
};

export default ArchScene;
