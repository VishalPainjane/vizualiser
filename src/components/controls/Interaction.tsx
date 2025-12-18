import { useCallback, useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useVisualizerStore } from '@/core/store';

/**
 * Hook for raycasting and object picking
 */
export function useRaycast() {
  const { camera, scene, gl } = useThree();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  
  const getIntersections = useCallback((event: MouseEvent | PointerEvent) => {
    const rect = gl.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(scene.children, true);
  }, [camera, scene, gl]);
  
  return { getIntersections };
}

/**
 * Keyboard shortcuts handler
 */
export function useKeyboardShortcuts() {
  const resetCamera = useVisualizerStore(state => state.resetCamera);
  const selectNode = useVisualizerStore(state => state.selectNode);
  const selection = useVisualizerStore(state => state.selection);
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  const updateConfig = useVisualizerStore(state => state.updateConfig);
  const config = useVisualizerStore(state => state.config);
  
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      switch (event.key) {
        case 'Escape':
          // Deselect current selection
          selectNode(null);
          break;
        
        case 'r':
        case 'R':
          // Reset camera
          if (!event.ctrlKey && !event.metaKey) {
            resetCamera();
          }
          break;
        
        case 'l':
        case 'L':
          // Toggle labels
          updateConfig({ showLabels: !config.showLabels });
          break;
        
        case 'e':
        case 'E':
          // Toggle edges
          updateConfig({ showEdges: !config.showEdges });
          break;
        
        case 'ArrowUp':
        case 'ArrowDown': {
          // Navigate between nodes
          if (selection.selectedNodeId) {
            const nodeIds = Array.from(computedNodes.keys());
            const currentIndex = nodeIds.indexOf(selection.selectedNodeId);
            const nextIndex = event.key === 'ArrowDown'
              ? Math.min(currentIndex + 1, nodeIds.length - 1)
              : Math.max(currentIndex - 1, 0);
            selectNode(nodeIds[nextIndex]);
          }
          break;
        }
        
        default:
          break;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [resetCamera, selectNode, selection, computedNodes, updateConfig, config]);
}

/**
 * Touch gesture handler for mobile
 */
export function useTouchGestures() {
  // Placeholder for touch gesture handling
  // Can be expanded for pinch-to-zoom, two-finger rotate, etc.
}

/**
 * LOD (Level of Detail) manager based on camera distance
 */
export function useLODManager() {
  const { camera } = useThree();
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  const updateNodeLOD = useVisualizerStore(state => state.updateNodeLOD);
  
  // LOD thresholds
  const LOD_DISTANCES = {
    HIGH: 20,    // LOD 0 (full detail) when closer than this
    MEDIUM: 40,  // LOD 1 (medium detail)
    LOW: 80,     // LOD 2 (low detail)
  };
  
  const updateLOD = useCallback(() => {
    const lodMap = new Map<string, number>();
    const cameraPos = camera.position;
    
    computedNodes.forEach((node, id) => {
      const nodePos = new THREE.Vector3(
        node.computedPosition.x,
        node.computedPosition.y,
        node.computedPosition.z
      );
      const distance = cameraPos.distanceTo(nodePos);
      
      let lod = 0;
      if (distance > LOD_DISTANCES.LOW) {
        lod = 2;
      } else if (distance > LOD_DISTANCES.MEDIUM) {
        lod = 1;
      }
      
      lodMap.set(id, lod);
    });
    
    updateNodeLOD(lodMap);
  }, [camera, computedNodes, updateNodeLOD]);
  
  return { updateLOD };
}
