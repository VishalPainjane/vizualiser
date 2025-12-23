import { useMemo } from 'react';
import { useVisualizerStore } from '@/core/store';
import { getEdgeComponent, EdgeStyle } from './EdgeGeometry';
import { SmartConnection } from './NeuralConnection';

/**
 * Renders all edges/connections in the network
 */
export function EdgeConnections() {
  const computedEdges = useVisualizerStore(state => state.computedEdges);
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  const config = useVisualizerStore(state => state.config);
  
  // Check if we should use enhanced neural visualization
  const useEnhancedEdges = useMemo(() => {
    // Check if any node has enhanced attributes (from backend analysis)
    for (const node of computedNodes.values()) {
      if (node.params && (
        node.params.out_features !== undefined ||
        node.params.outFeatures !== undefined ||
        node.params.out_channels !== undefined ||
        node.params.outChannels !== undefined
      )) {
        return true;
      }
    }
    return false;
  }, [computedNodes]);
  
  const EdgeComponent = useMemo(
    () => getEdgeComponent(config.edgeStyle as EdgeStyle || 'tube'),
    [config.edgeStyle]
  );
  
  if (!config.showEdges) return null;
  
  // Enhanced neural connections
  if (useEnhancedEdges) {
    return (
      <group name="edge-connections">
        {computedEdges.map((edge, index) => {
          // Get source and target node info for neuron counts
          const sourceNode = computedNodes.get(edge.source);
          const targetNode = computedNodes.get(edge.target);
          
          const sourceNeurons = sourceNode?.params?.out_features || 
                               sourceNode?.params?.outFeatures ||
                               sourceNode?.params?.out_channels ||
                               sourceNode?.params?.outChannels || 16;
          
          const targetNeurons = targetNode?.params?.in_features ||
                               targetNode?.params?.inFeatures ||
                               targetNode?.params?.in_channels ||
                               targetNode?.params?.inChannels || 16;
          
          return (
            <SmartConnection
              key={edge.id || `edge-${index}`}
              sourcePosition={edge.sourcePosition}
              targetPosition={edge.targetPosition}
              sourceNeurons={typeof sourceNeurons === 'number' ? sourceNeurons : 16}
              targetNeurons={typeof targetNeurons === 'number' ? targetNeurons : 16}
              color={edge.color}
              highlighted={edge.highlighted}
              animated={true}
              style="bundle"
            />
          );
        })}
      </group>
    );
  }
  
  // Original edge rendering
  return (
    <group name="edge-connections">
      {computedEdges.map((edge, index) => (
        <EdgeComponent
          key={edge.id || `edge-${index}`}
          edge={edge}
          style={config.edgeStyle as EdgeStyle}
          animated={false}
        />
      ))}
    </group>
  );
}

export { EdgeConnections as Edges };
