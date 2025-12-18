import { useCallback } from 'react';
import { DropZone, NeuralVisualizer } from './components';
import { useVisualizerStore } from './core/store';
import { LAYER_CATEGORIES, type LayerType } from '@/schema/types';
import type { NN3DModel } from '@/schema/types';
import './App.css';

/**
 * Main Application Component
 * 
 * Integrates the new 3D Neural Network Visualization System
 */
function App() {
  const model = useVisualizerStore(state => state.model);
  const isLoading = useVisualizerStore(state => state.isLoading);
  const error = useVisualizerStore(state => state.error);
  const selectNode = useVisualizerStore(state => state.selectNode);
  const clearModel = useVisualizerStore(state => state.clearModel);
  const loadModel = useVisualizerStore(state => state.loadModel);

  // Build architecture data from store model for the visualizer
  const architecture = model ? {
    name: model.metadata?.name || 'Model',
    framework: model.metadata?.framework || 'Unknown',
    totalParameters: model.metadata?.totalParams || 0,
    trainableParameters: model.metadata?.trainableParams,
    inputShape: model.metadata?.inputShape as number[] | null || null,
    outputShape: model.metadata?.outputShape as number[] | null || null,
    layers: model.graph.nodes.map(node => {
      // Get category from attributes (set by backend) or infer from layer type
      const backendCategory = node.attributes?.category as string | undefined;
      const layerType = node.type as LayerType;
      const category = backendCategory || LAYER_CATEGORIES[layerType] || 'other';
      
      // Extract num parameters from attributes or params
      const numParameters = 
        (node.attributes?.parameters as number) || 
        (node.params?.totalParams ? parseInt(String(node.params.totalParams).replace(/,/g, '')) : 0) ||
        0;
      
      return {
        id: node.id,
        name: node.name,
        type: node.type,
        category,
        inputShape: (node.inputShape as number[] | null) || null,
        outputShape: (node.outputShape as number[] | null) || null,
        params: node.params || {},
        numParameters,
        trainable: true,
      };
    }),
    connections: model.graph.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      tensorShape: (edge.tensorShape as number[] | null) || null,
    })),
  } : null;

  // Handle layer selection
  const handleLayerSelect = useCallback((layerId: string | null) => {
    selectNode(layerId);
  }, [selectNode]);

  // Handle uploading a new model (clear current)
  const handleUploadNew = useCallback(() => {
    clearModel();
  }, [clearModel]);

  // Handle loading a saved model from database
  const handleLoadSavedModel = useCallback((arch: any) => {
    // Convert architecture back to NN3DModel format
    const savedModel: NN3DModel = {
      version: '1.0',
      metadata: {
        name: arch.name,
        framework: arch.framework,
        totalParams: arch.totalParameters,
        trainableParams: arch.trainableParameters,
        inputShape: arch.inputShape,
        outputShape: arch.outputShape,
      },
      graph: {
        nodes: arch.layers.map((layer: any) => ({
          id: layer.id,
          name: layer.name,
          type: layer.type,
          inputShape: layer.inputShape,
          outputShape: layer.outputShape,
          params: layer.params,
          attributes: {
            category: layer.category,
            parameters: layer.numParameters,
          },
        })),
        edges: arch.connections.map((conn: any, idx: number) => ({
          id: `edge-${idx}`,
          source: conn.source,
          target: conn.target,
          tensorShape: conn.tensorShape,
        })),
      },
    };
    
    loadModel(savedModel);
  }, [loadModel]);

  return (
    <div className="app">
      <DropZone>
        <NeuralVisualizer
          architecture={architecture}
          isLoading={isLoading}
          error={error}
          onLayerSelect={handleLayerSelect}
          onUploadNew={handleUploadNew}
          onLoadSavedModel={handleLoadSavedModel}
        />
      </DropZone>
    </div>
  );
}

export default App;
