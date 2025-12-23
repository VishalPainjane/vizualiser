/**
 * Keras/TensorFlow HDF5 format parser
 * Parses HDF5 files directly in the browser using h5wasm
 */

import type { NN3DModel, LayerType } from '@/schema/types';
import type { ParseResult, FormatParser, ExtractedLayer } from './types';
import { detectFormatFromExtension } from './format-detector';

/**
 * Map Keras layer class names to NN3D types
 */
const KERAS_LAYER_MAP: Record<string, LayerType> = {
  // Core layers
  'Dense': 'linear',
  'Activation': 'relu',
  'Embedding': 'embedding',
  'Masking': 'custom',
  'Lambda': 'custom',
  
  // Convolutional layers
  'Conv1D': 'conv1d',
  'Conv2D': 'conv2d',
  'Conv3D': 'conv3d',
  'Conv2DTranspose': 'convTranspose2d',
  'Conv3DTranspose': 'conv3d',
  'SeparableConv1D': 'conv1d',
  'SeparableConv2D': 'separableConv2d',
  'DepthwiseConv2D': 'depthwiseConv2d',
  
  // Pooling layers
  'MaxPooling1D': 'maxPool1d',
  'MaxPooling2D': 'maxPool2d',
  'MaxPooling3D': 'maxPool2d',
  'AveragePooling1D': 'avgPool2d',
  'AveragePooling2D': 'avgPool2d',
  'AveragePooling3D': 'avgPool2d',
  'GlobalMaxPooling1D': 'globalAvgPool',
  'GlobalMaxPooling2D': 'globalAvgPool',
  'GlobalAveragePooling1D': 'globalAvgPool',
  'GlobalAveragePooling2D': 'globalAvgPool',
  
  // Recurrent layers
  'LSTM': 'lstm',
  'GRU': 'gru',
  'SimpleRNN': 'rnn',
  'Bidirectional': 'lstm',
  
  // Normalization layers
  'BatchNormalization': 'batchNorm2d',
  'LayerNormalization': 'layerNorm',
  'GroupNormalization': 'groupNorm',
  
  // Regularization layers
  'Dropout': 'dropout',
  'SpatialDropout1D': 'dropout',
  'SpatialDropout2D': 'dropout',
  'SpatialDropout3D': 'dropout',
  'GaussianDropout': 'dropout',
  'GaussianNoise': 'custom',
  
  // Attention layers
  'Attention': 'attention',
  'MultiHeadAttention': 'multiHeadAttention',
  'AdditiveAttention': 'attention',
  
  // Reshaping layers
  'Reshape': 'reshape',
  'Flatten': 'flatten',
  'RepeatVector': 'reshape',
  'Permute': 'reshape',
  'Cropping1D': 'custom',
  'Cropping2D': 'custom',
  'Cropping3D': 'custom',
  'UpSampling1D': 'upsample',
  'UpSampling2D': 'upsample',
  'UpSampling3D': 'upsample',
  'ZeroPadding1D': 'pad',
  'ZeroPadding2D': 'pad',
  'ZeroPadding3D': 'pad',
  
  // Merge layers
  'Concatenate': 'concat',
  'Add': 'add',
  'Subtract': 'add',
  'Multiply': 'multiply',
  'Average': 'add',
  'Maximum': 'add',
  'Minimum': 'add',
  'Dot': 'multiply',
  
  // Activation layers
  'ReLU': 'relu',
  'LeakyReLU': 'leakyRelu',
  'PReLU': 'leakyRelu',
  'ELU': 'relu',
  'ThresholdedReLU': 'relu',
  'Softmax': 'softmax',
  
  // Input layer
  'InputLayer': 'input',
};



/**
 * Extract layer info from Keras config
 */
function extractKerasLayerInfo(config: any): Partial<ExtractedLayer> {
  const params: Record<string, unknown> = {};
  
  if (config.units) params.outFeatures = config.units;
  if (config.filters) params.outChannels = config.filters;
  if (config.kernel_size) params.kernelSize = config.kernel_size;
  if (config.strides) params.stride = config.strides;
  if (config.padding) params.padding = config.padding;
  if (config.activation) params.activation = config.activation;
  if (config.rate) params.dropoutRate = config.rate;
  if (config.input_dim) params.inFeatures = config.input_dim;
  if (config.output_dim) params.outFeatures = config.output_dim;
  if (config.num_heads) params.numHeads = config.num_heads;
  if (config.key_dim) params.hiddenSize = config.key_dim;
  
  return { params };
}

/**
 * Parse model config from HDF5 attributes
 */
function parseModelConfig(configStr: string): any {
  try {
    return JSON.parse(configStr);
  } catch {
    return null;
  }
}

/**
 * Parse Keras Sequential model
 */
function parseSequentialModel(config: any): { layers: ExtractedLayer[], connections: Array<{source: string, target: string}> } {
  const layers: ExtractedLayer[] = [];
  const connections: Array<{source: string, target: string}> = [];
  
  const layerConfigs = config.config?.layers || config.config || [];
  
  // Add input node
  layers.push({
    id: 'input',
    name: 'Input',
    type: 'input',
  });
  
  let prevId = 'input';
  
  for (let i = 0; i < layerConfigs.length; i++) {
    const layerConfig = layerConfigs[i];
    const className = layerConfig.class_name || layerConfig.className || 'Unknown';
    const name = layerConfig.config?.name || layerConfig.name || `layer_${i}`;
    const layerId = name.replace(/[^a-zA-Z0-9_]/g, '_');
    
    const layerType = KERAS_LAYER_MAP[className] || 'custom';
    const layerInfo = extractKerasLayerInfo(layerConfig.config || {});
    
    layers.push({
      id: layerId,
      name: name,
      type: layerType,
      params: layerInfo.params,
      attributes: { kerasClass: className },
    });
    
    connections.push({ source: prevId, target: layerId });
    prevId = layerId;
  }
  
  // Add output node
  layers.push({
    id: 'output',
    name: 'Output',
    type: 'output',
  });
  connections.push({ source: prevId, target: 'output' });
  
  return { layers, connections };
}

/**
 * Parse Keras Functional model
 */
function parseFunctionalModel(config: any): { layers: ExtractedLayer[], connections: Array<{source: string, target: string}> } {
  const layers: ExtractedLayer[] = [];
  const connections: Array<{source: string, target: string}> = [];
  
  const layerConfigs = config.config?.layers || [];
  const inputLayers = config.config?.input_layers || [];
  const outputLayers = config.config?.output_layers || [];
  
  // Map layer names to IDs
  const nameToId = new Map<string, string>();
  
  for (const layerConfig of layerConfigs) {
    const className = layerConfig.class_name || 'Unknown';
    const name = layerConfig.config?.name || layerConfig.name;
    const layerId = name.replace(/[^a-zA-Z0-9_]/g, '_');
    nameToId.set(name, layerId);
    
    const layerType = KERAS_LAYER_MAP[className] || 'custom';
    const layerInfo = extractKerasLayerInfo(layerConfig.config || {});
    
    // Check if this is an input layer
    const isInput = className === 'InputLayer' || inputLayers.some((il: any) => il[0] === name);
    
    layers.push({
      id: layerId,
      name: name,
      type: isInput ? 'input' : layerType,
      params: layerInfo.params,
      attributes: { kerasClass: className },
    });
    
    // Parse inbound connections
    const inboundNodes = layerConfig.inbound_nodes || [];
    for (const inbound of inboundNodes) {
      if (Array.isArray(inbound)) {
        for (const conn of inbound) {
          const sourceName = Array.isArray(conn) ? conn[0] : conn;
          const sourceId = nameToId.get(sourceName);
          if (sourceId) {
            connections.push({ source: sourceId, target: layerId });
          }
        }
      }
    }
  }
  
  // Add output markers for output layers
  for (const outputLayer of outputLayers) {
    const name = outputLayer[0];
    const layerId = nameToId.get(name);
    if (layerId) {
      const layer = layers.find(l => l.id === layerId);
      if (layer) {
        layer.type = 'output';
      }
    }
  }
  
  return { layers, connections };
}

/**
 * Infer structure from weight names when no config available
 */
function inferStructureFromWeights(weightNames: string[]): { layers: ExtractedLayer[], connections: Array<{source: string, target: string}> } {
  const layers: ExtractedLayer[] = [];
  const connections: Array<{source: string, target: string}> = [];
  const layerSet = new Set<string>();
  
  // Extract layer names from weight paths
  for (const name of weightNames) {
    // Typical patterns: "layer_name/kernel:0", "layer_name/bias:0"
    const parts = name.split('/');
    if (parts.length >= 1) {
      layerSet.add(parts[0]);
    }
  }
  
  const layerNames = Array.from(layerSet).sort();
  
  // Add input
  layers.push({ id: 'input', name: 'Input', type: 'input' });
  
  let prevId = 'input';
  for (const name of layerNames) {
    const layerId = name.replace(/[^a-zA-Z0-9_]/g, '_');
    const layerType = inferLayerTypeFromName(name);
    
    layers.push({
      id: layerId,
      name: name,
      type: layerType,
    });
    
    connections.push({ source: prevId, target: layerId });
    prevId = layerId;
  }
  
  // Add output
  layers.push({ id: 'output', name: 'Output', type: 'output' });
  connections.push({ source: prevId, target: 'output' });
  
  return { layers, connections };
}

/**
 * Infer layer type from name
 */
function inferLayerTypeFromName(name: string): LayerType {
  const lower = name.toLowerCase();
  
  if (lower.includes('conv')) return 'conv2d';
  if (lower.includes('dense') || lower.includes('fc')) return 'linear';
  if (lower.includes('batch_norm') || lower.includes('bn')) return 'batchNorm2d';
  if (lower.includes('layer_norm') || lower.includes('ln')) return 'layerNorm';
  if (lower.includes('dropout')) return 'dropout';
  if (lower.includes('pool')) return 'maxPool2d';
  if (lower.includes('flatten')) return 'flatten';
  if (lower.includes('lstm')) return 'lstm';
  if (lower.includes('gru')) return 'gru';
  if (lower.includes('embed')) return 'embedding';
  if (lower.includes('attention') || lower.includes('attn')) return 'multiHeadAttention';
  if (lower.includes('relu')) return 'relu';
  if (lower.includes('softmax')) return 'softmax';
  
  return 'linear';
}

/**
 * Keras/TensorFlow HDF5 format parser
 */
export const KerasParser: FormatParser = {
  extensions: ['.h5', '.hdf5', '.keras'],
  
  async canParse(file: File): Promise<boolean> {
    const ext = file.name.toLowerCase();
    return ext.endsWith('.h5') || ext.endsWith('.hdf5') || ext.endsWith('.keras');
  },
  
  async parse(file: File): Promise<ParseResult> {
    const format = detectFormatFromExtension(file.name);
    const warnings: string[] = [];
    
    try {
      // Dynamically import h5wasm
      const h5wasm = await import('h5wasm');
      await h5wasm.ready;
      
      // Read file as ArrayBuffer
      const buffer = await file.arrayBuffer();
      
      // Create a temporary file name for h5wasm
      // h5wasm expects the file to be written to its virtual filesystem
      const tempFileName = `/${file.name}`;
      
      // Write the buffer to the virtual filesystem
      if (!h5wasm.FS) {
        throw new Error('h5wasm filesystem not initialized');
      }
      h5wasm.FS.writeFile(tempFileName, new Uint8Array(buffer));
      
      // Open HDF5 file from virtual filesystem
      const h5file = new h5wasm.File(tempFileName, 'r');
      
      let layers: ExtractedLayer[] = [];
      let connections: Array<{source: string, target: string}> = [];
      let modelName = file.name.replace(/\.(h5|hdf5|keras)$/i, '');
      let totalParams = 0;
      
      // Try to get model config from attributes
      let modelConfig: any = null;
      
      try {
        // Keras 2.x stores config in root attributes
        const modelConfigAttr = h5file.attrs['model_config'];
        if (modelConfigAttr) {
          const configStr = typeof modelConfigAttr.value === 'string' 
            ? modelConfigAttr.value 
            : new TextDecoder().decode(modelConfigAttr.value as Uint8Array);
          modelConfig = parseModelConfig(configStr);
        }
      } catch (e) {
        warnings.push('Could not read model_config attribute');
      }
      
      // Try keras config location
      if (!modelConfig) {
        try {
          const kerasConfigAttr = h5file.attrs['keras_version'];
          if (kerasConfigAttr) {
            warnings.push(`Keras version: ${kerasConfigAttr.value}`);
          }
        } catch {
          // Ignore
        }
      }
      
      // Parse based on model config or infer from structure
      if (modelConfig) {
        modelName = modelConfig.config?.name || modelConfig.name || modelName;
        const modelClass = modelConfig.class_name || 'Sequential';
        
        if (modelClass === 'Sequential') {
          const result = parseSequentialModel(modelConfig);
          layers = result.layers;
          connections = result.connections;
        } else {
          // Functional or Subclassed model
          const result = parseFunctionalModel(modelConfig);
          layers = result.layers;
          connections = result.connections;
        }
      } else {
        // No config - infer from weights
        warnings.push('No model config found, inferring structure from weights');
        
        const weightNames: string[] = [];
        
        // Recursively collect weight names
        function collectWeights(group: any, prefix: string = '') {
          const keys = group.keys();
          for (const key of keys) {
            const path = prefix ? `${prefix}/${key}` : key;
            const item = group.get(key);
            if (item && typeof item.keys === 'function') {
              collectWeights(item, path);
            } else {
              weightNames.push(path);
            }
          }
        }
        
        // Check common weight locations
        if (h5file.get('model_weights')) {
          collectWeights(h5file.get('model_weights'), '');
        } else {
          collectWeights(h5file, '');
        }
        
        const result = inferStructureFromWeights(weightNames);
        layers = result.layers;
        connections = result.connections;
      }
      
      // Try to count parameters from weight tensors
      try {
        function countParams(group: any) {
          let count = 0;
          const keys = group.keys();
          for (const key of keys) {
            const item = group.get(key);
            if (item) {
              if (item.shape) {
                // It's a dataset
                count += item.shape.reduce((a: number, b: number) => a * b, 1);
              } else if (typeof item.keys === 'function') {
                count += countParams(item);
              }
            }
          }
          return count;
        }
        
        if (h5file.get('model_weights')) {
          totalParams = countParams(h5file.get('model_weights'));
        } else {
          totalParams = countParams(h5file);
        }
      } catch {
        warnings.push('Could not count parameters');
      }
      
      h5file.close();
      
      // Clean up virtual filesystem
      try {
        if (h5wasm.FS) {
          h5wasm.FS.unlink(tempFileName);
        }
      } catch {
        // Ignore cleanup errors
      }
      
      // Build NN3D model
      const model: NN3DModel = {
        version: '1.0.0',
        metadata: {
          name: modelName,
          description: `Imported from Keras/TensorFlow HDF5 (${layers.length} layers)`,
          framework: 'keras',
          created: new Date().toISOString(),
          tags: ['keras', 'tensorflow', 'h5', 'imported'],
          totalParams,
          trainableParams: totalParams,
        },
        graph: {
          nodes: layers.map((layer, i) => ({
            id: layer.id,
            type: layer.type as LayerType,
            name: layer.name,
            params: layer.params as any,
            inputShape: layer.inputShape,
            outputShape: layer.outputShape,
            depth: i,
          })),
          edges: connections,
        },
        visualization: {
          layout: 'layered',
          theme: 'dark',
          layerSpacing: 2.5,
          nodeScale: 1.0,
          showLabels: true,
          showEdges: true,
          edgeStyle: 'bezier',
        },
      };
      
      return {
        success: true,
        model,
        warnings,
        format,
        inferredStructure: !modelConfig,
      };
      
    } catch (error) {
      console.error('HDF5 parse error:', error);
      return {
        success: false,
        error: `Failed to parse HDF5 file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        warnings,
        format,
        inferredStructure: false,
      };
    }
  }
};

export default KerasParser;
