import type { NN3DModel, NN3DNode, NN3DEdge } from '@/schema/types';
import { parseNN3DModel, validateModelSemantics } from '@/schema/validator';
import { 
  detectFormatFromExtension, 
  isSupportedExtension,
  getFormatDisplayName,
  SUPPORTED_EXTENSIONS,
} from './formats';
import { OnnxParser } from './formats/onnx-parser';
import { PyTorchParser } from './formats/pytorch-parser';
import { KerasParser } from './formats/keras-parser';
import { 
  isBackendAvailable, 
  analyzeUniversal,
  type ModelArchitecture,
  type LayerInfo 
} from './api-client';


// Instantiate parsers
const FORMAT_PARSERS = [
  OnnxParser,
  PyTorchParser,
  KerasParser,
];

/**
 * All supported file extensions
 */
export { SUPPORTED_EXTENSIONS };

/**
 * Track backend availability
 */
let backendAvailable: boolean | null = null;

/**
 * Check if backend is available (cached)
 */
async function checkBackend(): Promise<boolean> {
  if (backendAvailable === true) {
    return true;
  }

  const available = await isBackendAvailable();
  
  if (available && !backendAvailable) {
    console.log('[NN3D] Python backend available - using enhanced model analysis');
  } else if (!available && backendAvailable !== false) {
    console.log('[NN3D] Python backend unavailable - using JavaScript parsers');
  }
  
  backendAvailable = available;
  return backendAvailable;
}

/**
 * Convert backend layer type to NN3D type
 * Handles both PyTorch and Keras naming conventions
 */
function mapLayerType(layer: LayerInfo): string {
  const typeMap: Record<string, string> = {
    // PyTorch layers
    'Linear': 'linear',
    'Conv1d': 'conv1d',
    'Conv2d': 'conv2d',
    'Conv3d': 'conv3d',
    'BatchNorm1d': 'batchNorm1d',
    'BatchNorm2d': 'batchNorm2d',
    'BatchNorm3d': 'batchNorm3d',
    'LayerNorm': 'layerNorm',
    'GroupNorm': 'groupNorm',
    'ReLU': 'relu',
    'LeakyReLU': 'leakyRelu',
    'GELU': 'gelu',
    'Sigmoid': 'sigmoid',
    'Tanh': 'tanh',
    'Softmax': 'softmax',
    'Dropout': 'dropout',
    'MaxPool1d': 'maxPool1d',
    'MaxPool2d': 'maxPool2d',
    'AvgPool2d': 'avgPool2d',
    'AdaptiveAvgPool2d': 'adaptiveAvgPool',
    'LSTM': 'lstm',
    'GRU': 'gru',
    'RNN': 'rnn',
    'Embedding': 'embedding',
    'MultiheadAttention': 'multiHeadAttention',
    'Transformer': 'transformer',
    'Flatten': 'flatten',
    
    // Keras/TensorFlow layers
    'InputLayer': 'input',
    'Dense': 'dense',
    'Conv2D': 'conv2d',
    'Conv1D': 'conv1d',
    'Conv3D': 'conv3d',
    'MaxPooling2D': 'maxPool2d',
    'MaxPooling1D': 'maxPool1d',
    'AveragePooling2D': 'avgPool2d',
    'GlobalAveragePooling2D': 'globalAvgPool',
    'GlobalMaxPooling2D': 'maxPool2d',
    'BatchNormalization': 'batchNorm2d',
    'Activation': 'relu',
    'Add': 'add',
    'Concatenate': 'concat',
    'Multiply': 'multiply',
    'ZeroPadding2D': 'pad',
    'UpSampling2D': 'upsample',
    'Reshape': 'reshape',
    'Permute': 'reshape',
    'SeparableConv2D': 'separableConv2d',
    'DepthwiseConv2D': 'depthwiseConv2d',
    'Conv2DTranspose': 'convTranspose2d',
    'SimpleRNN': 'rnn',
    'Bidirectional': 'lstm',
    'TimeDistributed': 'custom',
    'Lambda': 'custom',
    'SpatialDropout2D': 'dropout',
    'AlphaDropout': 'dropout',
  };
  
  return typeMap[layer.type] || layer.type.toLowerCase().replace(/[0-9]d$/i, (m) => m.toLowerCase());
}

/**
 * Convert backend architecture to NN3DModel
 */
function architectureToNN3DModel(arch: ModelArchitecture): NN3DModel {
  const nodes: NN3DNode[] = arch.layers.map((layer, index) => {
    // Build params object from layer params with proper names
    const params: Record<string, unknown> = {};
    
    // Copy all layer params
    if (layer.params) {
      Object.entries(layer.params).forEach(([key, value]) => {
        // Map common param names to display-friendly names
        const keyMap: Record<string, string> = {
          'in_features': 'inFeatures',
          'out_features': 'outFeatures',
          'in_channels': 'inChannels',
          'out_channels': 'outChannels',
          'kernel_size': 'kernelSize',
          'hidden_size': 'hiddenSize',
          'input_size': 'inputSize',
          'num_layers': 'numLayers',
          'bidirectional': 'bidirectional',
          'batch_first': 'batchFirst',
          'dropout': 'dropout',
          'bias': 'bias',
        };
        const displayKey = keyMap[key] || key;
        params[displayKey] = value;
      });
    }
    
    // Add parameter count
    if (layer.numParameters > 0) {
      params.totalParams = layer.numParameters.toLocaleString();
    }
    
    // Build additional attributes - include category from backend!
    const attributes: Record<string, unknown> = { ...layer.params };
    if (layer.numParameters > 0) {
      attributes.parameters = layer.numParameters;
    }
    // Store the category from the backend so it can be used in visualization
    attributes.category = layer.category;
    
    return {
      id: layer.id,
      name: layer.name,
      type: mapLayerType(layer) as NN3DNode['type'],
      // Set inputShape and outputShape directly on the node
      inputShape: layer.inputShape || undefined,
      outputShape: layer.outputShape || undefined,
      params,
      attributes,
      position: {
        x: index * 3,
        y: 0,
        z: 0
      }
    };
  });
  
  const edges: NN3DEdge[] = arch.connections.map((conn, index) => ({
    id: `edge_${index}`,
    source: conn.source,
    target: conn.target,
    attributes: conn.tensorShape ? { tensorShape: conn.tensorShape } : undefined
  }));
  
  // Map framework string to valid type
  const frameworkMap: Record<string, 'pytorch' | 'tensorflow' | 'keras' | 'onnx' | 'jax' | 'custom'> = {
    'pytorch': 'pytorch',
    'tensorflow': 'tensorflow',
    'keras': 'keras',
    'onnx': 'onnx',
    'jax': 'jax',
  };
  const framework = frameworkMap[arch.framework] || 'custom';
  
  return {
    version: '1.0.0',
    metadata: {
      name: arch.name,
      description: `${arch.framework} model with ${arch.totalParameters.toLocaleString()} parameters (${arch.trainableParameters.toLocaleString()} trainable)`,
      framework,
      created: new Date().toISOString(),
      totalParams: arch.totalParameters,
      trainableParams: arch.trainableParameters,
      inputShape: arch.inputShape || undefined,
      outputShape: arch.outputShape || undefined,
    },
    graph: {
      nodes,
      edges
    },
    visualization: {
      layout: 'layered',
      layerSpacing: 2.5,
    }
  };
}

/**
 * All model extensions that can be analyzed by the universal backend endpoint
 */
const BACKEND_ONLY_EXTENSIONS = [
  '.pt', '.pth', '.ckpt',  // PyTorch
  '.h5', '.hdf5',          // Keras
  '.pb',                   // TensorFlow
];

const CLIENT_FIRST_EXTENSIONS = [
  '.onnx',          // Tier 1: Platinum Path
];

/**
 * Load model from file - auto-detects format with 4-Tier Pipeline
 */
export async function loadModelFromFile(file: File): Promise<NN3DModel> {
  // Check if extension is supported
  if (!isSupportedExtension(file.name)) {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    throw new Error(
      `Unsupported file format: ${ext}\n\n` +
      `Supported formats:\n${SUPPORTED_EXTENSIONS.join(', ')}`
    );
  }
  
  // Detect format
  const formatInfo = detectFormatFromExtension(file.name);
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  
  // =========================================================================
  // Tier 1: Client-Side Priority (ONNX)
  // =========================================================================
  
  if (CLIENT_FIRST_EXTENSIONS.includes(ext)) {
    console.log(`[Tier 1/2] Attempting client-side parsing for ${ext}...`);
    for (const parser of FORMAT_PARSERS) {
      if (await parser.canParse(file)) {
        try {
          const result = await parser.parse(file);
          if (result.success && result.model) {
            // Log success
            console.log(`[OK] Client-side parsing complete for ${ext}`);
            if (result.inferredStructure) {
               console.info('Note: Structure inferred from heuristics.');
            }
            return result.model;
          }
        } catch (err) {
          console.warn(`Client-side parser failed for ${ext}, trying backend fallback...`, err);
        }
      }
    }
  }
  
  // =========================================================================
  // Tier 3 & 4: Backend Required (.pt, .h5, etc) OR Fallback
  // =========================================================================
  
  // Check if we should use backend (either it's a backend-only format, or client-side failed)
  const shouldTryBackend = BACKEND_ONLY_EXTENSIONS.includes(ext) || CLIENT_FIRST_EXTENSIONS.includes(ext);
  
  if (shouldTryBackend) {
    const hasBackend = await checkBackend();
    
    if (hasBackend) {
      try {
        console.log(`[Tier 3/4] Analyzing ${ext} model with universal backend endpoint...`);
        const result = await analyzeUniversal(file);
        
        if (result.success) {
          console.log('[OK] Backend analysis complete:', result.model_type);
          console.log('[NN3D] Backend Response:', result);
          
          // Convert to NN3DModel
          const model = architectureToNN3DModel(result.architecture);
          
          // Attach backend messages/warnings to metadata for UI
          if (result.message) {
            model.metadata.description += `\n\n[Analysis Info]: ${result.message}`;
          }
          if (result.model_type === 'state_dict') {
             // Mark as weights-only for the layout engine to use Matrix/Grid view
             model.metadata.tags = [...(model.metadata.tags || []), 'weights-only'];
          }
          
          console.log('[NN3D] Converted Model:', model);
          return model;
        } 
      } catch (error) {
        console.warn('Backend analysis failed:', error);
        if (BACKEND_ONLY_EXTENSIONS.includes(ext)) {
            throw new Error(`Backend analysis failed for ${ext}: ${(error as Error).message}`);
        }
      }
    } else if (BACKEND_ONLY_EXTENSIONS.includes(ext)) {
        throw new Error(
            `The file ${file.name} requires the Python backend service to be running.\n` +
            `Please start the backend server to analyze ${ext} files.`
        );
    }
  }
  
  // =========================================================================
  // Final Fallback: Try remaining JS parsers (e.g. Keras if implemented client-side)
  // =========================================================================
  for (const parser of FORMAT_PARSERS) {
     // Skip if we already tried it in Tier 1/2 block
     if (CLIENT_FIRST_EXTENSIONS.includes(ext)) continue;

     if (await parser.canParse(file)) {
        const result = await parser.parse(file);
        if (result.success && result.model) return result.model;
     }
  }
  
  // Failure
  throw new Error(
    `Unable to parse ${getFormatDisplayName(formatInfo.category)} file.\n` +
    (formatInfo.conversionHint || 'Please convert to .nn3d or .onnx format.')
  );
}

/**
 * Load NN3D model from URL
 */
export async function loadModelFromUrl(url: string): Promise<NN3DModel> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  return parseModelFromString(text);
}

/**
 * Parse and validate model from JSON string
 */
export function parseModelFromString(jsonString: string): NN3DModel {
  const { model, validation } = parseNN3DModel(jsonString);
  
  if (!validation.valid || !model) {
    const errorMessages = validation.errors.map(e => `${e.path}: ${e.message}`).join('\n');
    throw new Error(`Model validation failed:\n${errorMessages}`);
  }
  
  // Additional semantic validation
  const semanticValidation = validateModelSemantics(model);
  if (!semanticValidation.valid) {
    const warnings = semanticValidation.errors.map(e => `${e.path}: ${e.message}`).join('\n');
    console.warn(`Model semantic warnings:\n${warnings}`);
  }
  
  return model;
}

/**
 * Export model to JSON string
 */
export function exportModelToString(model: NN3DModel, pretty = true): string {
  return JSON.stringify(model, null, pretty ? 2 : undefined);
}

/**
 * Download model as file
 */
export function downloadModel(model: NN3DModel, filename = 'model.nn3d'): void {
  const json = exportModelToString(model);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Create a simple file drop handler
 */
export function createFileDropHandler(
  element: HTMLElement,
  onFile: (file: File) => void,
  options: { accept?: string[]; onDragOver?: () => void; onDragLeave?: () => void } = {}
): () => void {
  const { accept = ['.nn3d', '.json'], onDragOver, onDragLeave } = options;
  
  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDragOver?.();
  };
  
  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDragLeave?.();
  };
  
  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDragLeave?.();
    
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      
      if (accept.includes(ext)) {
        onFile(file);
      } else {
        console.warn(`Unsupported file type: ${ext}`);
      }
    }
  };
  
  element.addEventListener('dragover', handleDragOver);
  element.addEventListener('dragleave', handleDragLeave);
  element.addEventListener('drop', handleDrop);
  
  // Return cleanup function
  return () => {
    element.removeEventListener('dragover', handleDragOver);
    element.removeEventListener('dragleave', handleDragLeave);
    element.removeEventListener('drop', handleDrop);
  };
}
