/**
 * API Client for NN3D Backend Service
 * Communicates with Python FastAPI server for model analysis
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface LayerInfo {
  id: string;
  name: string;
  type: string;
  category: string;
  inputShape: number[] | null;
  outputShape: number[] | null;
  params: Record<string, unknown>;
  numParameters: number;
  trainable: boolean;
}

export interface ConnectionInfo {
  source: string;
  target: string;
  tensorShape: number[] | null;
}

export interface ModelArchitecture {
  name: string;
  framework: string;
  totalParameters: number;
  trainableParameters: number;
  inputShape: number[] | null;
  outputShape: number[] | null;
  layers: LayerInfo[];
  connections: ConnectionInfo[];
}

export interface AnalysisResponse {
  success: boolean;
  model_type: string;
  architecture: ModelArchitecture;
  message: string | null;
}

export interface HealthResponse {
  status: string;
  pytorch_version: string;
  cuda_available: boolean;
}

/**
 * Check if the backend server is available
 */
export async function checkBackendHealth(): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Analyze a PyTorch model file using the backend service
 */
export async function analyzeModelWithBackend(
  file: File,
  inputShape?: number[]
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  let url = `${API_BASE_URL}/analyze`;
  if (inputShape && inputShape.length > 0) {
    url += `?input_shape=${inputShape.join(',')}`;
  }
  
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return await response.json();
}

/**
 * Analyze an ONNX model file using the backend service
 */
export async function analyzeONNXWithBackend(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/analyze/onnx`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return await response.json();
}

/**
 * Analyze any model file using the universal endpoint
 * Implements the 4-Tier Pipeline:
 * 1. Platinum Path: .nn3d, .onnx (Native/Zero-Latency)
 * 2. Gold Path: .safetensors, .h5 (Structure Inference)
 * 3. Silver Path: .pt (TorchScript/JIT - Backend Tracing)
 * 4. Bronze Path: .pt (State Dict - Stack Visualization)
 */
export async function analyzeUniversal(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  // Use the unified /upload endpoint which implements the Detect & Dispatch logic
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return await response.json();
}

/**
 * Check if backend is available and has required capabilities
 */
export async function isBackendAvailable(): Promise<boolean> {
  const health = await checkBackendHealth();
  return health !== null && health.status === 'healthy';
}

// =============================================================================
// Saved Models API
// =============================================================================

export interface SavedModelSummary {
  id: number;
  name: string;
  framework: string;
  total_parameters: number;
  layer_count: number;
  created_at: string;
}

export interface SavedModel extends SavedModelSummary {
  architecture: ModelArchitecture;
}

/**
 * Get list of all saved models
 */
export async function getSavedModels(): Promise<SavedModelSummary[]> {
  const response = await fetch(`${API_BASE_URL}/models/saved`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  const data = await response.json();
  return data.models || [];
}

/**
 * Check if a model with the given name already exists
 * Returns the model if found, null otherwise
 */
export async function findModelByName(name: string): Promise<SavedModelSummary | null> {
  try {
    const models = await getSavedModels();
    return models.find(m => m.name === name) || null;
  } catch {
    return null;
  }
}

/**
 * Get a saved model by ID with full architecture
 */
export async function getSavedModelById(id: number): Promise<SavedModel> {
  const response = await fetch(`${API_BASE_URL}/models/saved/${id}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  const data = await response.json();
  return data.model;
}

/**
 * Save a model to the database
 */
export async function saveModel(
  name: string,
  framework: string,
  totalParameters: number,
  layerCount: number,
  architecture: ModelArchitecture,
  fileHash?: string
): Promise<{ id: number; message: string }> {
  const response = await fetch(`${API_BASE_URL}/models/save`, {
    method: 'POST',
    headers: { 
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name,
      framework,
      totalParameters,
      layerCount,
      architecture,
      fileHash,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return await response.json();
}

/**
 * Delete a saved model
 */
export async function deleteSavedModel(id: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/models/saved/${id}`, {
    method: 'DELETE',
    headers: { 'Accept': 'application/json' },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
}
