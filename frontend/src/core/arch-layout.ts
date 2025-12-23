/**
 * Architecture Layout Engine
 * 
 * Creates a VGG-style 3D visualization where:
 * - Block HEIGHT represents spatial dimension (H×W shrinks through network)
 * - Block DEPTH represents channel count (grows through network)
 * - Block WIDTH is thin (represents a single layer or group)
 * - Position flows left-to-right (X axis)
 * - Non-linear architectures spread vertically (Y axis) for branches
 * 
 * This creates the classic "funnel" effect seen in CNN architecture diagrams,
 * while also supporting ResNets, U-Nets, and other branching architectures.
 */

// ============================================================================
// Types
// ============================================================================

export interface TensorShape {
  height: number;      // Spatial height (e.g., 224, 112, 56...)
  width: number;       // Spatial width (usually same as height for images)
  channels: number;    // Feature channels (e.g., 64, 128, 256, 512...)
}

export interface LayerBlock {
  id: string;
  name: string;
  displayName: string;
  type: string;
  category: string;
  
  // Graph info
  depth: number;       // Graph depth (for X positioning)
  branchIndex: number; // Branch index at this depth (for Y offset)
  
  // Tensor dimensions
  inputShape: TensorShape | null;
  outputShape: TensorShape | null;
  
  // Computed 3D dimensions for visualization
  position: { x: number; y: number; z: number };
  dimensions: { width: number; height: number; depth: number };
  
  // Visual properties
  color: string;
  opacity: number;
  
  // Layer params for display
  params: Record<string, unknown>;
  numParameters: number;
  
  // Label info
  label: string;
  dimensionLabel: string;
}

export interface ArchitectureLayout {
  blocks: LayerBlock[];
  connections: Array<{
    from: string;
    to: string;
    fromPos: { x: number; y: number; z: number };
    toPos: { x: number; y: number; z: number };
    isSkipConnection: boolean;
  }>;
  bounds: {
    minX: number; maxX: number;
    minY: number; maxY: number;
    minZ: number; maxZ: number;
  };
  center: { x: number; y: number; z: number };
  totalLayers: number;
  modelName: string;
  isLinear: boolean;
  totalParameters: number;
  cameraSuggestion?: {
    position: { x: number; y: number; z: number };
    target: { x: number; y: number; z: number };
  };
}

// ============================================================================
// Color Scheme (professional, distinct but muted colors)
// ============================================================================

const ARCH_COLORS: Record<string, string> = {
  // Core computation layers - blues
  convolution: '#5B8BD9',    // Soft Blue (Conv layers)
  conv2d: '#5B8BD9',         // Soft Blue
  conv1d: '#7BA3E0',         // Light Blue
  
  // Fully connected - greens
  linear: '#6BAF6B',         // Soft Green (Fully Connected)
  dense: '#6BAF6B',          // Soft Green
  fc: '#6BAF6B',             // Soft Green
  
  // Spatial reduction - coral/salmon
  pooling: '#E07070',        // Coral Red (Max/Avg Pooling)
  maxpool: '#E07070',        // Coral Red
  avgpool: '#E08850',        // Soft Orange
  
  // Activations - warm amber
  activation: '#D9A740',     // Soft Amber
  relu: '#D9A740',           // Soft Amber
  sigmoid: '#C99030',        // Deeper Amber
  softmax: '#E0A050',        // Light Amber
  
  // Normalization - teals
  normalization: '#50A8A0',  // Soft Teal
  batchnorm: '#50A8A0',      // Soft Teal
  layernorm: '#60B8B0',      // Light Teal
  
  // Regularization - slate
  regularization: '#708090', // Slate Gray
  dropout: '#708090',        // Slate Gray
  
  // Special layers - purple
  attention: '#9070C0',      // Soft Purple
  multiheadattention: '#9070C0',
  
  // Embeddings - cyan
  embedding: '#50A8C0',      // Soft Cyan
  
  // Recurrent - rose
  recurrent: '#C070A0',      // Soft Rose
  lstm: '#C070A0',           // Soft Rose
  gru: '#D080B0',            // Light Rose
  
  // Reshaping - cool gray
  reshape: '#808890',        // Cool Gray
  flatten: '#808890',        // Cool Gray
  
  // Input/Output
  input: '#50B080',          // Soft Emerald
  output: '#D06070',         // Soft Rose
  
  // Padding/Other
  padding: '#8070B0',        // Soft Violet
  concat: '#A060C0',         // Soft Fuchsia
  add: '#A060C0',            // Soft Fuchsia (for residual adds)
  
  // Default
  other: '#909090',          // Neutral Gray
};

export function getArchColor(category: string, type?: string): string {
  const lowerCategory = category.toLowerCase();
  const lowerType = (type || '').toLowerCase();
  
  // First try to match specific type
  if (lowerType && ARCH_COLORS[lowerType]) {
    return ARCH_COLORS[lowerType];
  }
  
  // Check type for partial matches
  if (lowerType) {
    if (lowerType.includes('conv')) return ARCH_COLORS.convolution;
    if (lowerType.includes('pool')) return ARCH_COLORS.pooling;
    if (lowerType.includes('relu')) return ARCH_COLORS.relu;
    if (lowerType.includes('norm')) return ARCH_COLORS.normalization;
    if (lowerType.includes('drop')) return ARCH_COLORS.dropout;
    if (lowerType.includes('attention')) return ARCH_COLORS.attention;
    if (lowerType.includes('embed')) return ARCH_COLORS.embedding;
    if (lowerType.includes('lstm') || lowerType.includes('gru')) return ARCH_COLORS.recurrent;
    if (lowerType.includes('flatten')) return ARCH_COLORS.flatten;
    if (lowerType.includes('dense') || lowerType.includes('linear') || lowerType.includes('fc')) return ARCH_COLORS.linear;
  }
  
  // Fall back to category
  return ARCH_COLORS[lowerCategory] || ARCH_COLORS.other;
}

// ============================================================================
// Dimension Parsing
// ============================================================================

/**
 * Parse tensor shape from various formats
 */
export function parseTensorShape(shape: number[] | null | undefined): TensorShape | null {
  if (!shape || shape.length === 0) return null;
  
  // Handle different shape formats:
  // [B, C, H, W] - PyTorch conv (batch, channels, height, width)
  // [B, H, W, C] - TensorFlow/Keras (batch, height, width, channels)
  // [B, features] - Linear layers
  // [B, seq, features] - Sequence models
  
  if (shape.length === 4) {
    // Assume PyTorch format [B, C, H, W]
    const [_b, c, h, w] = shape;
    return { height: h, width: w, channels: c };
  } else if (shape.length === 3) {
    // Could be [B, H, W] or [B, seq, features]
    const [_b, dim1, dim2] = shape;
    return { height: dim1, width: dim2, channels: 1 };
  } else if (shape.length === 2) {
    // [B, features] - treat as 1×1×features
    return { height: 1, width: 1, channels: shape[1] };
  } else if (shape.length === 1) {
    return { height: 1, width: 1, channels: shape[0] };
  }
  
  return null;
}

/**
 * Infer shape from layer parameters
 */
export function inferShapeFromParams(
  layer: { type: string; category: string; params: Record<string, unknown> },
  prevShape: TensorShape | null
): TensorShape {
  const params = layer.params || {};
  
  // Extract common parameters
  const filters = params.filters as number || params.out_channels as number || params.outChannels as number;
  const units = params.units as number || params.out_features as number || params.outFeatures as number;
  // Note: kernel_size and padding affect output dimensions but we use simplified calculation
  const strides = params.strides || params.stride;
  
  // Start with previous shape or default
  let height = prevShape?.height || 224;
  let width = prevShape?.width || 224;
  let channels = prevShape?.channels || 3;
  
  const category = layer.category.toLowerCase();
  const type = layer.type.toLowerCase();
  
  // Handle pooling - reduces spatial dimensions
  if (category === 'pooling' || type.includes('pool')) {
    const poolStride = Array.isArray(strides) ? strides[0] : (strides as number) || 2;
    height = Math.floor(height / poolStride);
    width = Math.floor(width / poolStride);
    // Channels stay the same for pooling
  }
  // Handle convolution
  else if (category === 'convolution' || type.includes('conv')) {
    if (filters) channels = filters;
    // Check if stride reduces size
    const convStride = Array.isArray(strides) ? strides[0] : (strides as number) || 1;
    if (convStride > 1) {
      height = Math.floor(height / convStride);
      width = Math.floor(width / convStride);
    }
  }
  // Handle linear/dense - flattens to 1×1×features
  else if (category === 'linear' || type.includes('dense') || type.includes('linear')) {
    height = 1;
    width = 1;
    if (units) channels = units;
  }
  // Handle flatten
  else if (type.includes('flatten')) {
    const totalFeatures = height * width * channels;
    height = 1;
    width = 1;
    channels = totalFeatures;
  }
  // Handle reshape
  else if (category === 'reshape') {
    // Keep previous or use output shape if available
  }
  
  return { height, width, channels };
}

/**
 * Calculate dimensions based on weight parameters for Bronze Path
 */
function calculateWeightDimensions(layer: any): { width: number; height: number; depth: number } {
  const params = layer.params || {};
  const category = layer.category.toLowerCase();
  
  // Default sizes
  let height = LAYOUT_CONFIG.defaultBlockHeight;
  let width = LAYOUT_CONFIG.defaultBlockWidth;
  let depth = LAYOUT_CONFIG.defaultBlockDepth;
  
  // Scaling factor for visualization
  const logScale = (val: number) => Math.log2(val + 1);

  if (category === 'convolution') {
    // Conv: [Out, In, kH, kW]
    // Height ~ Out Channels
    // Depth ~ In Channels
    // Width ~ Kernel Size
    const outCh = params.out_channels || params.outChannels || params.filters;
    const inCh = params.in_channels || params.inChannels;
    const kSize = params.kernel_size || params.kernelSize;
    
    if (outCh) height = Math.max(1, logScale(outCh));
    if (inCh) depth = Math.max(1, logScale(inCh));
    if (kSize) {
      const kVal = Array.isArray(kSize) ? kSize[0] : kSize;
      width = Math.max(0.5, kVal * 0.5); 
    }
  } else if (category === 'linear' || layer.type.toLowerCase().includes('linear')) {
    // Linear: [Out, In]
    // Height ~ Out Features
    // Depth ~ In Features
    // Width ~ Thin
    const outFeat = params.out_features || params.outFeatures || params.units;
    const inFeat = params.in_features || params.inFeatures;
    
    if (outFeat) height = Math.max(1, logScale(outFeat));
    if (inFeat) depth = Math.max(1, logScale(inFeat));
    width = 0.4; // Thin plate
  } else if (category === 'embedding') {
    // Embedding: [NumEmbed, Dim]
    // Height ~ Num Embeddings
    // Depth ~ Embedding Dim
    const num = params.num_embeddings || params.numEmbeddings || params.input_dim;
    const dim = params.embedding_dim || params.embeddingDim || params.output_dim;
    
    if (num) height = Math.max(1, logScale(num));
    if (dim) depth = Math.max(1, logScale(dim));
    width = 0.8;
  } else if (category === 'normalization') {
    // BatchNorm: [NumFeatures]
    // Height ~ Num Features
    // Depth ~ Num Features (Same)
    const num = params.num_features || params.numFeatures;
    if (num) {
        const s = Math.max(1, logScale(num));
        height = s;
        depth = s;
    }
    width = 0.2; // Thin slice
  }

  return { width, height, depth };
}

// ============================================================================
// Layout Calculation
// ============================================================================

const LAYOUT_CONFIG = {
  // Scaling factors for 3D dimensions
  spatialScale: 0.05,      // How much to scale spatial dimensions (large)
  channelScale: 0.008,     // How much to scale channel dimension
  layerThickness: 0.5,     // Base thickness of each layer block
  
  // Spacing
  layerSpacing: 1.8,       // Gap between layers (X) - wide for clear labels
  branchSpacing: 3.0,      // Gap between parallel branches (Y)
  groupSpacing: 2.0,       // Extra gap between groups (conv blocks)
  
  // Size limits
  minBlockSize: 0.3,       // Minimum visible size
  maxSpatialSize: 8.0,     // Max spatial size for large feature maps
  maxChannelSize: 6.0,     // Max channel size for deep networks (increased)
  
  // Default sizes for shape-less (Bronze Path) layouts
  defaultBlockHeight: 2.0,
  defaultBlockWidth: 2.0,
  defaultBlockDepth: 2.0,
  ghostBlockHeight: 1.5,
  ghostBlockWidth: 1.5,
  ghostBlockDepth: 1.5,

  // Pooling layer thickness (thinner)
  poolingThickness: 0.3,
  activationThickness: 0.15,
};

// ============================================================================
// Graph Analysis for Non-Linear Architectures
// ============================================================================

interface GraphNode {
  id: string;
  depth: number;
  branchIndex: number;
  parents: string[];
  children: string[];
}

/**
 * Build a graph from layers and connections, compute depth for each node
 */
function buildGraph(
  layers: Array<{ id: string }>,
  connections: Array<{ source: string; target: string }>
): Map<string, GraphNode> {
  const graph = new Map<string, GraphNode>();
  
  // Initialize nodes
  layers.forEach(layer => {
    graph.set(layer.id, {
      id: layer.id,
      depth: 0,
      branchIndex: 0,
      parents: [],
      children: [],
    });
  });
  
  // Build adjacency
  connections.forEach(conn => {
    const parent = graph.get(conn.source);
    const child = graph.get(conn.target);
    if (parent && child) {
      parent.children.push(conn.target);
      child.parents.push(conn.source);
    }
  });
  
  // Find root nodes (no parents)
  const roots = Array.from(graph.values()).filter(n => n.parents.length === 0);
  
  // BFS to compute depths
  const queue: string[] = roots.map(r => r.id);
  const visited = new Set<string>();
  
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    
    const node = graph.get(nodeId)!;
    
    // Depth is max parent depth + 1
    if (node.parents.length > 0) {
      const maxParentDepth = Math.max(
        ...node.parents.map(p => graph.get(p)?.depth || 0)
      );
      node.depth = maxParentDepth + 1;
    }
    
    // Add children to queue
    node.children.forEach(childId => {
      if (!visited.has(childId)) {
        queue.push(childId);
      }
    });
  }
  
  // Handle disconnected nodes (not in any connection)
  let currentDepth = 0;
  layers.forEach(layer => {
    const node = graph.get(layer.id)!;
    if (!visited.has(layer.id)) {
      node.depth = currentDepth++;
      visited.add(layer.id);
    }
  });
  
  // Assign branch indices for nodes at same depth
  const depthGroups = new Map<number, string[]>();
  graph.forEach(node => {
    const group = depthGroups.get(node.depth) || [];
    group.push(node.id);
    depthGroups.set(node.depth, group);
  });
  
  depthGroups.forEach(nodeIds => {
    nodeIds.forEach((id, index) => {
      const node = graph.get(id)!;
      // Center branches around 0
      node.branchIndex = index - (nodeIds.length - 1) / 2;
    });
  });
  
  return graph;
}

/**
 * Check if architecture is linear (sequential)
 */
function isLinearArchitecture(graph: Map<string, GraphNode>): boolean {
  for (const node of graph.values()) {
    if (node.parents.length > 1 || node.children.length > 1) {
      return false;
    }
  }
  return true;
}

/**
 * Detect skip connections (edges that skip depths)
 */
function isSkipConnection(
  fromId: string, 
  toId: string, 
  graph: Map<string, GraphNode>
): boolean {
  const from = graph.get(fromId);
  const to = graph.get(toId);
  if (!from || !to) return false;
  return Math.abs(to.depth - from.depth) > 1;
}


/**
 * Compute 3D layout for architecture visualization
 * Handles both linear (sequential) and non-linear (branching) architectures
 */
export function computeArchitectureLayout(
  architecture: {
    name: string;
    framework: string;
    tags?: string[];
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
      attributes?: Record<string, unknown>;
    }>;
  }
): ArchitectureLayout {
  const blocks: LayerBlock[] = [];
  const connections: ArchitectureLayout['connections'] = [];
  
  // Bronze Path Check: If explicit 'weights-only' tag exists OR missing shape data
  const isBronzePath = architecture.tags?.includes('weights-only') || (
    architecture.layers.length > 0 && 
    architecture.layers.slice(0, 3).every(l => !l.outputShape && !l.inputShape)
  );
    
  if (isBronzePath) {
    console.log('[Layout] Bronze Path detected: Using simple sequential layout due to missing shape info.');
  }

  // Track bounds
  let minX = 0, maxX = 0;
  let minY = 0, maxY = 0;
  let minZ = 0, maxZ = 0;
  
  // --- BRONZE PATH: Matrix/Grid Layout (Weights Only) ---
  if (isBronzePath) {
    console.log('[Layout] Generating Matrix Layout for Weights Only (Bronze Path)');
    
    // Grid configuration
    const cols = Math.ceil(Math.sqrt(architecture.layers.length));
    const spacing = 2.5;
    
    architecture.layers.forEach((layer, index) => {
      const isGhost = layer.params?.ghost === true;
      const category = layer.category.toLowerCase();

      // Calculate dimensions from weights if available, else use defaults
      let { width: blockWidth, height: blockHeight, depth: blockDepth } = calculateWeightDimensions(layer);
      
      if (isGhost) {
          blockWidth *= 0.7;
          blockHeight *= 0.7;
          blockDepth *= 0.7;
      }

      // Calculate grid position
      const row = Math.floor(index / cols);
      const col = index % cols;
      
      const posX = (col - cols / 2) * spacing;
      const posZ = (row - cols / 2) * spacing; // Spread on Z (ground plane)
      const posY = 0;
      
      const block: LayerBlock = {
        id: layer.id,
        name: layer.name,
        displayName: layer.name.split('.').pop() || layer.name,
        type: layer.type,
        category,
        depth: row, 
        branchIndex: col,
        inputShape: null,
        outputShape: null,
        position: { x: posX, y: posY, z: posZ },
        dimensions: { width: blockWidth, height: blockHeight, depth: blockDepth },
        color: getArchColor(category, layer.type),
        opacity: isGhost ? 0.6 : 1.0,
        params: layer.params,
        numParameters: layer.numParameters,
        label: layer.name.split('.').pop() || layer.name,
        dimensionLabel: '',
      };
      blocks.push(block);
      
      // Update bounds
      minX = Math.min(minX, posX - blockWidth / 2);
      maxX = Math.max(maxX, posX + blockWidth / 2);
      minY = Math.min(minY, posY - blockHeight / 2);
      maxY = Math.max(maxY, posY + blockHeight / 2);
      minZ = Math.min(minZ, posZ - blockDepth / 2);
      maxZ = Math.max(maxZ, posZ + blockDepth / 2);
    });

  } else {
    // --- SILVER/GOLD/PLATINUM PATH: Shape-based Layout ---
    const graph = buildGraph(architecture.layers, architecture.connections);
    const shapeMap = new Map<string, TensorShape>();
    const defaultShape = parseTensorShape(architecture.inputShape) || { height: 224, width: 224, channels: 3 };
    const blockCounter: Record<string, number> = {};
    let currentX = 0;
    const positionXMap = new Map<number, number>();

    minX = Infinity; maxX = -Infinity;
    minY = Infinity; maxY = -Infinity;
    minZ = Infinity; maxZ = -Infinity;

    architecture.layers.forEach((layer) => {
      const category = layer.category.toLowerCase();
      const type = layer.type.toLowerCase();
      const graphNode = graph.get(layer.id)!;
      
      let inputShape = parseTensorShape(layer.inputShape);
      let outputShape = parseTensorShape(layer.outputShape);
      
      let parentShape: TensorShape | null = null;
      if (graphNode.parents.length > 0) {
        parentShape = shapeMap.get(graphNode.parents[0]) || null;
      }
      
      if (!inputShape) inputShape = parentShape || { ...defaultShape };
      if (!outputShape) outputShape = inferShapeFromParams(layer, inputShape);
      
      shapeMap.set(layer.id, outputShape);

      const spatialH = outputShape.height;
      const spatialW = outputShape.width;
      const channels = outputShape.channels;
      
      const spatialScale = Math.sqrt(Math.max(spatialH, spatialW)) * LAYOUT_CONFIG.spatialScale * 5;
      let blockHeight = Math.min(LAYOUT_CONFIG.maxSpatialSize, Math.max(LAYOUT_CONFIG.minBlockSize, spatialScale));
      
      const cappedChannels = Math.min(channels, 512);
      let blockWidth = Math.min(LAYOUT_CONFIG.maxChannelSize, Math.max(LAYOUT_CONFIG.minBlockSize * 0.3, Math.sqrt(cappedChannels) * LAYOUT_CONFIG.channelScale * 5));
      
      let blockDepth = blockHeight;

      if (type.includes('flatten')) {
        blockHeight = LAYOUT_CONFIG.minBlockSize * 2;
        blockWidth = LAYOUT_CONFIG.minBlockSize * 0.4;
        blockDepth = blockHeight;
      } else if (category === 'linear' || type.includes('dense') || type.includes('fc')) {
        const units = Math.min(channels, 4096);
        blockHeight = Math.min(LAYOUT_CONFIG.maxSpatialSize, Math.max(LAYOUT_CONFIG.minBlockSize * 1.5, Math.log2(units + 1) * 0.4));
        blockWidth = LAYOUT_CONFIG.minBlockSize * 0.5;
        blockDepth = LAYOUT_CONFIG.minBlockSize * 0.5;
      }

      if (!blockCounter[category]) blockCounter[category] = 0;
      blockCounter[category]++;
      
      let displayName = layer.name;
      // Simplified display name logic
      displayName = layer.name.split('.').pop() || layer.name;

      let dimensionLabel = '';
      if (outputShape.height > 1) {
          dimensionLabel = `${outputShape.height}×${outputShape.width}×${outputShape.channels}`;
      } else {
          dimensionLabel = `${outputShape.channels}`;
      }

      if (!positionXMap.has(graphNode.depth)) {
        positionXMap.set(graphNode.depth, currentX);
        currentX += blockWidth + LAYOUT_CONFIG.layerSpacing;
      }
      const posX = positionXMap.get(graphNode.depth)!;
      const posY = graphNode.branchIndex * LAYOUT_CONFIG.branchSpacing;
      const posZ = 0;
      
      const block: LayerBlock = {
        id: layer.id,
        name: layer.name,
        displayName,
        type: layer.type,
        category,
        depth: graphNode.depth,
        branchIndex: graphNode.branchIndex,
        inputShape,
        outputShape,
        position: { x: posX, y: posY, z: posZ },
        dimensions: { width: blockWidth, height: blockHeight, depth: blockDepth },
        color: getArchColor(category, layer.type),
        opacity: 1.0,
        params: layer.params,
        numParameters: layer.numParameters,
        label: displayName,
        dimensionLabel,
      };
      blocks.push(block);
      
      minX = Math.min(minX, posX - blockWidth / 2);
      maxX = Math.max(maxX, posX + blockWidth / 2);
      minY = Math.min(minY, posY - blockHeight / 2);
      maxY = Math.max(maxY, posY + blockHeight / 2);
      minZ = Math.min(minZ, posZ - blockDepth / 2);
      maxZ = Math.max(maxZ, posZ + blockDepth / 2);
    });
  }

  // Build connections for all paths
  architecture.connections.forEach(conn => {
    const fromBlock = blocks.find(b => b.id === conn.source);
    const toBlock = blocks.find(b => b.id === conn.target);
    
    if (fromBlock && toBlock) {
      const isImplicit = conn.attributes?.implicit === true;
      const graph = buildGraph(architecture.layers, architecture.connections); // Rebuild or pass graph
      const isSkip = isImplicit || isSkipConnection(conn.source, conn.target, graph);
      
      connections.push({
        from: conn.source,
        to: conn.target,
        fromPos: {
          x: fromBlock.position.x + fromBlock.dimensions.width / 2,
          y: fromBlock.position.y,
          z: fromBlock.position.z,
        },
        toPos: {
          x: toBlock.position.x - toBlock.dimensions.width / 2,
          y: toBlock.position.y,
          z: toBlock.position.z,
        },
        isSkipConnection: isSkip,
      });
    }
  });

  const center = {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
    z: (minZ + maxZ) / 2,
  };

  // Calculate camera distance based on bounds size
  const sizeX = maxX - minX;
  const sizeY = maxY - minY;
  const sizeZ = maxZ - minZ;
  const maxDim = Math.max(sizeX, sizeY, sizeZ);
  
  // Basic heuristic: distance ~ maxDim
  // Ensure a minimum distance so we don't start inside a small model
  const cameraDistance = Math.max(20, maxDim * 0.8 + 10);
  
  const cameraSuggestion = {
    position: { 
      x: isBronzePath ? 0 : 0, 
      y: isBronzePath ? cameraDistance * 0.8 : cameraDistance * 0.2, 
      z: isBronzePath ? cameraDistance * 0.6 : cameraDistance 
    },
    target: { x: 0, y: 0, z: 0 } // Since we center the scene at 0,0,0
  };
  
  return {
    blocks,
    connections,
    bounds: { minX, maxX, minY, maxY, minZ, maxZ },
    center,
    totalLayers: architecture.layers.length,
    modelName: architecture.name,
    isLinear: isBronzePath || isLinearArchitecture(buildGraph(architecture.layers, architecture.connections)),
    totalParameters: architecture.totalParameters,
    cameraSuggestion,
  };
}


/**
 * Group consecutive layers of same category into "stages"
 * (e.g., Conv+ReLU+Conv+ReLU → "Conv Block 1")
 */
export function groupLayersIntoStages(
  layout: ArchitectureLayout
): ArchitectureLayout {
  // For now, return as-is. Can implement grouping later.
  return layout;
}
