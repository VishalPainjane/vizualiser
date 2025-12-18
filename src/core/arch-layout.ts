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
  }
): ArchitectureLayout {
  const blocks: LayerBlock[] = [];
  const connections: ArchitectureLayout['connections'] = [];
  
  // Build graph to analyze topology
  const graph = buildGraph(architecture.layers, architecture.connections);
  const isLinear = isLinearArchitecture(graph);
  
  // Track shapes for each node
  const shapeMap = new Map<string, TensorShape>();
  const defaultShape = parseTensorShape(architecture.inputShape) || { height: 224, width: 224, channels: 3 };
  
  // Track block counter per category for naming
  const blockCounter: Record<string, number> = {};
  
  // Track cumulative X position (since blocks have different widths)
  let currentX = 0;
  const positionXMap = new Map<number, number>(); // depth -> X position
  
  // Track bounds
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  
  // Process each layer
  architecture.layers.forEach((layer) => {
    const category = layer.category.toLowerCase();
    const type = layer.type.toLowerCase();
    const graphNode = graph.get(layer.id)!;
    
    // Parse shapes
    let inputShape = parseTensorShape(layer.inputShape);
    let outputShape = parseTensorShape(layer.outputShape);
    
    // Get parent shape if available
    let parentShape: TensorShape | null = null;
    if (graphNode.parents.length > 0) {
      parentShape = shapeMap.get(graphNode.parents[0]) || null;
    }
    
    // If shapes not provided, infer from params
    if (!inputShape) {
      inputShape = parentShape || { ...defaultShape };
    }
    if (!outputShape) {
      outputShape = inferShapeFromParams(layer, inputShape);
    }
    
    // Store output shape for children
    shapeMap.set(layer.id, outputShape);
    
    // Calculate 3D dimensions based on tensor shape
    // The block represents a 3D tensor: H × W × C
    // HEIGHT (Y-axis) = spatial frame height
    // WIDTH (X-axis) = channels (stacked frames, going left-to-right)
    // DEPTH (Z-axis) = layer thickness (thin slice)
    const spatialH = outputShape.height;
    const spatialW = outputShape.width;
    const channels = outputShape.channels;
    
    // Spatial size → Block Height (frame size, forms square with itself visually)
    const spatialScale = Math.sqrt(Math.max(spatialH, spatialW)) * LAYOUT_CONFIG.spatialScale * 5;
    let blockHeight = Math.min(
      LAYOUT_CONFIG.maxSpatialSize,
      Math.max(LAYOUT_CONFIG.minBlockSize, spatialScale)
    );
    
    // Channels → Block Width (stacked frames going along X-axis)
    // Cap channels at 512 for visual scaling to avoid huge blocks
    const cappedChannels = Math.min(channels, 512);
    let blockWidth = Math.min(
      LAYOUT_CONFIG.maxChannelSize,
      Math.max(LAYOUT_CONFIG.minBlockSize * 0.3, Math.sqrt(cappedChannels) * LAYOUT_CONFIG.channelScale * 5)
    );
    
    // Layer thickness → Block Depth (thin slice in Z)
    let blockDepth = blockHeight; // Same as height for square face when viewed from side
    
    // Special handling for Flatten layers - compact transition block
    if (type.includes('flatten')) {
      // Flatten is a transition - show as a thin vertical bar
      blockHeight = LAYOUT_CONFIG.minBlockSize * 2;
      blockWidth = LAYOUT_CONFIG.minBlockSize * 0.4;
      blockDepth = blockHeight;
    }
    // Make linear/dense/FC layers appear as vertical bars proportional to units
    else if (category === 'linear' || type.includes('dense') || type.includes('linear') || type.includes('fc')) {
      // Height scales with number of units (log scale to keep manageable)
      const units = Math.min(channels, 4096); // Cap for visualization
      blockHeight = Math.min(
        LAYOUT_CONFIG.maxSpatialSize,
        Math.max(LAYOUT_CONFIG.minBlockSize * 1.5, Math.log2(units + 1) * 0.4)
      );
      blockWidth = LAYOUT_CONFIG.minBlockSize * 0.5; // Thin width
      blockDepth = LAYOUT_CONFIG.minBlockSize * 0.5; // Thin depth - appears as vertical bar
    }
    // Pooling layers - inherit parent's width (channels don't change)
    else if (category === 'pooling') {
      const parentBlock = blocks.find(b => graphNode.parents.includes(b.id));
      if (parentBlock) {
        blockWidth = parentBlock.dimensions.width;
      }
    } 
    // Activation/norm layers - thinner version of parent
    else if (category === 'activation' || category === 'normalization') {
      const parentBlock = blocks.find(b => graphNode.parents.includes(b.id));
      if (parentBlock) {
        blockWidth = parentBlock.dimensions.width * 0.5;
      }
    }
    
    // Generate display name
    if (!blockCounter[category]) blockCounter[category] = 0;
    blockCounter[category]++;
    
    let displayName = layer.name;
    if (type.includes('conv')) {
      displayName = `Conv-${blockCounter[category]}`;
    } else if (type.includes('pool')) {
      displayName = `Pool`;
    } else if (type.includes('dense') || type.includes('linear')) {
      displayName = `FC-${blockCounter[category]}`;
    } else if (type.includes('flatten')) {
      displayName = 'Flatten';
    } else if (type.includes('add') || type.includes('concat')) {
      displayName = type.includes('add') ? '⊕ Add' : '⊕ Concat';
    } else if (type.includes('attention')) {
      displayName = 'Attention';
    }
    
    // Format dimension label
    let dimensionLabel = '';
    if (outputShape.height > 1) {
      dimensionLabel = `${outputShape.height}×${outputShape.width}×${outputShape.channels}`;
    } else {
      dimensionLabel = `${outputShape.channels}`;
    }
    
    // Calculate position based on graph topology
    // Use cumulative X position to account for varying block widths
    if (!positionXMap.has(graphNode.depth)) {
      positionXMap.set(graphNode.depth, currentX);
      currentX += blockWidth + LAYOUT_CONFIG.layerSpacing;
    }
    const posX = positionXMap.get(graphNode.depth)!;
    const posY = graphNode.branchIndex * LAYOUT_CONFIG.branchSpacing;
    const posZ = 0;
    
    // Create block
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
      position: {
        x: posX,
        y: posY,
        z: posZ,
      },
      dimensions: {
        width: blockWidth,
        height: blockHeight,
        depth: blockDepth,
      },
      color: getArchColor(category, layer.type),
      opacity: 1.0,
      params: layer.params,
      numParameters: layer.numParameters,
      label: displayName,
      dimensionLabel,
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
  
  // Build connections with skip connection detection
  architecture.connections.forEach(conn => {
    const fromBlock = blocks.find(b => b.id === conn.source);
    const toBlock = blocks.find(b => b.id === conn.target);
    
    if (fromBlock && toBlock) {
      const isSkip = isSkipConnection(conn.source, conn.target, graph);
      
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
  
  // Handle empty model
  if (blocks.length === 0) {
    minX = 0; maxX = 1;
    minY = -1; maxY = 1;
    minZ = -1; maxZ = 1;
  }
  
  // Calculate center
  const center = {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
    z: (minZ + maxZ) / 2,
  };
  
  return {
    blocks,
    connections,
    bounds: { minX, maxX, minY, maxY, minZ, maxZ },
    center,
    totalLayers: architecture.layers.length,
    modelName: architecture.name,
    isLinear,
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
