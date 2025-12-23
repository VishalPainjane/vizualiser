/**
 * Hierarchical Model Parser
 * 
 * Transforms flat layer list into hierarchical structure for 3-level visualization:
 * Level 1: Macro (Encoder → Decoder → Head)
 * Level 2: Stage/Block (layer1, layer2, etc.)
 * Level 3: Individual layers
 */

export interface LayerData {
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

export interface ConnectionData {
  source: string;
  target: string;
  tensorShape: number[] | null;
}

export interface HierarchyNode {
  id: string;
  name: string;
  displayName: string;
  level: 1 | 2 | 3;
  type: 'group' | 'layer';
  category: string;
  
  // Aggregated stats
  layerCount: number;
  totalParams: number;
  inputShape: number[] | null;
  outputShape: number[] | null;
  
  // Hierarchy
  children: HierarchyNode[];
  parent: string | null;
  
  // For level 3 (actual layers)
  layerData?: LayerData;
  
  // Visual properties
  depth: number; // X position factor
  channelSize: number; // Z size factor
  blockIndex: number; // Y position factor
}

export interface HierarchyConnection {
  id: string;
  source: string;
  target: string;
  level: 1 | 2 | 3;
  tensorShape: number[] | null;
  isSkipConnection: boolean;
  isDownsample: boolean;
}

export interface ModelHierarchy {
  name: string;
  framework: string;
  totalParameters: number;
  inputShape: number[] | null;
  outputShape: number[] | null;
  
  // Hierarchical structure
  macroNodes: HierarchyNode[]; // Level 1
  allNodes: Map<string, HierarchyNode>;
  connections: HierarchyConnection[];
  
  // Flat access
  layers: LayerData[];
}

/**
 * Parse layer name to extract hierarchy information
 */
function parseLayerName(name: string): { 
  macro: string; 
  stage: string | null;
  block: string | null;
  component: string;
} {
  const parts = name.split('.');
  
  // Extract macro level (encoder, decoder, segmentation_head, etc.)
  const macro = parts[0];
  
  // Check for stage patterns like "layer1", "layer2", "blocks"
  let stage: string | null = null;
  let block: string | null = null;
  let componentStart = 1;
  
  if (parts.length > 1) {
    // Check for "layerX" or "blocks" pattern
    if (/^layer\d+$/.test(parts[1]) || parts[1] === 'blocks') {
      stage = parts[1];
      componentStart = 2;
      
      // Check for block index (e.g., "0", "1", "2")
      if (parts.length > 2 && /^\d+$/.test(parts[2])) {
        block = parts[2];
        componentStart = 3;
      }
    }
  }
  
  const component = parts.slice(componentStart).join('.') || parts[parts.length - 1];
  
  return { macro, stage, block, component };
}

/**
 * Infer channel size from layer data
 */
function inferChannelSize(layer: LayerData): number {
  // Try output channels first
  if (layer.outputShape && layer.outputShape.length >= 2) {
    return layer.outputShape[1]; // Channels dimension
  }
  
  // Try from params
  if (layer.params) {
    if (typeof layer.params.out_channels === 'number') return layer.params.out_channels;
    if (typeof layer.params.out_features === 'number') return layer.params.out_features;
    if (typeof layer.params.num_features === 'number') return layer.params.num_features;
  }
  
  return 64; // Default
}

/**
 * Detect if connection is a skip/residual connection
 */
function isSkipConnection(conn: ConnectionData, layers: Map<string, LayerData>): boolean {
  const source = layers.get(conn.source);
  const target = layers.get(conn.target);
  
  if (!source || !target) return false;
  
  // Check if name suggests skip connection
  if (source.name.includes('downsample') || target.name.includes('downsample')) {
    return true;
  }
  
  // Check if shapes match but there are intermediate layers
  const sourceIdx = parseInt(conn.source.split('_')[1] || '0');
  const targetIdx = parseInt(conn.target.split('_')[1] || '0');
  
  return targetIdx - sourceIdx > 2; // Skip if more than 2 layers apart
}

/**
 * Build hierarchical model structure from flat layer list
 */
export function buildModelHierarchy(
  architecture: {
    name: string;
    framework: string;
    totalParameters: number;
    inputShape?: number[] | null;
    outputShape?: number[] | null;
    layers: LayerData[];
    connections: ConnectionData[];
  }
): ModelHierarchy {
  const allNodes = new Map<string, HierarchyNode>();
  const macroNodes: HierarchyNode[] = [];
  const layerMap = new Map<string, LayerData>();
  
  // Build layer map for quick lookup
  architecture.layers.forEach(layer => {
    layerMap.set(layer.id, layer);
  });
  
  // Group layers by hierarchy
  const macroGroups = new Map<string, {
    layers: LayerData[];
    stages: Map<string, {
      layers: LayerData[];
      blocks: Map<string, LayerData[]>;
    }>;
  }>();
  
  // First pass: Group layers
  architecture.layers.forEach((layer) => {
    const { macro, stage, block } = parseLayerName(layer.name);
    
    if (!macroGroups.has(macro)) {
      macroGroups.set(macro, { layers: [], stages: new Map() });
    }
    
    const macroGroup = macroGroups.get(macro)!;
    macroGroup.layers.push(layer);
    
    if (stage) {
      if (!macroGroup.stages.has(stage)) {
        macroGroup.stages.set(stage, { layers: [], blocks: new Map() });
      }
      
      const stageGroup = macroGroup.stages.get(stage)!;
      stageGroup.layers.push(layer);
      
      if (block) {
        if (!stageGroup.blocks.has(block)) {
          stageGroup.blocks.set(block, []);
        }
        stageGroup.blocks.get(block)!.push(layer);
      }
    }
  });
  
  // Second pass: Build hierarchy nodes
  let macroDepth = 0;
  
  macroGroups.forEach((macroGroup, macroName) => {
    const macroId = `macro_${macroName}`;
    
    // Calculate macro stats
    const totalParams = macroGroup.layers.reduce((sum, l) => sum + l.numParameters, 0);
    const firstLayer = macroGroup.layers[0];
    const lastLayer = macroGroup.layers[macroGroup.layers.length - 1];
    const maxChannels = Math.max(...macroGroup.layers.map(inferChannelSize));
    
    const macroNode: HierarchyNode = {
      id: macroId,
      name: macroName,
      displayName: formatDisplayName(macroName),
      level: 1,
      type: 'group',
      category: inferMacroCategory(macroName),
      layerCount: macroGroup.layers.length,
      totalParams,
      inputShape: firstLayer?.inputShape || null,
      outputShape: lastLayer?.outputShape || null,
      children: [],
      parent: null,
      depth: macroDepth,
      channelSize: maxChannels,
      blockIndex: 0,
    };
    
    // Build stage nodes (Level 2)
    let stageIndex = 0;
    if (macroGroup.stages.size > 0) {
      macroGroup.stages.forEach((stageGroup, stageName) => {
        const stageId = `stage_${macroName}_${stageName}`;
        const stageParams = stageGroup.layers.reduce((sum, l) => sum + l.numParameters, 0);
        const stageFirst = stageGroup.layers[0];
        const stageLast = stageGroup.layers[stageGroup.layers.length - 1];
        const stageMaxChannels = Math.max(...stageGroup.layers.map(inferChannelSize));
        
        const stageNode: HierarchyNode = {
          id: stageId,
          name: stageName,
          displayName: formatDisplayName(stageName),
          level: 2,
          type: 'group',
          category: 'stage',
          layerCount: stageGroup.layers.length,
          totalParams: stageParams,
          inputShape: stageFirst?.inputShape || null,
          outputShape: stageLast?.outputShape || null,
          children: [],
          parent: macroId,
          depth: macroDepth + stageIndex * 0.5,
          channelSize: stageMaxChannels,
          blockIndex: stageIndex,
        };
        
        // Build block nodes or individual layers (Level 3)
        if (stageGroup.blocks.size > 0) {
          let blockIdx = 0;
          stageGroup.blocks.forEach((blockLayers) => {
            blockLayers.forEach((layer, layerIdx) => {
              const layerNode = createLayerNode(layer, stageId, macroDepth + stageIndex * 0.5, blockIdx, layerIdx);
              stageNode.children.push(layerNode);
              allNodes.set(layer.id, layerNode);
            });
            blockIdx++;
          });
        } else {
          stageGroup.layers.forEach((layer, layerIdx) => {
            const layerNode = createLayerNode(layer, stageId, macroDepth + stageIndex * 0.5, 0, layerIdx);
            stageNode.children.push(layerNode);
            allNodes.set(layer.id, layerNode);
          });
        }
        
        macroNode.children.push(stageNode);
        allNodes.set(stageId, stageNode);
        stageIndex++;
      });
    } else {
      // No stages, add layers directly
      macroGroup.layers.forEach((layer, layerIdx) => {
        const layerNode = createLayerNode(layer, macroId, macroDepth, 0, layerIdx);
        macroNode.children.push(layerNode);
        allNodes.set(layer.id, layerNode);
      });
    }
    
    macroNodes.push(macroNode);
    allNodes.set(macroId, macroNode);
    macroDepth++;
  });
  
  // Build connections with hierarchy awareness
  const connections: HierarchyConnection[] = architecture.connections.map((conn, idx) => ({
    id: `conn_${idx}`,
    source: conn.source,
    target: conn.target,
    level: 3,
    tensorShape: conn.tensorShape,
    isSkipConnection: isSkipConnection(conn, layerMap),
    isDownsample: conn.source.includes('downsample') || conn.target.includes('downsample'),
  }));
  
  return {
    name: architecture.name,
    framework: architecture.framework,
    totalParameters: architecture.totalParameters,
    inputShape: architecture.inputShape || null,
    outputShape: architecture.outputShape || null,
    macroNodes,
    allNodes,
    connections,
    layers: architecture.layers,
  };
}

function createLayerNode(
  layer: LayerData,
  parentId: string,
  depth: number,
  blockIndex: number,
  layerIndex: number
): HierarchyNode {
  return {
    id: layer.id,
    name: layer.name,
    displayName: layer.name.split('.').pop() || layer.name,
    level: 3,
    type: 'layer',
    category: layer.category,
    layerCount: 1,
    totalParams: layer.numParameters,
    inputShape: layer.inputShape,
    outputShape: layer.outputShape,
    children: [],
    parent: parentId,
    depth: depth + layerIndex * 0.1,
    channelSize: inferChannelSize(layer),
    blockIndex,
    layerData: layer,
  };
}

function formatDisplayName(name: string): string {
  // Convert "encoder" → "Encoder", "layer1" → "Layer 1"
  return name
    .replace(/([a-z])(\d)/g, '$1 $2')
    .replace(/^./, c => c.toUpperCase())
    .replace(/_/g, ' ');
}

function inferMacroCategory(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('encoder')) return 'encoder';
  if (lower.includes('decoder')) return 'decoder';
  if (lower.includes('head') || lower.includes('output')) return 'output';
  if (lower.includes('embed')) return 'embedding';
  if (lower.includes('attention') || lower.includes('transformer')) return 'attention';
  return 'other';
}

/**
 * Get nodes at a specific level
 */
export function getNodesAtLevel(hierarchy: ModelHierarchy, level: 1 | 2 | 3): HierarchyNode[] {
  const nodes: HierarchyNode[] = [];
  
  function traverse(node: HierarchyNode) {
    if (node.level === level) {
      nodes.push(node);
    }
    node.children.forEach(traverse);
  }
  
  hierarchy.macroNodes.forEach(traverse);
  return nodes;
}

/**
 * Get connections between nodes at a specific level
 */
export function getConnectionsAtLevel(
  hierarchy: ModelHierarchy, 
  level: 1 | 2 | 3
): HierarchyConnection[] {
  if (level === 3) return hierarchy.connections;
  
  // Aggregate connections for higher levels
  const connectionSet = new Set<string>();
  const aggregatedConnections: HierarchyConnection[] = [];
  
  hierarchy.connections.forEach(conn => {
    const sourceNode = hierarchy.allNodes.get(conn.source);
    const targetNode = hierarchy.allNodes.get(conn.target);
    
    if (!sourceNode || !targetNode) return;
    
    let sourceId: string;
    let targetId: string;
    
    if (level === 1) {
      // Find macro parents
      sourceId = findAncestorAtLevel(sourceNode, 1, hierarchy) || conn.source;
      targetId = findAncestorAtLevel(targetNode, 1, hierarchy) || conn.target;
    } else {
      // Find stage parents
      sourceId = findAncestorAtLevel(sourceNode, 2, hierarchy) || conn.source;
      targetId = findAncestorAtLevel(targetNode, 2, hierarchy) || conn.target;
    }
    
    if (sourceId === targetId) return; // Skip internal connections
    
    const key = `${sourceId}_${targetId}`;
    if (!connectionSet.has(key)) {
      connectionSet.add(key);
      aggregatedConnections.push({
        ...conn,
        source: sourceId,
        target: targetId,
        level,
      });
    }
  });
  
  return aggregatedConnections;
}

function findAncestorAtLevel(node: HierarchyNode, level: 1 | 2 | 3, hierarchy: ModelHierarchy): string | null {
  if (node.level === level) return node.id;
  if (node.level < level) return null;
  
  if (node.parent) {
    const parent = hierarchy.allNodes.get(node.parent);
    if (parent) return findAncestorAtLevel(parent, level, hierarchy);
  }
  
  return null;
}
