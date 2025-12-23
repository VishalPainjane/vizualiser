import type { NN3DModel, Position3D } from '@/schema/types';

/**
 * Layout algorithm type
 */
export type LayoutType = 'layered' | 'force' | 'circular' | 'hierarchical' | 'custom';

/**
 * Layout configuration options
 */
export interface LayoutOptions {
  type: LayoutType;
  layerSpacing: number;
  nodeSpacing: number;
  direction: 'horizontal' | 'vertical' | 'depth';
  centerGraph: boolean;
}

/**
 * Default layout options
 */
const DEFAULT_OPTIONS: LayoutOptions = {
  type: 'layered',
  layerSpacing: 3.0,
  nodeSpacing: 2.0,
  direction: 'horizontal',
  centerGraph: true,
};

/**
 * Layout result
 */
export interface LayoutResult {
  positions: Map<string, Position3D>;
  bounds: {
    min: Position3D;
    max: Position3D;
    center: Position3D;
  };
}

/**
 * Compute layout for a neural network graph
 */
export function computeLayout(
  model: NN3DModel,
  options: Partial<LayoutOptions> = {}
): LayoutResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  switch (opts.type) {
    case 'layered':
      return computeLayeredLayout(model, opts);
    case 'force':
      return computeForceLayout(model, opts);
    case 'circular':
      return computeCircularLayout(model, opts);
    case 'hierarchical':
      return computeHierarchicalLayout(model, opts);
    default:
      return computeLayeredLayout(model, opts);
  }
}

/**
 * Layered layout - nodes arranged in layers based on depth
 */
function computeLayeredLayout(model: NN3DModel, options: LayoutOptions): LayoutResult {
  const { nodes, edges } = model.graph;
  const positions = new Map<string, Position3D>();
  
  // Build adjacency and compute depths
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();
  
  nodes.forEach(node => {
    inDegree.set(node.id, 0);
    adjacency.set(node.id, []);
  });
  
  edges.forEach(edge => {
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    adjacency.get(edge.source)?.push(edge.target);
  });
  
  // Topological sort to determine layers
  const layers: string[][] = [];
  const nodeDepths = new Map<string, number>();
  const queue: string[] = [];
  
  // Start with nodes having no incoming edges
  nodes.forEach(node => {
    if (inDegree.get(node.id) === 0) {
      queue.push(node.id);
      nodeDepths.set(node.id, 0);
    }
  });
  
  // BFS to assign depths
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const depth = nodeDepths.get(nodeId) || 0;
    
    // Ensure layer array exists
    if (!layers[depth]) {
      layers[depth] = [];
    }
    layers[depth].push(nodeId);
    
    // Process neighbors
    const neighbors = adjacency.get(nodeId) || [];
    for (const neighbor of neighbors) {
      const newDegree = (inDegree.get(neighbor) || 1) - 1;
      inDegree.set(neighbor, newDegree);
      
      if (newDegree === 0) {
        nodeDepths.set(neighbor, depth + 1);
        queue.push(neighbor);
      }
    }
  }
  
  // Handle cycles (nodes not yet placed)
  nodes.forEach(node => {
    if (!nodeDepths.has(node.id)) {
      const lastLayerIndex = layers.length;
      if (!layers[lastLayerIndex]) {
        layers[lastLayerIndex] = [];
      }
      layers[lastLayerIndex].push(node.id);
      nodeDepths.set(node.id, lastLayerIndex);
    }
  });
  
  // Compute positions
  let maxWidth = 0;
  layers.forEach(layer => {
    maxWidth = Math.max(maxWidth, layer.length);
  });
  
  layers.forEach((layer, layerIndex) => {
    const layerWidth = (layer.length - 1) * options.nodeSpacing;
    const startOffset = -layerWidth / 2;
    
    layer.forEach((nodeId, nodeIndex) => {
      let x: number, y: number, z: number;
      
      if (options.direction === 'vertical') {
        // Vertical: layers stack along Y (top to bottom), nodes spread on X
        x = startOffset + nodeIndex * options.nodeSpacing;
        y = -layerIndex * options.layerSpacing;
        z = 0;
      } else if (options.direction === 'horizontal') {
        // Horizontal: layers stack along X (left to right), nodes spread on Y
        x = layerIndex * options.layerSpacing;
        y = startOffset + nodeIndex * options.nodeSpacing;
        z = 0;
      } else {
        // Depth: layers stack along Z (front to back), nodes spread on X
        x = startOffset + nodeIndex * options.nodeSpacing;
        y = 0;
        z = -layerIndex * options.layerSpacing;
      }
      
      positions.set(nodeId, { x, y, z });
    });
  });
  
  return {
    positions,
    bounds: computeBounds(positions),
  };
}

/**
 * Force-directed layout using simple spring simulation
 */
function computeForceLayout(model: NN3DModel, options: LayoutOptions): LayoutResult {
  const { nodes, edges } = model.graph;
  const positions = new Map<string, Position3D>();
  const velocities = new Map<string, Position3D>();
  
  // Initialize random positions
  nodes.forEach((node, i) => {
    const angle = (i / nodes.length) * Math.PI * 2;
    const radius = 5 + Math.random() * 5;
    positions.set(node.id, {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      z: (Math.random() - 0.5) * 5,
    });
    velocities.set(node.id, { x: 0, y: 0, z: 0 });
  });
  
  // Simulation parameters
  const iterations = 100;
  const repulsion = 50;
  const attraction = 0.1;
  const damping = 0.9;
  
  for (let iter = 0; iter < iterations; iter++) {
    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const posA = positions.get(nodes[i].id)!;
        const posB = positions.get(nodes[j].id)!;
        
        const dx = posB.x - posA.x;
        const dy = posB.y - posA.y;
        const dz = posB.z - posA.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;
        
        const force = repulsion / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        const fz = (dz / dist) * force;
        
        const velA = velocities.get(nodes[i].id)!;
        const velB = velocities.get(nodes[j].id)!;
        
        velA.x -= fx;
        velA.y -= fy;
        velA.z -= fz;
        velB.x += fx;
        velB.y += fy;
        velB.z += fz;
      }
    }
    
    // Attraction along edges
    edges.forEach(edge => {
      const posA = positions.get(edge.source);
      const posB = positions.get(edge.target);
      if (!posA || !posB) return;
      
      const dx = posB.x - posA.x;
      const dy = posB.y - posA.y;
      const dz = posB.z - posA.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      const force = attraction * (dist - options.nodeSpacing);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      const fz = (dz / dist) * force;
      
      const velA = velocities.get(edge.source)!;
      const velB = velocities.get(edge.target)!;
      
      velA.x += fx;
      velA.y += fy;
      velA.z += fz;
      velB.x -= fx;
      velB.y -= fy;
      velB.z -= fz;
    });
    
    // Apply velocities with damping
    nodes.forEach(node => {
      const pos = positions.get(node.id)!;
      const vel = velocities.get(node.id)!;
      
      pos.x += vel.x;
      pos.y += vel.y;
      pos.z += vel.z;
      
      vel.x *= damping;
      vel.y *= damping;
      vel.z *= damping;
    });
  }
  
  // Center the graph
  if (options.centerGraph) {
    const bounds = computeBounds(positions);
    positions.forEach((pos, id) => {
      positions.set(id, {
        x: pos.x - bounds.center.x,
        y: pos.y - bounds.center.y,
        z: pos.z - bounds.center.z,
      });
    });
  }
  
  return {
    positions,
    bounds: computeBounds(positions),
  };
}

/**
 * Circular layout - nodes arranged in a circle by layer
 */
function computeCircularLayout(model: NN3DModel, options: LayoutOptions): LayoutResult {
  const { nodes } = model.graph;
  const positions = new Map<string, Position3D>();
  
  const nodeCount = nodes.length;
  const radius = Math.max(nodeCount * options.nodeSpacing / (2 * Math.PI), 5);
  
  nodes.forEach((node, i) => {
    const angle = (i / nodeCount) * Math.PI * 2;
    positions.set(node.id, {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      z: 0,
    });
  });
  
  return {
    positions,
    bounds: computeBounds(positions),
  };
}

/**
 * Hierarchical layout - tree-like structure
 */
function computeHierarchicalLayout(model: NN3DModel, options: LayoutOptions): LayoutResult {
  // Use layered layout as base, but with different spacing
  return computeLayeredLayout(model, {
    ...options,
    layerSpacing: options.layerSpacing * 1.5,
  });
}

/**
 * Compute bounding box of positions
 */
function computeBounds(positions: Map<string, Position3D>): {
  min: Position3D;
  max: Position3D;
  center: Position3D;
} {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  
  positions.forEach(pos => {
    minX = Math.min(minX, pos.x);
    minY = Math.min(minY, pos.y);
    minZ = Math.min(minZ, pos.z);
    maxX = Math.max(maxX, pos.x);
    maxY = Math.max(maxY, pos.y);
    maxZ = Math.max(maxZ, pos.z);
  });
  
  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      z: (minZ + maxZ) / 2,
    },
  };
}

export { computeBounds };
