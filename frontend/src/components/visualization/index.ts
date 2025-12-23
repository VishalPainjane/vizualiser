/**
 * Visualization Components Index
 * 
 * Export all visualization-related components
 */

// Main visualizer container
export { default as NeuralVisualizer } from './NeuralVisualizer';
export type { NeuralVisualizerProps, ModelArchitectureData } from './NeuralVisualizer';

// Architecture scene (VGG-style 3D visualization)
export { ArchScene } from './ArchScene';
export type { ArchSceneProps } from './ArchScene';

// Legacy components (kept for reference)
// export { NeuralScene } from './NeuralScene';
// export { VisualizationControls } from './VisualizationControls';
// export { LayerDetailPanel } from './LayerDetailPanel';
