import React, { useRef, useState, useCallback } from 'react';
import { useVisualizerStore } from '@/core/store';
import { loadModelFromFile, SUPPORTED_EXTENSIONS } from '@/core/loader';
import { computeLayout } from '@/core/layout';
import styles from './Toolbar.module.css';

/**
 * Main toolbar component
 */
export function Toolbar() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const loadModel = useVisualizerStore(state => state.loadModel);
  const updateNodePositions = useVisualizerStore(state => state.updateNodePositions);
  const config = useVisualizerStore(state => state.config);
  const updateConfig = useVisualizerStore(state => state.updateConfig);
  const model = useVisualizerStore(state => state.model);
  const resetCamera = useVisualizerStore(state => state.resetCamera);
  
  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setIsLoading(true);
    try {
      const loadedModel = await loadModelFromFile(file);
      loadModel(loadedModel);
      
      // Compute layout
      const layoutResult = computeLayout(loadedModel, {
        type: config.layout || 'layered',
        layerSpacing: config.layerSpacing || 3,
      });
      updateNodePositions(layoutResult.positions);
    } catch (error) {
      console.error('Failed to load model:', error);
      alert(`Failed to load model: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [loadModel, updateNodePositions, config]);
  
  const handleLayoutChange = useCallback((layout: string) => {
    if (!model) return;
    
    updateConfig({ layout: layout as any });
    const layoutResult = computeLayout(model, {
      type: layout as any,
      layerSpacing: config.layerSpacing || 3,
    });
    updateNodePositions(layoutResult.positions);
  }, [model, config, updateConfig, updateNodePositions]);
  
  const handleEdgeStyleChange = useCallback((style: string) => {
    updateConfig({ edgeStyle: style as any });
  }, [updateConfig]);
  
  return (
    <div className={styles.toolbar}>
      <div className={styles.section}>
        <input
          ref={fileInputRef}
          type="file"
          accept={SUPPORTED_EXTENSIONS.join(',')}
          onChange={handleFileSelect}
          className={styles.fileInput}
          id="model-file-input"
        />
        <label htmlFor="model-file-input" className={styles.button}>
          {isLoading ? '// LOADING...' : '[+] LOAD_MODEL'}
        </label>
      </div>
      
      {model && (
        <>
          <div className={styles.divider} />
          
          <div className={styles.section}>
            <span className={styles.label}>Layout:</span>
            <select
              className={styles.select}
              value={config.layout || 'layered'}
              onChange={(e) => handleLayoutChange(e.target.value)}
            >
              <option value="layered">Layered</option>
              <option value="force">Force-Directed</option>
              <option value="circular">Circular</option>
              <option value="hierarchical">Hierarchical</option>
            </select>
          </div>
          
          <div className={styles.section}>
            <span className={styles.label}>Edges:</span>
            <select
              className={styles.select}
              value={config.edgeStyle || 'tube'}
              onChange={(e) => handleEdgeStyleChange(e.target.value)}
            >
              <option value="tube">Tube</option>
              <option value="line">Line</option>
              <option value="bezier">Bezier</option>
              <option value="arrow">Arrow</option>
            </select>
          </div>
          
          <div className={styles.section}>
            <label className={styles.checkbox}>
              <input
                type="checkbox"
                checked={config.showLabels ?? true}
                onChange={(e) => updateConfig({ showLabels: e.target.checked })}
              />
              Labels
            </label>
            <label className={styles.checkbox}>
              <input
                type="checkbox"
                checked={config.showEdges ?? true}
                onChange={(e) => updateConfig({ showEdges: e.target.checked })}
              />
              Edges
            </label>
          </div>
          
          <div className={styles.divider} />
          
          <div className={styles.section}>
            <button className={styles.button} onClick={resetCamera}>
              [x] RESET_VIEW
            </button>
          </div>
          
          <div className={styles.modelInfo}>
            <span className={styles.modelName}>{model.metadata.name}</span>
            <span className={styles.modelStats}>
              {model.graph.nodes.length} layers • {model.graph.edges.length} connections
            </span>
          </div>
        </>
      )}
    </div>
  );
}

export default Toolbar;
