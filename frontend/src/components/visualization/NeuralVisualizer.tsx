/**
 * Neural Network Visualizer
 * 
 * Main container component with VGG-style 3D architecture visualization.
 * Shows layers as 3D blocks where size represents tensor dimensions.
 */

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { ArchScene } from './ArchScene'; // Visualization component
import { SavedModelsPanel } from './SavedModelsPanel';
import styles from './NeuralVisualizer.module.css';

// ============================================================================
// Types
// ============================================================================

export interface ModelArchitectureData {
  name: string;
  framework: string;
  tags?: string[];
  totalParameters: number;
  trainableParameters?: number;
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
    trainable: boolean;
  }>;
  connections: Array<{
    source: string;
    target: string;
    tensorShape: number[] | null;
  }>;
}

export interface NeuralVisualizerProps {
  /** Model architecture data from backend */
  architecture: ModelArchitectureData | null;
  
  /** Loading state */
  isLoading?: boolean;
  
  /** Error message */
  error?: string | null;

  /** Warning message (e.g. for Bronze Path/Weights Only) */
  warning?: string | null;
  
  /** Callback when a layer is selected */
  onLayerSelect?: (layerId: string | null) => void;
  
  /** Callback to upload a new model */
  onUploadNew?: () => void;
  
  /** Callback to load a saved model */
  onLoadSavedModel?: (architecture: any) => void;
}

// ============================================================================
// Main Component
// ============================================================================

export const NeuralVisualizer: React.FC<NeuralVisualizerProps> = ({
  architecture,
  isLoading = false,
  error = null,
  warning = null,
  onLayerSelect,
  onUploadNew,
  onLoadSavedModel,
}) => {
  // Ref for scene container (for screenshot)
  const sceneContainerRef = useRef<HTMLDivElement>(null);
  
  // View state
  const [showLabels, setShowLabels] = useState(true);
  const [showConnections, setShowConnections] = useState(false);
  const [showWarning, setShowWarning] = useState(true);
  
  // Selection state
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  
  // Saved models panel state
  const [showSavedModels, setShowSavedModels] = useState(false);
  const [saveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  
  // Key for forcing camera reset
  const [sceneKey, setSceneKey] = useState(0);
  
  // Reset warning visibility when warning changes
  useEffect(() => {
    if (warning) setShowWarning(true);
  }, [warning]);

  // Derived state
  const selectedLayer = useMemo(() => {
    if (!architecture || !selectedLayerId) return null;
    return architecture.layers.find(l => l.id === selectedLayerId);
  }, [architecture, selectedLayerId]);

  // Handlers
  const handleLayerClick = useCallback((layerId: string) => {
    setSelectedLayerId(layerId);
    onLayerSelect?.(layerId);
  }, [onLayerSelect]);

  const handleResetView = useCallback(() => {
    setSceneKey(prev => prev + 1);
    setSelectedLayerId(null);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedLayerId(null);
    onLayerSelect?.(null);
  }, [onLayerSelect]);

  const handleSaveImage = useCallback(async () => {
    if (!sceneContainerRef.current) return;
    
    try {
      // Find the canvas element
      const canvas = sceneContainerRef.current.querySelector('canvas');
      if (!canvas) return;
      
      // Convert to blob and download
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${architecture?.name || 'model'}_visualization.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    } catch (err) {
      console.error('Failed to save image:', err);
    }
  }, [architecture]);

  const handleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }, []);

  const handleLoadSavedModel = useCallback((model: any) => {
    onLoadSavedModel?.(model);
    setShowSavedModels(false);
  }, [onLoadSavedModel]);
  
  // Loading state
  if (isLoading) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <span>PROCESSING_MODEL...</span>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return <div className={styles.error}>Error: {error}</div>;
  }
  
  // Empty state
  if (!architecture) {
    return <div className={styles.empty}>No model loaded.</div>;
  }
  
  return (
    <div className={styles.container}>
      {/* Informational Toast for Weights-Only / Matrix Layout */}
      {architecture.tags?.includes('weights-only') && (
        <div className={styles.warningToast} style={{ background: '#2D3748', border: '1px solid #4A5568' }}>
           <span className={styles.warningIcon}>ℹ️</span>
           <span className={styles.warningText}>
             Model structure not detected. Visualizing layer weights in 3D space (Graphical Matrix Layout).
           </span>
           <button className={styles.warningClose} onClick={(e) => {
             // Just hide this specific toast instance
             (e.target as HTMLElement).parentElement!.style.display = 'none';
           }}>[x]</button>
        </div>
      )}

      {/* Warning Toast */}
      {warning && showWarning && (
        <div className={styles.warningToast}>
           <span className={styles.warningIcon}>⚠️</span>
           <span className={styles.warningText}>{warning}</span>
           <button className={styles.warningClose} onClick={() => setShowWarning(false)}>[x]</button>
        </div>
      )}

      {/* Top Toolbar */}
      <div className={styles.topToolbar}>

        <div className={styles.modelInfo}>
          <button 
            className={styles.modelIcon} 
            onClick={onUploadNew}
            title="Back to Home"
          >
            NN
          </button>
          <span className={styles.modelName}>{architecture.name}</span>
          {saveStatus === 'saved' && (
            <span className={styles.savedBadge} title="Model saved">[SAVED]</span>
          )}
        </div>
        <div className={styles.toolbarActions}>
          <button 
            className={styles.toolbarBtn}
            onClick={() => setShowSavedModels(true)}
            title="View Saved Models"
          >
            <span className={styles.icon}>&gt;_</span>
            <span>SAVED</span>
          </button>
          <button 
            className={styles.toolbarBtn}
            onClick={handleSaveImage}
            title="Save as PNG"
          >
            <span className={styles.icon}>[*]</span>
            <span>EXPORT</span>
          </button>
          <button 
            className={styles.toolbarBtn}
            onClick={handleFullscreen}
            title="Toggle Fullscreen"
          >
            <span className={styles.icon}>&lt;&gt;</span>
            <span>FULLSCREEN</span>
          </button>
          {onUploadNew && (
            <button 
              className={`${styles.toolbarBtn} ${styles.uploadBtn}`}
              onClick={onUploadNew}
              title="Upload New Model"
            >
              <span className={styles.icon}>++</span>
              <span>NEW</span>
            </button>
          )}
        </div>
      </div>
      
      {/* 3D Architecture Scene */}
      <div className={styles.scene} ref={sceneContainerRef}>
        <ArchScene
          key={sceneKey}
          architecture={architecture}
          showLabels={showLabels}
          showConnections={showConnections}
          selectedLayerId={selectedLayerId}
          onLayerClick={handleLayerClick}
        />
      </div>
      
      {/* Camera Control Panel */}
      <div className={styles.cameraControls}>
        <div className={styles.controlTitle}>// CAMERA</div>
        <button className={styles.resetBtn} onClick={handleResetView}>
          [x] RESET_VIEW
        </button>
      </div>
      
      {/* Controls Panel */}
      <div className={styles.controls}>
        <div className={styles.controlTitle}>// DISPLAY_OPTIONS</div>
        
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
          />
          <span>LAYER_NAMES</span>
        </label>
        
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={showConnections}
            onChange={(e) => setShowConnections(e.target.checked)}
          />
          <span>CONNECTIONS</span>
        </label>
        
        <div className={styles.navigationSection}>
          <div className={styles.controlTitle}>// NAVIGATION</div>
          <div className={styles.navHint}>
            <span className={styles.navIcon}>[?]</span>
            <div>
              <div><strong>L_DRAG</strong> - MOVE</div>
              <div><strong>CTRL+L_DRAG</strong> - ROTATE</div>
              <div><strong>R_DRAG</strong> - ROTATE</div>
              <div><strong>SCROLL</strong> - ZOOM</div>
            </div>
          </div>
        </div>
        
        <div className={styles.shortcutsSection}>
          <div className={styles.controlTitle}>// SHORTCUTS</div>
          <div className={styles.shortcut}><kbd>1-4</kbd> VIEWS</div>
          <div className={styles.shortcut}><kbd>R</kbd> RESET</div>
          <div className={styles.shortcut}><kbd>L</kbd> LABELS</div>
          <div className={styles.shortcut}><kbd>D</kbd> DIMS</div>
          <div className={styles.shortcut}><kbd>C</kbd> CONNECT</div>
        </div>
      </div>
      
      {/* Layer Detail Panel */}
      {selectedLayer && (
        <div className={styles.detailPanel}>
          <button className={styles.closeBtn} onClick={handleCloseDetail}>[x]</button>
          
          <h3 className={styles.detailTitle}>{selectedLayer.name}</h3>
          
          <div className={styles.detailBadge} data-category={selectedLayer.category}>
            {selectedLayer.category.toUpperCase()}
          </div>
          
          <div className={styles.detailSection}>
            <div className={styles.detailLabel}>TYPE</div>
            <div className={styles.detailValue}>{selectedLayer.type}</div>
          </div>
          
          {selectedLayer.inputShape && (
            <div className={styles.detailSection}>
              <div className={styles.detailLabel}>INPUT_SHAPE</div>
              <div className={styles.detailValue}>[{selectedLayer.inputShape.join(', ')}]</div>
            </div>
          )}
          
          {selectedLayer.outputShape && (
            <div className={styles.detailSection}>
              <div className={styles.detailLabel}>OUTPUT_SHAPE</div>
              <div className={styles.detailValue}>[{selectedLayer.outputShape.join(', ')}]</div>
            </div>
          )}
          
          {selectedLayer.numParameters > 0 && (
            <div className={styles.detailSection}>
              <div className={styles.detailLabel}>PARAMETERS</div>
              <div className={styles.detailValue}>{selectedLayer.numParameters.toLocaleString()}</div>
            </div>
          )}
          
          {Object.keys(selectedLayer.params).length > 0 && (
            <div className={styles.detailSection}>
              <div className={styles.detailLabel}>CONFIG</div>
              <div className={styles.configTable}>
                {Object.entries(selectedLayer.params).map(([key, value]) => (
                  <div key={key} className={styles.configRow}>
                    <span className={styles.configKey}>{key}</span>
                    <span className={styles.configValue}>
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Stats Bar */}
      <div className={styles.statsBar}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Model</span>
          <span className={styles.statValue}>{architecture.name}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Framework</span>
          <span className={styles.statValue}>{architecture.framework}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Layers</span>
          <span className={styles.statValue}>{architecture.layers.length}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Parameters</span>
          <span className={styles.statValue}>{formatParams(architecture.totalParameters)}</span>
        </div>
      </div>
      
      {/* Saved Models Panel */}
      {showSavedModels && (
        <SavedModelsPanel
          onLoadModel={handleLoadSavedModel}
          onClose={() => setShowSavedModels(false)}
        />
      )}
    </div>
  );
};

// Helper
function formatParams(params: number): string {
  if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`;
  return params.toString();
}

export default NeuralVisualizer;
