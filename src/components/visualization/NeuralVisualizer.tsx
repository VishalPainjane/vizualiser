/**
 * Neural Network Visualizer
 * 
 * Main container component with VGG-style 3D architecture visualization.
 * Shows layers as 3D blocks where size represents tensor dimensions.
 */

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { ArchScene, type CameraView } from './ArchScene';
import { SavedModelsPanel } from './SavedModelsPanel';
import { saveModel, findModelByName, type ModelArchitecture } from '@/core/api-client';
import styles from './NeuralVisualizer.module.css';

// ============================================================================
// Types
// ============================================================================

export interface ModelArchitectureData {
  name: string;
  framework: string;
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
  onLayerSelect,
  onUploadNew,
  onLoadSavedModel,
}) => {
  // Ref for scene container (for screenshot)
  const sceneContainerRef = useRef<HTMLDivElement>(null);
  
  // View state
  const [showLabels, setShowLabels] = useState(true);
  const [showDimensions, setShowDimensions] = useState(true);
  const [showConnections, setShowConnections] = useState(false);
  
  // Selection state
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  
  // Saved models panel state
  const [showSavedModels, setShowSavedModels] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  
  // Key for forcing camera reset
  const [sceneKey, setSceneKey] = useState(0);
  
  // Camera view preset
  const [cameraView, setCameraView] = useState<CameraView | undefined>(undefined);
  
  // Get selected layer data
  const selectedLayer = useMemo(() => {
    if (!selectedLayerId || !architecture) return null;
    return architecture.layers.find(l => l.id === selectedLayerId) || null;
  }, [selectedLayerId, architecture]);
  
  // Handlers
  const handleLayerClick = useCallback((layerId: string) => {
    setSelectedLayerId(prev => prev === layerId ? null : layerId);
    onLayerSelect?.(layerId);
  }, [onLayerSelect]);
  
  const handleLayerHover = useCallback((_layerId: string | null) => {
    // Could show preview on hover
  }, []);
  
  const handleCloseDetail = useCallback(() => {
    setSelectedLayerId(null);
    onLayerSelect?.(null);
  }, [onLayerSelect]);
  
  const handleResetView = useCallback(() => {
    setSceneKey(prev => prev + 1);
    setCameraView(undefined);
  }, []);
  
  const handleSetView = useCallback((view: CameraView) => {
    setCameraView(view);
    // Force re-render to apply new view
    setSceneKey(prev => prev + 1);
  }, []);
  
  // Save image handler
  const handleSaveImage = useCallback(() => {
    const canvas = sceneContainerRef.current?.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `${architecture?.name || 'neural-network'}-visualization.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  }, [architecture?.name]);
  
  // Toggle fullscreen
  const handleFullscreen = useCallback(() => {
    const container = sceneContainerRef.current?.parentElement;
    if (!container) return;
    
    if (!document.fullscreenElement) {
      container.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, []);
  
  // Auto-save model after loading (only if not already saved)
  useEffect(() => {
    if (architecture && saveStatus === 'idle') {
      setSaveStatus('saving');
      
      // First check if model with this name already exists
      findModelByName(architecture.name)
        .then((existing) => {
          if (existing) {
            // Model already exists, mark as saved without creating new entry
            console.log(`[NN3D] Model "${architecture.name}" already saved (id: ${existing.id})`);
            setSaveStatus('saved');
            return;
          }
          
          // Model doesn't exist, save it
          const archForSave: ModelArchitecture = {
            name: architecture.name,
            framework: architecture.framework,
            totalParameters: architecture.totalParameters,
            trainableParameters: architecture.trainableParameters || 0,
            inputShape: architecture.inputShape || null,
            outputShape: architecture.outputShape || null,
            layers: architecture.layers,
            connections: architecture.connections,
          };
          
          return saveModel(
            architecture.name,
            architecture.framework,
            architecture.totalParameters,
            architecture.layers.length,
            archForSave
          ).then(() => {
            console.log(`[NN3D] Model "${architecture.name}" saved successfully`);
            setSaveStatus('saved');
          });
        })
        .catch((err) => {
          console.error('Failed to save model:', err);
          setSaveStatus('error');
        });
    }
  }, [architecture, saveStatus]);
  
  // Reset save status when architecture changes
  useEffect(() => {
    setSaveStatus('idle');
  }, [architecture?.name]);
  
  // Handle loading saved model
  const handleLoadSavedModel = useCallback((arch: any) => {
    if (onLoadSavedModel) {
      onLoadSavedModel(arch);
    }
    setShowSavedModels(false);
  }, [onLoadSavedModel]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      switch (e.key.toLowerCase()) {
        case 'l':
          setShowLabels(prev => !prev);
          break;
        case 'd':
          setShowDimensions(prev => !prev);
          break;
        case 'c':
          setShowConnections(prev => !prev);
          break;
        case 'r':
          handleResetView();
          break;
        case 'escape':
          handleCloseDetail();
          break;
        // View presets
        case '1':
          handleSetView('front');
          break;
        case '2':
          handleSetView('side');
          break;
        case '3':
          handleSetView('top');
          break;
        case '4':
          handleSetView('isometric');
          break;
        case '5':
          handleSetView('back');
          break;
        case '6':
          handleSetView('bottom');
          break;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleCloseDetail, handleResetView, handleSetView]);
  
  // Loading state
  if (isLoading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <span>// ANALYZING MODEL...</span>
        </div>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          <span className={styles.errorIcon}>[ERROR]</span>
          <h3>Analysis Failed</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }
  
  // Empty state
  if (!architecture) {
    return (
      <div className={styles.container}>
        <div className={styles.empty}>
          <div className={styles.emptyIcon}></div>
          <h3>Neural Network Visualizer</h3>
          <p>Drop a model file to visualize its architecture</p>
          <div className={styles.supportedFormats}>
            <span>// .pt .pth .onnx .h5 .keras .safetensors</span>
          </div>
          <button 
            className={styles.savedModelsBtn}
            onClick={() => setShowSavedModels(true)}
          >
            [SAVED MODELS]
          </button>
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
  }
  
  return (
    <div className={styles.container}>
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
          showDimensions={showDimensions}
          showConnections={showConnections}
          selectedLayerId={selectedLayerId}
          onLayerClick={handleLayerClick}
          onLayerHover={handleLayerHover}
          cameraView={cameraView}
        />
      </div>
      
      {/* Camera Control Panel */}
      <div className={styles.cameraControls}>
        <div className={styles.controlTitle}>// CAMERA_VIEWS</div>
        <div className={styles.viewButtons}>
          <button 
            className={styles.viewBtn} 
            onClick={() => handleSetView('front')}
            title="Front View (1)"
          >
            <span>[F]</span>
            <span>FRONT</span>
          </button>
          <button 
            className={styles.viewBtn} 
            onClick={() => handleSetView('side')}
            title="Side View (2)"
          >
            <span>[S]</span>
            <span>SIDE</span>
          </button>
          <button 
            className={styles.viewBtn} 
            onClick={() => handleSetView('top')}
            title="Top View (3)"
          >
            <span>[T]</span>
            <span>TOP</span>
          </button>
          <button 
            className={styles.viewBtn} 
            onClick={() => handleSetView('isometric')}
            title="Isometric View (4)"
          >
            <span>[3D]</span>
            <span>ISO</span>
          </button>
        </div>
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
            checked={showDimensions}
            onChange={(e) => setShowDimensions(e.target.checked)}
          />
          <span>DIMENSIONS</span>
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
