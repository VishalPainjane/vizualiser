/**
 * Saved Models Panel Component
 * 
 * Displays a list of previously saved models that can be loaded or deleted.
 */

import React, { useEffect, useState, useCallback } from 'react';
import styles from './SavedModelsPanel.module.css';
import { 
  getSavedModels, 
  getSavedModelById, 
  deleteSavedModel,
  type SavedModelSummary 
} from '@/core/api-client';

interface SavedModelsPanelProps {
  onLoadModel: (architecture: any) => void;
  onClose: () => void;
}

export const SavedModelsPanel: React.FC<SavedModelsPanelProps> = ({
  onLoadModel,
  onClose,
}) => {
  const [models, setModels] = useState<SavedModelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Fetch saved models on mount
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSavedModels();
      setModels(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load saved models');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Load a model
  const handleLoad = useCallback(async (id: number) => {
    setLoadingId(id);
    try {
      const model = await getSavedModelById(id);
      onLoadModel(model.architecture);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
    } finally {
      setLoadingId(null);
    }
  }, [onLoadModel, onClose]);

  // Delete a model
  const handleDelete = useCallback(async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!confirm('Are you sure you want to delete this model?')) {
      return;
    }
    
    setDeletingId(id);
    try {
      await deleteSavedModel(id);
      setModels(prev => prev.filter(m => m.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    } finally {
      setDeletingId(null);
    }
  }, []);

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Format parameters
  const formatParams = (params: number) => {
    if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
    if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`;
    return params.toString();
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>
            <span>&gt;_</span>
            SAVED_MODELS
          </h2>
          <button className={styles.closeBtn} onClick={onClose}>[x]</button>
        </div>
        
        <div className={styles.content}>
          {loading && (
            <div className={styles.loading}>
              <div className={styles.spinner} />
              <span>// LOADING SAVED MODELS...</span>
            </div>
          )}
          
          {error && (
            <div className={styles.error}>
              <span>[ERROR]</span>
              <span>{error}</span>
              <button onClick={fetchModels}>RETRY</button>
            </div>
          )}
          
          {!loading && !error && models.length === 0 && (
            <div className={styles.empty}>
              <span className={styles.emptyIcon}>[--]</span>
              <p>NO SAVED MODELS</p>
              <p className={styles.emptyHint}>
                UPLOAD AND PROCESS A MODEL TO SAVE IT HERE
              </p>
            </div>
          )}
          
          {!loading && models.length > 0 && (
            <div className={styles.modelList}>
              {models.map(model => (
                <div 
                  key={model.id}
                  className={styles.modelCard}
                  onClick={() => handleLoad(model.id)}
                >
                  <div className={styles.modelIcon}>[NN]</div>
                  <div className={styles.modelInfo}>
                    <div className={styles.modelName}>{model.name}</div>
                    <div className={styles.modelMeta}>
                      <span className={styles.framework}>{model.framework}</span>
                      <span className={styles.separator}>|</span>
                      <span>{model.layer_count} LAYERS</span>
                      <span className={styles.separator}>|</span>
                      <span>{formatParams(model.total_parameters)}</span>
                    </div>
                    <div className={styles.modelDate}>
                      {formatDate(model.created_at)}
                    </div>
                  </div>
                  <div className={styles.modelActions}>
                    {loadingId === model.id ? (
                      <div className={styles.miniSpinner} />
                    ) : (
                      <>
                        <button 
                          className={styles.loadBtn}
                          title="Load model"
                        >
                          <span className={styles.btnIcon}>&#9654;</span>
                          <span>LOAD</span>
                        </button>
                        <button 
                          className={styles.deleteBtn}
                          onClick={(e) => handleDelete(model.id, e)}
                          disabled={deletingId === model.id}
                          title="Delete model"
                        >
                          <span className={styles.btnIcon}>&#10005;</span>
                          {deletingId !== model.id && <span>DEL</span>}
                        </button>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SavedModelsPanel;
