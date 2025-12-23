/**
 * Custom hook to run architecture layout computation in a web worker
 */
import { useState, useEffect, useMemo } from 'react';
import type { ArchitectureLayout } from '@/core/arch-layout';

export function useLayoutWorker(architecture: any | null) {
  const [layout, setLayout] = useState<ArchitectureLayout | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Memoize the worker instance to avoid re-creating it on every render
  const worker = useMemo(() => {
    // Check for SSR or environments where Worker is not available
    if (typeof Worker === 'undefined') {
      return null;
    }
    return new Worker(new URL('../core/layout.worker.ts', import.meta.url), { type: 'module' });
  }, []);

  useEffect(() => {
    if (!worker) {
      console.error("Web Workers are not supported in this environment.");
      return;
    }

    // Handle messages from the worker
    worker.onmessage = (event) => {
      const { type, payload } = event.data;
      if (type === 'SUCCESS') {
        setLayout(payload);
        setError(null);
      } else if (type === 'ERROR') {
        setLayout(null);
        setError(payload);
      }
      setIsLoading(false);
    };

    // Handle errors from the worker
    worker.onerror = (err) => {
      console.error('[useLayoutWorker] Worker error:', err);
      setError(err.message);
      setIsLoading(false);
    };

    // Cleanup function to terminate the worker when the component unmounts
    return () => {
      worker.terminate();
    };
  }, [worker]);

  // Effect to run the layout computation when the architecture changes
  useEffect(() => {
    if (architecture && worker) {
      setIsLoading(true);
      setLayout(null);
      setError(null);
      
      // Post the architecture data to the worker to start computation
      worker.postMessage({ architecture });
    } else {
      // If there's no architecture, reset the state
      setIsLoading(false);
      setLayout(null);
      setError(null);
    }
  }, [architecture, worker]);

  return { layout, isLoading, error };
}
