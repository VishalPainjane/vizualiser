/**
 * Web Worker for offloading heavy layout computation
 */
import { computeArchitectureLayout } from './arch-layout';
import type { ArchitectureLayout } from './arch-layout';

self.addEventListener('message', (event) => {
  console.log('[Worker] Received architecture for layout computation.');
  
  const { architecture } = event.data;
  
  if (architecture) {
    try {
      const layout: ArchitectureLayout = computeArchitectureLayout(architecture);
      console.log('[Worker] Layout computation complete.');
      // Post the result back to the main thread
      self.postMessage({ type: 'SUCCESS', payload: layout });
    } catch (error) {
      console.error('[Worker] Layout computation failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred in the layout worker.';
      self.postMessage({ type: 'ERROR', payload: errorMessage });
    }
  }
});
