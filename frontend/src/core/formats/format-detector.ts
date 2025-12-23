/**
 * Format detection for model files
 */

import { FormatInfo, FormatCategory, SUPPORTED_EXTENSIONS } from './types';

/**
 * Extension to format category mapping
 */
const EXTENSION_CATEGORIES: Record<string, FormatCategory> = {
  '.onnx': 'onnx',
  '.pt': 'pytorch',
  '.pth': 'pytorch',
  '.ckpt': 'pytorch',
  '.h5': 'tensorflow',
  '.hdf5': 'tensorflow',
  '.pb': 'tensorflow',
};

/**
 * Format descriptions and hints
 */
const FORMAT_DESCRIPTIONS: Record<FormatCategory, { description: string; hint?: string; canParse: boolean }> = {
  native: {
    description: 'NN3D native format (JSON)',
    canParse: true,
  },
  onnx: {
    description: 'Open Neural Network Exchange format',
    canParse: true,
  },
  pytorch: {
    description: 'PyTorch model checkpoint',
    canParse: false,
    hint: 'Use the Python exporter to convert: from nn3d_exporter import PyTorchExporter',
  },
  tensorflow: {
    description: 'TensorFlow/Keras HDF5 model',
    canParse: false,
    hint: 'Use the Python exporter to convert TensorFlow models to ONNX first, then load the ONNX file.',
  },
  safetensors: {
    description: 'SafeTensors format (header contains tensor metadata)',
    canParse: true,
  },
  generic: {
    description: 'Generic binary model format',
    canParse: false,
    hint: 'This format contains only weights. Use the Python exporter with your original model code.',
  },
  unknown: {
    description: 'Unknown model format',
    canParse: false,
    hint: 'Convert to ONNX or NN3D format using the appropriate framework tools.',
  },
};

/**
 * Detect format from file extension
 */
export function detectFormatFromExtension(filename: string): FormatInfo {
  const ext = ('.' + filename.split('.').pop()?.toLowerCase()) as string;
  const category = EXTENSION_CATEGORIES[ext] || 'unknown';
  const info = FORMAT_DESCRIPTIONS[category];
  
  return {
    extension: ext,
    category,
    canParseInBrowser: info.canParse,
    requiresBackend: !info.canParse,
    description: info.description,
    conversionHint: info.hint,
  };
}

/**
 * Detect format from file content (magic bytes)
 */
export async function detectFormatFromContent(file: File): Promise<FormatCategory> {
  const buffer = await file.slice(0, 16).arrayBuffer();
  const bytes = new Uint8Array(buffer);
  
  // Check for HDF5 magic bytes
  if (bytes.length >= 8 && 
      bytes[0] === 0x89 && bytes[1] === 0x48 && bytes[2] === 0x44 && bytes[3] === 0x46) {
    return 'tensorflow';
  }
  
  // Check for pickle protocol
  if (bytes.length >= 2 && bytes[0] === 0x80 && (bytes[1] === 0x04 || bytes[1] === 0x05)) {
    return 'pytorch';
  }
  
  // Check for SafeTensors (starts with u64 length, then JSON)
  if (bytes.length >= 8) {
    // Read first 8 bytes as little-endian u64 (header length)
    const headerLen = new DataView(buffer).getBigUint64(0, true);
    if (headerLen > 0 && headerLen < 1000000) { // Reasonable header size
      // Could be safetensors, verify by checking for JSON
      const possibleJson = await file.slice(8, 8 + Number(headerLen)).text();
      try {
        JSON.parse(possibleJson);
        return 'safetensors';
      } catch {
        // Not safetensors
      }
    }
  }
  
  // Check if it's JSON (native format)
  try {
    const text = await file.slice(0, 1000).text();
    JSON.parse(text.includes('{') ? text.substring(text.indexOf('{')) : text);
    return 'native';
  } catch {
    // Not JSON
  }
  
  // Default based on extension
  return detectFormatFromExtension(file.name).category;
}

/**
 * Check if extension is supported
 */
export function isSupportedExtension(filename: string): boolean {
  const ext = '.' + filename.split('.').pop()?.toLowerCase();
  return SUPPORTED_EXTENSIONS.includes(ext as any);
}

/**
 * Get all supported extensions as a string for file input accept attribute
 */
export function getSupportedExtensionsString(): string {
  return SUPPORTED_EXTENSIONS.join(',');
}

/**
 * Get user-friendly format name
 */
export function getFormatDisplayName(category: FormatCategory): string {
  const names: Record<FormatCategory, string> = {
    native: 'NN3D',
    onnx: 'ONNX',
    pytorch: 'PyTorch',
    tensorflow: 'TensorFlow/Keras',
    safetensors: 'SafeTensors',
    generic: 'Binary Model',
    unknown: 'Unknown',
  };
  return names[category];
}
