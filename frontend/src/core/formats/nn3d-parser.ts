
import type { ParseResult, FormatParser } from './types';
import { detectFormatFromExtension } from './format-detector';
import { parseNN3DModel } from '@/schema/validator';

export const NN3DParser: FormatParser = {
  extensions: ['.nn3d', '.json'],
  
  async canParse(file: File): Promise<boolean> {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    return ext === '.nn3d' || ext === '.json';
  },
  
  async parse(file: File): Promise<ParseResult> {
    const format = detectFormatFromExtension(file.name);
    const warnings: string[] = [];
    
    try {
      const text = await file.text();
      const { model, validation } = parseNN3DModel(text);
      
      if (!validation.valid || !model) {
        return {
            success: false,
            error: validation.errors.map(e => e.message).join('\n'),
            warnings,
            format,
            inferredStructure: false
        };
      }

      return {
        success: true,
        model,
        warnings,
        format,
        inferredStructure: false,
      };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to parse NN3D model',
        warnings,
        format,
        inferredStructure: false,
      };
    }
  }
};
