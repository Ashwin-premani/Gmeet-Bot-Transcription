import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { PdfReader } from 'pdfreader';
import mammoth from 'mammoth';

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = 'uploads/resumes';
    try {
      await fs.mkdir(uploadDir, { recursive: true });
      cb(null, uploadDir);
    } catch (error) {
      cb(error as Error, uploadDir);
    }
  },
  filename: (req, file, cb) => {
    // Generate unique filename with timestamp
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

// File filter for allowed types
const fileFilter = (req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ];
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only PDF, DOC, and DOCX files are allowed.'));
  }
};

export const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

// Text extraction functions
export async function extractTextFromFile(filePath: string, mimeType: string): Promise<string> {
  console.log(`Extracting text from: ${filePath}, type: ${mimeType}`);
  try {
    switch (mimeType) {
      case 'application/pdf':
        return await extractTextFromPDF(filePath);
      
      case 'application/msword':
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return await extractTextFromDOCX(filePath);
      
      default:
        throw new Error(`Unsupported file type: ${mimeType}`);
    }
  } catch (error) {
    console.error('Text extraction error:', error, 'File:', filePath, 'Type:', mimeType);
    throw new Error(`Failed to extract text from file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function extractTextFromPDF(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const textItems: string[] = [];
    
    new PdfReader().parseFileItems(filePath, (err, item) => {
      if (err) {
        console.error('PDF extraction error:', err);
        reject(err);
        return;
      }
      
      if (!item) {
        // End of file
        resolve(textItems.join(' '));
        return;
      }
      
      if (item.text) {
        textItems.push(item.text);
      }
    });
  });
}

async function extractTextFromDOCX(filePath: string): Promise<string> {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const result = await mammoth.extractRawText({ buffer: dataBuffer });
    return result.value || '';
  } catch (error) {
    console.error('DOCX extraction error:', error);
    throw error;
  }
}

// Clean up uploaded file
export async function deleteFile(filePath: string): Promise<void> {
  try {
    await fs.unlink(filePath);
  } catch (error) {
    console.error('File deletion error:', error);
    // Don't throw - file cleanup is not critical
  }
}

// Sanitize text for AI processing (remove sensitive personal info)
export function sanitizeResumeText(text: string): string {
  // Remove potential phone numbers
  text = text.replace(/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, '[PHONE]');
  
  // Remove potential email addresses (keep domain for context)
  text = text.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL]');
  
  // Remove potential addresses (very basic)
  text = text.replace(/\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b/gi, '[ADDRESS]');
  
  return text;
}

export interface ProcessedFile {
  filename: string;
  originalName: string;
  path: string;
  size: number;
  mimeType: string;
  extractedText: string;
  sanitizedText: string;
}

export async function processUploadedFile(file: Express.Multer.File): Promise<ProcessedFile> {
  const extractedText = await extractTextFromFile(file.path, file.mimetype);
  const sanitizedText = sanitizeResumeText(extractedText);
  
  return {
    filename: file.filename,
    originalName: file.originalname,
    path: file.path,
    size: file.size,
    mimeType: file.mimetype,
    extractedText,
    sanitizedText
  };
}
