
export enum AppStep {
  PROJECTS = 'PROJECTS',
  UPLOAD = 'UPLOAD',
  PROCESSING_TRANSCRIPTION = 'PROCESSING_TRANSCRIPTION',
  EDITOR = 'EDITOR',
  PROCESSING_DUBBING = 'PROCESSING_DUBBING',
  RESULT = 'RESULT'
}

export enum TTSProvider {
  GEMINI = 'gemini',
  OPENAI = 'openai',
  COQUI = 'coqui',
  MINIMAX = 'minimax'
}

export type PresetId = 'fast' | 'hq' | 'ultra';

export interface PresetConfig {
  id: PresetId;
  name: string;
  description: string;
  icon: string;
  llmProvider: LlmProvider;
  llmModelName: string;
  llmTemperature: number;
  refinementLlmProvider?: LlmProvider;
  refinementModelName?: string;
  refinementTemperature: number;
  ttsSystem: string;
  ttsModel?: string;
  ttsFallbackModel?: string;
  ttsPromptPrefix?: string;
  voiceAutoSelection?: boolean;
  enableEmotionAnalysis?: boolean;
  enableEmotionEnrichment?: boolean;
  dubbedVolume?: number;
  backgroundVolume?: number;
  useTwoPassEncoding?: boolean;
  maxWorkers?: number;
  pauseRemoval?: 'cut' | 'disabled';
  videoMinterpolateThreshold?: number;
}

export type ProjectStatus = 
  | 'draft'
  | 'transcribing'
  | 'transcribed'
  | 'dubbing'
  | 'dubbed'
  | 'error';

export interface Voice {
  id: string;
  name: string;
  provider: string;
  gender?: string;
}

export interface Segment {
  id: string;
  projectId?: string;
  speaker: string;
  startTime: number;
  endTime: number;
  originalText: string;
  translatedText: string;
  isMuted: boolean;
  voiceId: string;
  provider: string;
  isLocked?: boolean;
  audioUrl?: string;
  isSynthesizing?: boolean;
  ttsPrompt?: string;
  speakerColor?: string;
}

export interface Persona {
  id: string;
  name: string;
  description: string;
  promptTemplate?: string;
}

export interface ProcessingLog {
  id: string;
  message: string;
  timestamp: Date;
  type: 'info' | 'success' | 'error';
  text?: string;
  progress?: {
    current: number;
    total: number;
  };
}

export type LlmProvider = 'gemini' | 'openrouter';

export interface AppConfig {
  sourceLang: string;
  targetLang: string;
  defaultTargetLang?: string;
  personaId: string;
  speakerCount?: number;
  keepBackground: boolean;
  pauseRemoval: 'cut' | 'disabled';
  preset?: PresetId;
  apiKeys: {
    openai?: string;
    gemini?: string;
    minimax?: string;
    assemblyai?: string;
  };
  
  // Translation model settings (Extra)
  llmProvider?: LlmProvider;
  llmModelName?: string;
  llmTemperature?: number;
  speakerTtsPrompts?: Record<string, string>;
  
  // Refinement model settings (Extra)
  refinementLlmProvider?: LlmProvider;
  refinementModelName?: string;
  refinementTemperature?: number;
  translationPromptPrefix?: string;
  
  // TTS extras
  ttsSystem?: string;
  ttsModel?: string;
  voiceAutoSelection?: boolean;
  enableEmotionAnalysis?: boolean;
  enableEmotionEnrichment?: boolean;
  ttsPromptPrefix?: string;
  
  // Audio extras
  dubbedVolume?: number;
  backgroundVolume?: number;
  useTwoPassEncoding?: boolean;
  
  // Transcription extras
  transcriptionSystem?: 'assemblyai' | 'openai' | 'whisperx';
  whisperModel?: string;
  
  // Processing extras
  maxWorkers?: number;
  startTime?: number;
  duration?: number;
  
  // Segment merging
  postDiarizationMergeGap?: number;
  postTranslationMergeGap?: number;
  
  // Segment duration limits
  maxSegmentDuration?: number;
  minSegmentDuration?: number;
  
  // Audio/Video sync settings
  segmentStretch?: 'audio' | 'audio_and_video' | 'video';
  // Audio comfort zone (TTS speed adjustment)
  comfortMinAdjustmentRatio?: number;
  comfortMaxAdjustmentRatio?: number;
  
  // Pause processing
  minPauseDuration?: number;
  preservePauseDuration?: number;
}

export interface Project {
  id: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  status: ProjectStatus;
  config: AppConfig;
  segments: Segment[];
  videoFile: File | null;
  sourceFilename?: string;
  sourceSize?: number;
  
  // Upload state
  isUploading?: boolean;
  uploadProgress?: number;
  
  // Processing state
  isAutoProcessing?: boolean;
  processProgress?: number;
  processStage?: string;
  currentJobId?: string;
  error?: string;
}

export interface Language {
  code: string;
  name: string;
}
