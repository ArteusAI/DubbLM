
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
export type VideoQualityPreset = '720p' | '1080p' | 'original';
export type TtsStyleId = 'podcast' | 'lecture' | 'gothic' | 'news' | 'custom' | 'auto';
export type ResolvedTtsStyleId = 'podcast' | 'lecture' | 'gothic' | 'news';

export interface PresetConfig {
  id: PresetId;
  name: string;
  description: string;
  icon: string;
  personaId?: string;
  keepBackground?: boolean;
  llmProvider: LlmProvider;
  llmModelName: string;
  llmTemperature: number;
  enableLlmEditor?: boolean;
  enableLlmTextAdjustment?: boolean;
  editorLlmProvider?: LlmProvider;
  editorModelName?: string;
  editorTemperature?: number;
  editorReasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'none';
  refinementLlmProvider?: LlmProvider;
  refinementModelName?: string;
  refinementTemperature: number;
  ttsSystem: string;
  ttsModel?: string;
  ttsFallbackModel?: string;
  ttsStyle?: TtsStyleId;
  ttsPromptPrefix?: string;
  resolvedTtsStyle?: ResolvedTtsStyleId;
  voiceAutoSelection?: boolean;
  enableEmotionAnalysis?: boolean;
  enableEmotionEnrichment?: boolean;
  enableContentValidation?: boolean;
  dubbedVolume?: number;
  backgroundVolume?: number;
  useTwoPassEncoding?: boolean;
  videoQualityPreset?: VideoQualityPreset;
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

export type SpeakerGender = 'male' | 'female' | 'unknown';

export interface SpeakerMetadata {
  inferredGender: SpeakerGender;
  inferredConfidence: number;
  rawLabel?: string | null;
  modelId?: string | null;
  overrideGender?: SpeakerGender | null;
}

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
    openrouter?: string;
    minimax?: string;
    assemblyai?: string;
  };
  
  // Translation model settings (Extra)
  llmProvider?: LlmProvider;
  llmModelName?: string;
  llmTemperature?: number;
  enableLlmEditor?: boolean;
  enableLlmTextAdjustment?: boolean;
  editorLlmProvider?: LlmProvider;
  editorModelName?: string;
  editorTemperature?: number;
  editorReasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'none';
  speakerTtsPrompts?: Record<string, string>;
  speakerVoiceMappings?: Record<string, string>;
  enableSpeakerGenderInference?: boolean;
  speakerMetadata?: Record<string, SpeakerMetadata>;
  
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
  enableContentValidation?: boolean;
  contentValidatorProvider?: 'whisper' | 'assemblyai';
  contentValidatorWhisperModel?: string;
  contentValidatorWhisperComputeType?: string;
  contentValidatorWhisperCpuThreads?: number;
  contentValidatorSpeechModel?: string;
  ttsStyle?: TtsStyleId;
  ttsPromptPrefix?: string;
  resolvedTtsStyle?: ResolvedTtsStyleId;
  
  // Audio extras
  dubbedVolume?: number;
  backgroundVolume?: number;
  keepOriginalAudioRanges?: string[];
  useTwoPassEncoding?: boolean;
  videoQualityPreset?: VideoQualityPreset;
  
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
  speakerGenderTranslationStale?: boolean;
}

export interface Language {
  code: string;
  name: string;
}
