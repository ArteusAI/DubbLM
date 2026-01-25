
import { Persona, Language, PresetConfig } from './types';

export const LANGUAGES: Language[] = [
  { code: 'auto', name: 'Auto Detect' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'ja', name: 'Japanese' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ko', name: 'Korean' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
  { code: 'tr', name: 'Turkish' },
  { code: 'pl', name: 'Polish' },
  { code: 'nl', name: 'Dutch' },
  { code: 'uk', name: 'Ukrainian' },
];

export const DEFAULT_PERSONAS: Persona[] = [
  { id: 'none', name: 'None', description: 'No post-processing, raw translation only.' },
  { id: 'normal', name: 'Normal', description: 'Standard, natural translation preserving all details.' },
  { id: 'casual_manager', name: 'Casual Manager', description: 'Simplifies technical content for business audiences.' },
  { id: 'child', name: 'Child Friendly', description: 'Transforms complex topics into child-friendly stories.' },
  { id: 'housewife', name: 'Housewife', description: 'Explains technology for household and family management context.' },
  { id: 'science_popularizer', name: 'Science Popularizer', description: 'Makes complex topics engaging and understandable for general audience.' },
  { id: 'it_buddy', name: 'IT Buddy', description: 'Informal IT jargon with transliterated technical terms.' },
  { id: 'ai_buddy', name: 'AI Buddy', description: 'Clear, professional language for AI practitioners.' },
  { id: 'pedantic', name: 'Pedantic', description: 'Maximal fidelity to original wording and sentiment.' },
  { id: 'ai_visioner', name: 'AI Visioner', description: 'Strategic perspective on technical details with visionary context.' },
  { id: 'poet', name: 'Poet', description: 'Compact, rhymed poetry preserving original meaning.' },
  { id: 'pushkin_style', name: 'Pushkin Style', description: 'Classical poetry with perfect rhyme and meter in Pushkin tradition.' },
  { id: 'adhd_clarity', name: 'ADHD Clarity', description: 'Transforms rapid, scattered speech into clear, coherent dialogue.' },
];

export const TTS_PROVIDERS = [
  { id: 'openai', name: 'OpenAI' },
  { id: 'gemini', name: 'Gemini' },
  { id: 'minimax', name: 'MiniMax' },
];

export const LLM_PROVIDERS = [
  { id: 'gemini', name: 'Gemini' },
  { id: 'openrouter', name: 'OpenRouter' },
];

export const TRANSCRIPTION_PROVIDERS = [
  { id: 'assemblyai', name: 'AssemblyAI' },
  { id: 'openai', name: 'OpenAI + PyAnnote' },
  { id: 'whisperx', name: 'WhisperX (Local)' },
];

export const WHISPER_MODELS = [
  { id: 'large-v3', name: 'Large V3 (Best)' },
  { id: 'large-v2', name: 'Large V2' },
  { id: 'medium', name: 'Medium' },
  { id: 'small', name: 'Small' },
  { id: 'base', name: 'Base' },
  { id: 'tiny', name: 'Tiny (Fastest)' },
];

export const SPEAKER_COLORS = [
  'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
  'bg-rose-500/20 text-rose-300 border-rose-500/30',
  'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  'bg-amber-500/20 text-amber-300 border-amber-500/30',
  'bg-purple-500/20 text-purple-300 border-purple-500/30',
  'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  'bg-pink-500/20 text-pink-300 border-pink-500/30',
  'bg-teal-500/20 text-teal-300 border-teal-500/30',
];

export const PRESETS: PresetConfig[] = [
  {
    id: 'fast',
    name: 'Fast',
    description: '~1x video speed',
    icon: '⚡',
    llmProvider: 'gemini',
    llmModelName: 'gemini-flash-lite-latest',
    llmTemperature: 0.5,
    refinementLlmProvider: 'gemini',
    refinementModelName: 'gemini-flash-latest',
    refinementTemperature: 1.0,
    ttsSystem: 'openai',
    voiceAutoSelection: true,
    enableEmotionAnalysis: false,
    enableEmotionEnrichment: false,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    useTwoPassEncoding: true,
    maxWorkers: 4,
    pauseRemoval: 'disabled',
  },
  {
    id: 'hq',
    name: 'HQ',
    description: '~2x video duration',
    icon: '✨',
    llmProvider: 'gemini',
    llmModelName: 'gemini-flash-latest',
    llmTemperature: 0.5,
    refinementLlmProvider: 'gemini',
    refinementModelName: 'gemini-2.5-pro',
    refinementTemperature: 1.0,
    ttsSystem: 'gemini',
    ttsModel: 'gemini-2.5-flash-preview-tts',
    ttsFallbackModel: 'gemini-2.5-flash-preview-tts',
    ttsPromptPrefix: 'Speak with natural conversational energy, clear articulation:',
    voiceAutoSelection: true,
    enableEmotionAnalysis: false,
    enableEmotionEnrichment: false,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    useTwoPassEncoding: true,
    maxWorkers: 4,
    pauseRemoval: 'cut',
  },
  {
    id: 'ultra',
    name: 'Ultra',
    description: '~4x video duration',
    icon: '💎',
    llmProvider: 'gemini',
    llmModelName: 'gemini-2.5-pro',
    llmTemperature: 0.5,
    refinementLlmProvider: 'gemini',
    refinementModelName: 'gemini-2.5-pro',
    refinementTemperature: 1.0,
    ttsSystem: 'gemini',
    ttsModel: 'gemini-2.5-pro-preview-tts',
    ttsFallbackModel: 'gemini-2.5-pro-preview-tts',
    ttsPromptPrefix: 'Speak with natural conversational energy, clear articulation:',
    voiceAutoSelection: true,
    enableEmotionAnalysis: true,
    enableEmotionEnrichment: true,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    useTwoPassEncoding: true,
    maxWorkers: 4,
    pauseRemoval: 'cut',
    videoMinterpolateThreshold: 1.0,
  },
];
