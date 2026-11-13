
import { Persona, Language, PresetConfig } from './types';

// Deprecated: Gemini (legacy) provider is hidden from the UI and existing
// configs are migrated to gemini38. Kept for a later cleanup.
// Keep aligned with DEFAULT_GEMINI_TTS_MODEL / DEFAULT_GEMINI_TTS_FALLBACK_MODEL
// in src/tts/gemini_tts_wrapper.py (the backend source of truth).
export const GEMINI_DEFAULT_TTS_MODEL = 'gemini-2.5-pro-preview-tts';
export const GEMINI_DEFAULT_TTS_FALLBACK_MODEL = 'gemini-2.5-flash-preview-tts';
export const GEMINI_EXPERIMENTAL_TTS_MODEL = 'gemini-3.1-flash-tts-preview';
export const GEMINI_TTS_MODEL_OPTIONS = [
  GEMINI_DEFAULT_TTS_MODEL,
  GEMINI_DEFAULT_TTS_FALLBACK_MODEL,
  GEMINI_EXPERIMENTAL_TTS_MODEL,
];

// Gemini 3.8 Flash TTS (Interactions API, per-segment synthesis).
// Keep aligned with DEFAULT_GEMINI38_TTS_MODEL in
// src/tts/gemini38_tts_wrapper.py (the backend source of truth).
export const GEMINI38_DEFAULT_TTS_MODEL = 'google/gemini-3.8-flash-tts';
export const GEMINI38_DEFAULT_TTS_FALLBACK_MODEL = 'google/gemini-3.8-flash-lite-tts';
export const GEMINI38_TTS_MODEL_OPTIONS = [
  GEMINI38_DEFAULT_TTS_MODEL,
  GEMINI38_DEFAULT_TTS_FALLBACK_MODEL,
];

// Deprecated legacy provider id. The backend still accepts it for old API
// clients, but the UI never offers it; saved configs are migrated on load.
export const LEGACY_GEMINI_TTS_SYSTEM = 'gemini';

export const normalizeTtsSystem = (
  ttsSystem: string,
  ttsModel?: string,
  ttsFallbackModel?: string,
): { ttsSystem: string; ttsModel?: string; ttsFallbackModel?: string } => {
  if (ttsSystem !== LEGACY_GEMINI_TTS_SYSTEM) {
    return { ttsSystem, ttsModel, ttsFallbackModel };
  }
  return {
    ttsSystem: 'gemini38',
    ttsModel: GEMINI38_DEFAULT_TTS_MODEL,
    ttsFallbackModel: GEMINI38_DEFAULT_TTS_FALLBACK_MODEL,
  };
};

// Gemini 3.8 and the legacy Gemini provider share the same prebuilt voice
// catalog. The API exposes that catalog under the legacy "gemini" id, while
// the UI only offers "gemini38".
export const resolveVoiceCatalogProvider = (ttsSystem: string): string =>
  ttsSystem === 'gemini38' ? LEGACY_GEMINI_TTS_SYSTEM : ttsSystem;

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
  { id: 'tractorman', name: 'Tractorman', description: 'Casual informal style with sparing profanity for emphasis.' },
  { id: 'ai_buddy', name: 'AI Buddy', description: 'Clear, professional language for AI practitioners.' },
  { id: 'pedantic', name: 'Pedantic', description: 'Maximal fidelity to original wording and sentiment.' },
  { id: 'ai_visioner', name: 'AI Visioner', description: 'Strategic perspective on technical details with visionary context.' },
  { id: 'poet', name: 'Poet', description: 'Compact, rhymed poetry preserving original meaning.' },
  { id: 'pushkin_style', name: 'Pushkin Style', description: 'Classical poetry with perfect rhyme and meter in Pushkin tradition.' },
  { id: 'adhd_clarity', name: 'ADHD Clarity', description: 'Transforms rapid, scattered speech into clear, coherent dialogue.' },
];

export const TTS_PROVIDERS = [
  { id: 'gemini38', name: 'Gemini 3.8 (OpenRouter)' },
  { id: 'openrouter', name: 'OpenRouter (Qwen / Fish / Grok)' },
  { id: 'openai', name: 'OpenAI (legacy)' },
  { id: 'minimax', name: 'MiniMax (legacy)' },
];

export const OPENROUTER_DEFAULT_TTS_MODEL = 'qwen/qwen-audio-3.0-tts-flash';
export const OPENROUTER_TTS_MODEL_OPTIONS = [
  'qwen/qwen-audio-3.0-tts-flash',
  'qwen/qwen-audio-3.0-tts-plus',
  'x-ai/grok-voice-tts-1.0',
  'fish-audio/drama-3',
] as const;

// Voices accepted by each known OpenRouter speech model.
// The API reports the same information per voice (`models`); this map is the
// offline fallback for voices that do not carry that metadata.
export const OPENROUTER_MODEL_VOICES: Record<string, string[]> = {
  'qwen/qwen-audio-3.0-tts-flash': ['loongjohn', 'longanhuan_v3.6'],
  'qwen/qwen-audio-3.0-tts-plus': ['longanlingxin', 'longanlufeng'],
  'x-ai/grok-voice-tts-1.0': ['eve', 'ara', 'rex', 'sal', 'leo'],
  // Fish Audio Drama 3 selects voices by Fish voice id. These are the curated
  // Russian adult voices; the backend returns them with `curated: true`.
  'fish-audio/drama-3': [
    '67ecc0169ec54f4eb99ff59680cd0458', // Тёплый рассказчик (male)
    '7312c38557eb4fb384e3874e8e9cea67', // Профессиональный баритон (male)
    'fa3e3b7d3c2a44d88d9f018bc45db1ce', // Низкий русский (male)
    'd315712918f542e5bf866367e4a4975f', // Мягкий наблюдатель (male)
    '7f34b2a34b6e461aa2dc9632d7c737f3', // Тёплый баритон (male)
    '9f414bb4bdc14378854ad9320f9e6f0e', // Молодой разговорный (male)
    '6ada82e315e648b6a30a2e2210844cbf', // Дерзкий молодой (male)
    'cfd7057a69204777af3a570d3e3ef164', // Тихий баритон (male)
    '773ed9d827bb4c0d8b7a1b9267a4255e', // Спокойный резонансный (male)
    '172565be42474bf093d16c9b60885357', // Сбалансированный баритон (male)
    '9cc4883d4bbe4276bb67584db3cd117b', // Спокойная рассказчица (female)
    '12f7ab6416aa490880e63c864a434618', // Мягкий интимный голос (female)
    '4bf88e0a955748358c0e06a7b656b2af', // Ясный молодой альт (female)
    '345193fc9ae942888141767d2bc9c4c9', // Чистый молодой голос (female)
    '9a6b5d73a86042dc8191a8c62adfbe47', // Тёплый мелодичный альт (female)
  ],
};

export const LLM_PROVIDERS = [
  { id: 'openrouter', name: 'OpenRouter' },
  { id: 'gemini', name: 'Gemini (legacy)' },
];

export const TTS_STYLES: {
  id: 'podcast' | 'lecture' | 'gothic' | 'news' | 'custom' | 'auto';
  label: string;
  icon: string;
  description: string;
  prompt?: string;
}[] = [
  {
    id: 'podcast',
    label: 'Podcast',
    icon: 'Mic',
    description: 'Casual conversation, two hosts, warm and energetic.',
    prompt: 'Speak with natural conversational energy, clear articulation:',
  },
  {
    id: 'lecture',
    label: 'Lecture',
    icon: 'GraduationCap',
    description: 'Instructional e-learning, clear and articulate.',
    prompt: 'Read aloud in a calm, authoritative and articulate manner with measured pacing:',
  },
  {
    id: 'gothic',
    label: 'Gothic',
    icon: 'BookMarked',
    description: 'Slow, atmospheric storytelling with deep resonance.',
    prompt: 'Read in a slow, deliberate, suspenseful tone with deep resonance:',
  },
  {
    id: 'news',
    label: 'News',
    icon: 'Radio',
    description: 'Energetic broadcast narration for news episodes.',
    prompt: 'Synthesize only the transcript below in the language it is written; do not read these directions. Deliver it like a modern news explainer: confident, crisp broadcast narration with forward momentum, energetic narrative cadence, clear factual emphasis, bright but serious tone, no shouting or parody, short clean pauses between clauses:',
  },
  {
    id: 'custom',
    label: 'Custom',
    icon: 'Pencil',
    description: 'Write your own scene & style prompt in Advanced settings.',
  },
  {
    id: 'auto',
    label: 'Auto',
    icon: 'Sparkles',
    description: 'LLM picks a style during context analysis.',
  },
];

export const TRANSCRIPTION_PROVIDERS = [
  { id: 'openrouter', name: 'OpenRouter (MAI-Transcribe 2)' },
  { id: 'assemblyai', name: 'AssemblyAI (legacy)' },
  { id: 'openai', name: 'OpenAI + PyAnnote (legacy)' },
  { id: 'whisperx', name: 'WhisperX (Local, legacy)' },
];

export const OPENROUTER_STT_DEFAULT_MODEL = 'microsoft/mai-transcribe-2';
export const OPENROUTER_STT_MODEL_OPTIONS = [
  'microsoft/mai-transcribe-2',
  'elevenlabs/scribe-v2',
  'deepgram/nova-3',
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
    personaId: 'none',
    keepBackground: false,
    llmProvider: 'openrouter',
    llmModelName: '~google/gemini-flash-latest',
    llmTemperature: 0.5,
    enableLlmEditor: false,
    enableLlmTextAdjustment: false,
    editorLlmProvider: 'openrouter',
    editorModelName: 'openai/gpt-5.6-terra',
    editorTemperature: 1.0,
    editorReasoningEffort: 'xhigh',
    refinementLlmProvider: 'openrouter',
    refinementModelName: 'openai/gpt-6-sol',
    refinementTemperature: 1.0,
    ttsSystem: 'gemini38',
    ttsTransport: 'openrouter',
    ttsModel: 'google/gemini-3.8-flash-lite-tts',
    ttsFallbackModel: 'google/gemini-3.8-flash-lite-tts',
    ttsStyle: 'podcast',
    voiceAutoSelection: true,
    enableEmotionAnalysis: false,
    enableLaughterDetection: false,
    laughterMode: 'dub',
    enableEmotionEnrichment: false,
    enableContentValidation: false,
    enableContextStyle: false,
    contextStyleMaxChars: 140,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    normalizeAudio: false,
    useTwoPassEncoding: true,
    videoQualityPreset: '720p',
    maxWorkers: 4,
    pauseRemoval: 'disabled',
  },
  {
    id: 'hq',
    name: 'HQ',
    description: '~2x video duration',
    icon: '✨',
    personaId: 'none',
    keepBackground: false,
    llmProvider: 'openrouter',
    llmModelName: '~google/gemini-flash-latest',
    llmTemperature: 0.5,
    enableLlmEditor: false,
    enableLlmTextAdjustment: false,
    editorLlmProvider: 'openrouter',
    editorModelName: 'openai/gpt-5.6-terra',
    editorTemperature: 1.0,
    editorReasoningEffort: 'xhigh',
    refinementLlmProvider: 'openrouter',
    refinementModelName: 'openai/gpt-6-sol',
    refinementTemperature: 1.0,
    ttsSystem: 'gemini38',
    ttsTransport: 'openrouter',
    ttsModel: 'google/gemini-3.8-flash-lite-tts',
    ttsFallbackModel: 'google/gemini-3.8-flash-lite-tts',
    ttsStyle: 'podcast',
    voiceAutoSelection: true,
    enableEmotionAnalysis: false,
    enableLaughterDetection: false,
    laughterMode: 'dub',
    enableEmotionEnrichment: false,
    enableContentValidation: true,
    enableContextStyle: true,
    contextStyleMaxChars: 140,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    normalizeAudio: false,
    useTwoPassEncoding: true,
    videoQualityPreset: '1080p',
    maxWorkers: 4,
    pauseRemoval: 'disabled',
  },
  {
    id: 'ultra',
    name: 'Ultra',
    description: '~4x video duration',
    icon: '💎',
    personaId: 'normal',
    keepBackground: false,
    llmProvider: 'openrouter',
    llmModelName: '~google/gemini-flash-latest',
    llmTemperature: 0.5,
    enableLlmEditor: false,
    enableLlmTextAdjustment: true,
    editorLlmProvider: 'openrouter',
    editorModelName: 'openai/gpt-5.6-sol',
    editorTemperature: 1.0,
    editorReasoningEffort: 'xhigh',
    refinementLlmProvider: 'openrouter',
    refinementModelName: 'google/gemini-4-carbon',
    refinementTemperature: 0.8,
    ttsSystem: 'gemini38',
    ttsTransport: 'openrouter',
    ttsModel: 'google/gemini-3.8-flash-tts',
    ttsFallbackModel: 'google/gemini-3.8-flash-lite-tts',
    ttsStyle: 'podcast',
    voiceAutoSelection: true,
    enableEmotionAnalysis: true,
    enableLaughterDetection: false,
    laughterMode: 'dub',
    enableEmotionEnrichment: true,
    enableContentValidation: true,
    enableContextStyle: true,
    contextStyleMaxChars: 140,
    dubbedVolume: 1.0,
    backgroundVolume: 0.56,
    backgroundVoiceVolume: 0.0,
    normalizeAudio: false,
    useTwoPassEncoding: true,
    videoQualityPreset: 'original',
    maxWorkers: 4,
    pauseRemoval: 'disabled',
    videoMinterpolateThreshold: 1.0,
  },
];
