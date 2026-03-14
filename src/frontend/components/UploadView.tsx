
import React, { useState, useMemo, useRef, useEffect } from 'react';
import { ChevronDown, ChevronRight, Speaker, Loader2, Info, Brain, Plus, Trash2, Users, Play, Pause, RotateCcw } from 'lucide-react';
import { AppConfig, Persona, PresetId, LlmProvider, VideoQualityPreset } from '../types';
import { LANGUAGES, PRESETS, LLM_PROVIDERS, TTS_PROVIDERS, TRANSCRIPTION_PROVIDERS, WHISPER_MODELS } from '../constants';
import api, { VoiceResponse } from '../api';

const TTS_DEFAULT_MODELS: Record<string, string> = {
  gemini: 'gemini-2.5-flash-preview-tts',
  openai: 'gpt-4o-mini-tts',
  minimax: 'speech-02-hd',
};

const VIDEO_QUALITY_OPTIONS: Array<{ value: VideoQualityPreset; label: string }> = [
  { value: '720p', label: 'Fast - 720p' },
  { value: '1080p', label: 'HQ - 1080p' },
  { value: 'original', label: 'Ultra - Original quality' },
];

interface TimeRangeRow {
  id: string;
  start: string;
  end: string;
}

const createRangeRow = (start = '', end = ''): TimeRangeRow => ({
  id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  start,
  end,
});

const parseTimeRange = (range: string): TimeRangeRow => {
  const separatorIndex = range.indexOf('-');
  if (separatorIndex === -1) {
    return createRangeRow(range.trim(), '');
  }

  return createRangeRow(
    range.slice(0, separatorIndex).trim(),
    range.slice(separatorIndex + 1).trim(),
  );
};

const InfoTip: React.FC<{ text: string }> = ({ text }) => (
  <span className="relative group ml-1 cursor-help">
    <Info className="w-3 h-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
    <span className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 px-2 py-1 text-[10px] text-zinc-200 bg-zinc-800 border border-zinc-700 rounded shadow-lg whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50">
      {text}
    </span>
  </span>
);

const LANG_FLAGS: Record<string, string> = {
  auto: '🔍',
  en: '🇺🇸',
  es: '🇪🇸',
  fr: '🇫🇷',
  de: '🇩🇪',
  it: '🇮🇹',
  pt: '🇵🇹',
  ru: '🇷🇺',
  ja: '🇯🇵',
  zh: '🇨🇳',
  ko: '🇰🇷',
  ar: '🇸🇦',
  hi: '🇮🇳',
  tr: '🇹🇷',
  pl: '🇵🇱',
  nl: '🇳🇱',
  uk: '🇺🇦',
};

interface UploadViewProps {
  config: AppConfig;
  personas: Persona[];
  voices: VoiceResponse[];
  projectId: string;
  videoFile: File | null;
  isUploading: boolean;
  uploadProgress: number;
  onConfigChange: (cfg: Partial<AppConfig>) => void;
  onNext: () => void;
  onResetAndStart: () => void;
  onOpenSettings: () => void;
}

export const UploadView: React.FC<UploadViewProps> = ({ 
  config, 
  personas,
  voices,
  projectId,
  videoFile,
  isUploading,
  uploadProgress,
  onConfigChange, 
  onNext,
  onResetAndStart,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExtra, setShowExtra] = useState(false);
  const [newSpeakerName, setNewSpeakerName] = useState('');
  const [newSpeakerPrompt, setNewSpeakerPrompt] = useState('');
  const [newSpeakerVoiceName, setNewSpeakerVoiceName] = useState('');
  const [newSpeakerVoiceId, setNewSpeakerVoiceId] = useState('');
  const [playingSampleKey, setPlayingSampleKey] = useState<string | null>(null);
  const [showStartMenu, setShowStartMenu] = useState(false);
  const [originalAudioRanges, setOriginalAudioRanges] = useState<TimeRangeRow[]>(
    () => (config.keepOriginalAudioRanges || []).map(parseTimeRange)
  );
  
  // Use ref to track latest speakerTtsPrompts to avoid race conditions with async state updates
  const speakerPromptsRef = useRef<Record<string, string>>(config.speakerTtsPrompts || {});
  const speakerVoiceMappingsRef = useRef<Record<string, string>>(config.speakerVoiceMappings || {});
  const sampleAudioRef = useRef<HTMLAudioElement | null>(null);
  const startMenuRef = useRef<HTMLDivElement | null>(null);
  
  useEffect(() => {
    speakerPromptsRef.current = config.speakerTtsPrompts || {};
  }, [config.speakerTtsPrompts]);

  useEffect(() => {
    speakerVoiceMappingsRef.current = config.speakerVoiceMappings || {};
  }, [config.speakerVoiceMappings]);

  useEffect(() => {
    setOriginalAudioRanges((config.keepOriginalAudioRanges || []).map(parseTimeRange));
  }, [projectId]);

  const handleAddSpeakerPrompt = () => {
    if (!newSpeakerName.trim()) return;
    const currentPrompts = { ...speakerPromptsRef.current };
    currentPrompts[newSpeakerName.trim()] = newSpeakerPrompt.trim();
    speakerPromptsRef.current = currentPrompts;
    onConfigChange({ speakerTtsPrompts: currentPrompts });
    setNewSpeakerName('');
    setNewSpeakerPrompt('');
  };

  const handleRemoveSpeakerPrompt = (speakerName: string) => {
    const currentPrompts = { ...speakerPromptsRef.current };
    delete currentPrompts[speakerName];
    speakerPromptsRef.current = currentPrompts;
    onConfigChange({ speakerTtsPrompts: currentPrompts });
  };

  const handleUpdateSpeakerPrompt = (speakerName: string, prompt: string) => {
    const currentPrompts = { ...speakerPromptsRef.current };
    currentPrompts[speakerName] = prompt;
    speakerPromptsRef.current = currentPrompts;
    onConfigChange({ speakerTtsPrompts: currentPrompts });
  };

  const currentPreset = useMemo(() => 
    PRESETS.find(p => p.id === (config.preset || 'hq')) || PRESETS[1],
    [config.preset]
  );

  const selectedTtsProvider = config.ttsSystem || currentPreset.ttsSystem;
  const providerVoices = useMemo(
    () => voices.filter((voice) => voice.provider === selectedTtsProvider),
    [voices, selectedTtsProvider]
  );
  const defaultProviderVoiceId = providerVoices[0]?.id || '';
  const getVoiceKey = (voice: VoiceResponse): string => `${voice.provider}:${voice.id}`;

  useEffect(() => {
    setNewSpeakerVoiceId((prev) => {
      if (prev && providerVoices.some((voice) => voice.id === prev)) {
        return prev;
      }
      return defaultProviderVoiceId;
    });
  }, [providerVoices, defaultProviderVoiceId]);

  const handleAddSpeakerVoiceMapping = () => {
    const speakerName = newSpeakerVoiceName.trim();
    const voiceId = newSpeakerVoiceId || defaultProviderVoiceId;
    if (!speakerName || !voiceId) return;

    const currentMappings = { ...speakerVoiceMappingsRef.current };
    currentMappings[speakerName] = voiceId;
    speakerVoiceMappingsRef.current = currentMappings;
    onConfigChange({ speakerVoiceMappings: currentMappings });

    setNewSpeakerVoiceName('');
    setNewSpeakerVoiceId(defaultProviderVoiceId);
  };

  const handleRemoveSpeakerVoiceMapping = (speakerName: string) => {
    const currentMappings = { ...speakerVoiceMappingsRef.current };
    delete currentMappings[speakerName];
    speakerVoiceMappingsRef.current = currentMappings;
    onConfigChange({ speakerVoiceMappings: currentMappings });
  };

  const handleUpdateSpeakerVoiceMapping = (speakerName: string, voiceId: string) => {
    const currentMappings = { ...speakerVoiceMappingsRef.current };
    currentMappings[speakerName] = voiceId;
    speakerVoiceMappingsRef.current = currentMappings;
    onConfigChange({ speakerVoiceMappings: currentMappings });
  };

  const syncOriginalAudioRanges = (rows: TimeRangeRow[]) => {
    setOriginalAudioRanges(rows);
    onConfigChange({
      keepOriginalAudioRanges: rows
        .map((row) => {
          const start = row.start.trim();
          const end = row.end.trim();
          return start && end ? `${start}-${end}` : '';
        })
        .filter(Boolean),
    });
  };

  const handleAddOriginalAudioRange = () => {
    setOriginalAudioRanges((prev) => [...prev, createRangeRow()]);
  };

  const handleUpdateOriginalAudioRange = (id: string, field: 'start' | 'end', value: string) => {
    syncOriginalAudioRanges(
      originalAudioRanges.map((row) => (
        row.id === id ? { ...row, [field]: value } : row
      ))
    );
  };

  const handleRemoveOriginalAudioRange = (id: string) => {
    syncOriginalAudioRanges(originalAudioRanges.filter((row) => row.id !== id));
  };

  const stopSamplePlayback = () => {
    if (sampleAudioRef.current) {
      sampleAudioRef.current.pause();
      sampleAudioRef.current.currentTime = 0;
      sampleAudioRef.current = null;
    }
    setPlayingSampleKey(null);
  };

  const handleToggleVoiceSample = (voice?: VoiceResponse) => {
    if (!voice?.preview_url) return;

    const voiceKey = getVoiceKey(voice);
    if (playingSampleKey === voiceKey) {
      stopSamplePlayback();
      return;
    }

    stopSamplePlayback();

    const audio = new Audio(voice.preview_url);
    sampleAudioRef.current = audio;
    setPlayingSampleKey(voiceKey);
    audio.onended = () => stopSamplePlayback();
    audio.onerror = () => stopSamplePlayback();
    audio.play().catch(() => stopSamplePlayback());
  };

  useEffect(() => {
    return () => stopSamplePlayback();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!startMenuRef.current) return;
      if (!startMenuRef.current.contains(event.target as Node)) {
        setShowStartMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const videoPreviewUrl = useMemo(() => {
    if (videoFile) {
      return URL.createObjectURL(videoFile);
    }
    if (!isUploading && projectId) {
      return api.getVideoUrl(projectId);
    }
    return null;
  }, [videoFile, isUploading, projectId]);

  React.useEffect(() => {
    return () => {
      if (videoPreviewUrl && videoFile) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
    };
  }, [videoPreviewUrl, videoFile]);

  const canProceed = !isUploading && (videoFile || projectId);

  return (
    <div className="h-full w-full overflow-y-auto p-6 animate-in fade-in duration-500">
      <div className="w-full max-w-2xl mx-auto flex flex-col items-center gap-6">
        
        {/* Video Preview */}
        <div className="w-full rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800">
          {isUploading ? (
            <div className="aspect-video flex flex-col items-center justify-center gap-4">
              <div className="relative">
                <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-bold text-blue-400">{Math.round(uploadProgress)}%</span>
                </div>
              </div>
              <div className="text-center space-y-2 flex flex-col items-center">
                <p className="text-white font-medium text-sm">Uploading video...</p>
                <div className="w-40 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-500 transition-all duration-300 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            </div>
          ) : videoPreviewUrl ? (
            <video 
              src={videoPreviewUrl}
              className="w-full aspect-video object-contain bg-black"
              controls
              preload="metadata"
            />
          ) : (
            <div className="aspect-video flex flex-col items-center justify-center gap-2 text-zinc-500">
              <p className="text-sm">No video selected</p>
            </div>
          )}
        </div>

        {/* Main Controls: Target Language + Preset */}
        <div className="w-full flex items-center gap-4">
          {/* Target Language */}
          <div className="flex items-center gap-3 flex-1">
            <span className="text-sm text-zinc-400 whitespace-nowrap">Translate to</span>
            <div className="flex-1 relative">
              <select 
                value={config.targetLang}
                onChange={(e) => onConfigChange({ targetLang: e.target.value })}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 text-sm text-white appearance-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 outline-none"
              >
                {LANGUAGES.filter(lang => lang.code !== 'auto').map(lang => (
                  <option key={lang.code} value={lang.code}>{LANG_FLAGS[lang.code]} {lang.name}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-3.5 w-4 h-4 text-zinc-500 pointer-events-none" />
            </div>
          </div>

          {/* Preset Selector */}
          <div className="flex bg-zinc-900 border border-zinc-700 rounded-lg p-1 gap-1">
            {PRESETS.map((preset) => {
              const isSelected = (config.preset || 'hq') === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={async () => {
                    const presetFromApi = await api.getPreset(preset.id).catch(() => null);
                    const updates: Partial<AppConfig> = { 
                      preset: preset.id as PresetId,
                      personaId: preset.id === 'fast' ? 'none' : 'normal',
                      keepBackground: presetFromApi?.keepBackground ?? preset.keepBackground ?? false,
                      llmProvider: (presetFromApi?.llmProvider as LlmProvider) ?? preset.llmProvider,
                      llmModelName: presetFromApi?.llmModelName ?? preset.llmModelName,
                      llmTemperature: presetFromApi?.llmTemperature ?? preset.llmTemperature,
                      refinementLlmProvider: (presetFromApi?.refinementLlmProvider as LlmProvider | undefined) ?? preset.refinementLlmProvider,
                      refinementModelName: presetFromApi?.refinementModelName ?? preset.refinementModelName,
                      refinementTemperature: presetFromApi?.refinementTemperature ?? preset.refinementTemperature,
                      ttsSystem: presetFromApi?.ttsSystem ?? preset.ttsSystem,
                      ttsModel: presetFromApi?.ttsModel ?? preset.ttsModel,
                      ttsPromptPrefix: presetFromApi?.ttsPromptPrefix ?? preset.ttsPromptPrefix,
                      voiceAutoSelection: presetFromApi?.voiceAutoSelection ?? preset.voiceAutoSelection,
                      enableEmotionEnrichment: presetFromApi?.enableEmotionEnrichment ?? preset.enableEmotionEnrichment,
                      dubbedVolume: presetFromApi?.dubbedVolume ?? preset.dubbedVolume,
                      backgroundVolume: presetFromApi?.backgroundVolume ?? preset.backgroundVolume,
                      useTwoPassEncoding: presetFromApi?.useTwoPassEncoding ?? preset.useTwoPassEncoding,
                      videoQualityPreset: presetFromApi?.videoQualityPreset ?? preset.videoQualityPreset,
                      maxWorkers: presetFromApi?.maxWorkers ?? preset.maxWorkers,
                      pauseRemoval: presetFromApi?.pauseRemoval ?? preset.pauseRemoval,
                    };
                    onConfigChange(updates);
                  }}
                  className={`
                    flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-all
                    ${isSelected 
                      ? 'bg-brand-600 text-white shadow-md' 
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-800'}
                  `}
                  title={preset.description}
                >
                  <span>{preset.icon}</span>
                  <span>{preset.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Advanced Options */}
        <div className="w-full">
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {showAdvanced ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            Advanced Options
          </button>

          {showAdvanced && (
            <div className="mt-4 p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-4 animate-in slide-in-from-top-2">
              <div className="grid grid-cols-4 gap-3 items-start">
                {/* Source Language */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400 h-4 flex items-center">Source Language</label>
                  <div className="relative">
                    <select 
                      value={config.sourceLang}
                      onChange={(e) => onConfigChange({ sourceLang: e.target.value })}
                      className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white appearance-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 outline-none"
                    >
                      {LANGUAGES.map(lang => (
                        <option key={lang.code} value={lang.code}>{LANG_FLAGS[lang.code]} {lang.name}</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-2.5 w-4 h-4 text-zinc-500 pointer-events-none" />
                  </div>
                </div>

                {/* Expected Speakers */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400 h-4 flex items-center gap-1.5">
                    <Speaker className="w-3 h-3" /> Speakers
                  </label>
                  <input 
                    type="number" 
                    min="1"
                    placeholder="Auto"
                    value={config.speakerCount || ''}
                    onChange={(e) => onConfigChange({ speakerCount: e.target.value ? parseInt(e.target.value) : undefined })}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                  />
                </div>

                {/* Start Time */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400 h-4 flex items-center">Start (sec)</label>
                  <input 
                    type="number"
                    min="0"
                    step="1"
                    placeholder="0"
                    value={config.startTime ?? ''}
                    onChange={(e) => onConfigChange({ startTime: e.target.value ? parseFloat(e.target.value) : undefined })}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                  />
                </div>

                {/* Duration */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400 h-4 flex items-center">Duration (sec)</label>
                  <input 
                    type="number"
                    min="1"
                    step="1"
                    placeholder="Full"
                    value={config.duration ?? ''}
                    onChange={(e) => onConfigChange({ duration: e.target.value ? parseFloat(e.target.value) : undefined })}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                  />
                </div>
              </div>

              {/* Translator Persona */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-400">Translator Persona</label>
                <div className="relative">
                  <select 
                    value={config.personaId}
                    onChange={(e) => onConfigChange({ personaId: e.target.value })}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white appearance-none outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                  >
                    {personas.map(p => (
                      <option key={p.id} value={p.id}>{p.name} — {p.description}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-2.5 w-4 h-4 text-zinc-500 pointer-events-none" />
                </div>
              </div>

              {/* Keep Background Audio */}
              <label className="flex items-center gap-2 cursor-pointer group">
                <input 
                  type="checkbox" 
                  checked={config.keepBackground}
                  onChange={(e) => onConfigChange({ keepBackground: e.target.checked })}
                  className="w-4 h-4 rounded border-zinc-700 bg-zinc-900 text-brand-600 focus:ring-brand-500 focus:ring-offset-zinc-900"
                />
                <span className="text-xs text-zinc-300 group-hover:text-white transition-colors">Keep background audio</span>
              </label>

              {/* Extra Settings Toggle */}
              <div className="pt-2 border-t border-zinc-800/50">
                <button 
                  onClick={() => setShowExtra(!showExtra)}
                  className="flex items-center gap-1.5 text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
                >
                  {showExtra ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  Extra
                  <Brain className="w-2.5 h-2.5" />
                  <Brain className="w-2.5 h-2.5" />
                </button>

                {showExtra && (
                  <div className="mt-3 space-y-4 animate-in slide-in-from-top-1">
                    {/* Transcription Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Transcription</h4>
                      <div className="grid grid-cols-2 gap-2 max-w-[320px]">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Provider
                            <InfoTip text="Transcription service (AssemblyAI, OpenAI+PyAnnote, or local WhisperX)" />
                          </label>
                          <select 
                            value={config.transcriptionSystem || 'assemblyai'}
                            onChange={(e) => onConfigChange({ transcriptionSystem: e.target.value as 'assemblyai' | 'openai' | 'whisperx' })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                          >
                            {TRANSCRIPTION_PROVIDERS.map(p => (
                              <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="space-y-1">
                          <label className={`text-[10px] flex items-center ${(config.transcriptionSystem || 'assemblyai') !== 'assemblyai' ? 'text-zinc-500' : 'text-zinc-600'}`}>
                            Whisper Model
                            <InfoTip text="Model size for WhisperX/OpenAI transcription" />
                          </label>
                          <select 
                            value={config.whisperModel || 'large-v3'}
                            onChange={(e) => onConfigChange({ whisperModel: e.target.value })}
                            disabled={(config.transcriptionSystem || 'assemblyai') === 'assemblyai'}
                            className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none ${(config.transcriptionSystem || 'assemblyai') === 'assemblyai' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          >
                            {WHISPER_MODELS.map(m => (
                              <option key={m.id} value={m.id}>{m.name}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Translation Model Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Translation Model</h4>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Provider
                            <InfoTip text="AI provider for translation (gemini or openrouter)" />
                          </label>
                          <select 
                            value={config.llmProvider || currentPreset.llmProvider}
                            onChange={(e) => onConfigChange({ llmProvider: e.target.value as LlmProvider })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                          >
                            {LLM_PROVIDERS.map(p => (
                              <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Model
                            <InfoTip text="Model name for translation (e.g., gemini-2.5-pro)" />
                          </label>
                          <input 
                            type="text"
                            value={config.llmModelName || currentPreset.llmModelName}
                            onChange={(e) => onConfigChange({ llmModelName: e.target.value })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="gemini-2.5-pro"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Temp
                            <InfoTip text="Creativity level 0.0-2.0 (lower = more conservative)" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            max="2"
                            value={config.llmTemperature ?? currentPreset.llmTemperature}
                            onChange={(e) => onConfigChange({ llmTemperature: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Refinement Model Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Refinement Model</h4>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Provider
                            <InfoTip text="AI provider for style refinement pass" />
                          </label>
                          <select 
                            value={config.refinementLlmProvider || currentPreset.refinementLlmProvider || ''}
                            onChange={(e) => onConfigChange({ refinementLlmProvider: e.target.value as LlmProvider || undefined })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                          >
                            <option value="">None</option>
                            {LLM_PROVIDERS.map(p => (
                              <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Model
                            <InfoTip text="Model for polishing translations" />
                          </label>
                          <input 
                            type="text"
                            value={config.refinementModelName || currentPreset.refinementModelName || ''}
                            onChange={(e) => onConfigChange({ refinementModelName: e.target.value || undefined })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="gemini-2.5-pro"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Temp
                            <InfoTip text="Refinement creativity level" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            max="2"
                            value={config.refinementTemperature ?? currentPreset.refinementTemperature}
                            onChange={(e) => onConfigChange({ refinementTemperature: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[10px] text-zinc-500 flex items-center">
                          Translation Prompt Prefix
                          <InfoTip text="Custom context to prepend to translation prompts" />
                        </label>
                        <textarea 
                          value={config.translationPromptPrefix || ''}
                          onChange={(e) => onConfigChange({ translationPromptPrefix: e.target.value || undefined })}
                          className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none resize-none"
                          rows={2}
                          placeholder="Optional: Add context for translation..."
                        />
                      </div>
                    </div>

                    {/* TTS Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">TTS Settings</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Provider
                            <InfoTip text="TTS provider (openai, gemini, minimax)" />
                          </label>
                          <select 
                            value={selectedTtsProvider}
                            onChange={(e) => {
                              const provider = e.target.value;
                              onConfigChange({ 
                                ttsSystem: provider,
                                ttsModel: TTS_DEFAULT_MODELS[provider] || '',
                                ttsPromptPrefix: provider === 'gemini' ? config.ttsPromptPrefix : undefined,
                              });
                            }}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                          >
                            {TTS_PROVIDERS.map(p => (
                              <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Model
                            <InfoTip text="TTS model name (e.g., gemini-2.5-flash-preview-tts)" />
                          </label>
                          <input 
                            type="text"
                            value={config.ttsModel || currentPreset.ttsModel || ''}
                            onChange={(e) => onConfigChange({ ttsModel: e.target.value || undefined })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="Default model"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input 
                            type="checkbox" 
                            checked={config.voiceAutoSelection ?? currentPreset.voiceAutoSelection ?? true}
                            onChange={(e) => onConfigChange({ voiceAutoSelection: e.target.checked })}
                            className="w-3 h-3 rounded border-zinc-700 bg-zinc-900 text-brand-600"
                          />
                          <span className="text-[10px] text-zinc-400">Auto Voice</span>
                          <InfoTip text="Automatically match voices to speakers based on audio" />
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input 
                            type="checkbox" 
                            checked={config.enableEmotionEnrichment ?? currentPreset.enableEmotionEnrichment ?? false}
                            onChange={(e) => onConfigChange({ enableEmotionEnrichment: e.target.checked })}
                            className="w-3 h-3 rounded border-zinc-700 bg-zinc-900 text-brand-600"
                          />
                          <span className="text-[10px] text-zinc-400">Enrich emotions</span>
                          <InfoTip text="Enrich TTS prompts with detected emotions" />
                        </label>
                      </div>
                      <div className="space-y-1">
                        <label className={`text-[10px] flex items-center ${selectedTtsProvider === 'gemini' ? 'text-zinc-500' : 'text-zinc-600'}`}>
                          TTS Prompt Prefix
                          <InfoTip text="Global instruction prefix for TTS (Gemini only)" />
                        </label>
                        <input 
                          type="text"
                          value={selectedTtsProvider === 'gemini' ? (config.ttsPromptPrefix || currentPreset.ttsPromptPrefix || '') : ''}
                          onChange={(e) => onConfigChange({ ttsPromptPrefix: e.target.value || undefined })}
                          disabled={selectedTtsProvider !== 'gemini'}
                          className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${selectedTtsProvider !== 'gemini' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          placeholder={selectedTtsProvider === 'gemini' ? "Speak with natural conversational energy..." : "Only available for Gemini"}/>
                      </div>

                      {/* Per-Speaker Voice Mapping */}
                      <div className="space-y-2 pt-2 border-t border-zinc-800/30">
                        <label className="text-[10px] text-zinc-500 flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          Per-Speaker Voice Mapping
                          <InfoTip text="Pin a voice for each speaker using the selected TTS provider. Speaker names are matched after transcription." />
                        </label>

                        {Object.entries(config.speakerVoiceMappings || {}).length > 0 && (
                          <div className="space-y-1.5">
                            {Object.entries(config.speakerVoiceMappings || {}).map(([speaker, voiceId]) => {
                              const selectedVoice = providerVoices.find((voice) => voice.id === voiceId);
                              const hasProviderVoice = providerVoices.some((voice) => voice.id === voiceId);
                              const canPreview = Boolean(selectedVoice?.preview_url);
                              const isPlaying = selectedVoice ? playingSampleKey === getVoiceKey(selectedVoice) : false;

                              return (
                                <div key={speaker} className="flex items-center gap-2 bg-zinc-950/50 rounded p-1.5">
                                  <span className="text-[10px] font-medium text-zinc-400 min-w-[70px] shrink-0">{speaker}</span>
                                  <select
                                    value={voiceId}
                                    onChange={(e) => handleUpdateSpeakerVoiceMapping(speaker, e.target.value)}
                                    className="flex-1 bg-zinc-900 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                                  >
                                    {!hasProviderVoice && (
                                      <option value={voiceId}>{voiceId} (from another provider)</option>
                                    )}
                                    {providerVoices.map((voice) => (
                                      <option key={voice.id} value={voice.id}>
                                        {voice.name}
                                      </option>
                                    ))}
                                  </select>
                                  <button
                                    onClick={() => handleToggleVoiceSample(selectedVoice)}
                                    disabled={!canPreview}
                                    className="p-1 text-zinc-500 hover:text-brand-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                    title={canPreview ? (isPlaying ? 'Stop sample' : 'Play sample') : 'Sample unavailable'}
                                  >
                                    {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                                  </button>
                                  <button
                                    onClick={() => handleRemoveSpeakerVoiceMapping(speaker)}
                                    className="p-1 text-zinc-500 hover:text-red-400 transition-colors"
                                    title="Remove mapping"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </button>
                                </div>
                              );
                            })}
                          </div>
                        )}

                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={newSpeakerVoiceName}
                            onChange={(e) => setNewSpeakerVoiceName(e.target.value)}
                            className="w-[90px] bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="Speaker"
                          />
                          <select
                            value={newSpeakerVoiceId}
                            onChange={(e) => setNewSpeakerVoiceId(e.target.value)}
                            disabled={providerVoices.length === 0}
                            className={`flex-1 bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${providerVoices.length === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
                          >
                            {providerVoices.length === 0 ? (
                              <option value="">No voices for {selectedTtsProvider}</option>
                            ) : (
                              providerVoices.map((voice) => (
                                <option key={voice.id} value={voice.id}>
                                  {voice.name}
                                </option>
                              ))
                            )}
                          </select>
                          <button
                            onClick={() => handleToggleVoiceSample(providerVoices.find((voice) => voice.id === newSpeakerVoiceId))}
                            disabled={!providerVoices.find((voice) => voice.id === newSpeakerVoiceId)?.preview_url}
                            className="p-1 text-zinc-500 hover:text-brand-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                            title="Play selected voice sample"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                          <button
                            onClick={handleAddSpeakerVoiceMapping}
                            disabled={!newSpeakerVoiceName.trim() || !newSpeakerVoiceId}
                            className="p-1 text-zinc-500 hover:text-brand-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                            title="Add speaker voice mapping"
                          >
                            <Plus className="w-4 h-4" />
                          </button>
                        </div>
                        <p className="text-[9px] text-zinc-600">
                          Example speaker names: SPEAKER_00, SPEAKER_01
                        </p>
                      </div>

                      {/* Per-Speaker TTS Prompts */}
                      <div className={`space-y-2 pt-2 border-t border-zinc-800/30 ${selectedTtsProvider !== 'gemini' ? 'opacity-40 pointer-events-none' : ''}`}>
                        <label className="text-[10px] text-zinc-500 flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          Per-Speaker TTS Prompts
                          <InfoTip text="Set individual TTS style prompts for specific speakers. Speaker names will be matched after transcription." />
                        </label>
                        
                        {/* Existing speaker prompts */}
                        {Object.entries(config.speakerTtsPrompts || {}).length > 0 && (
                          <div className="space-y-1.5">
                            {Object.entries(config.speakerTtsPrompts || {}).map(([speaker, prompt]) => (
                              <div key={speaker} className="flex items-center gap-2 bg-zinc-950/50 rounded p-1.5">
                                <span className="text-[10px] font-medium text-zinc-400 min-w-[60px] shrink-0">{speaker}</span>
                                <input
                                  type="text"
                                  value={prompt}
                                  onChange={(e) => handleUpdateSpeakerPrompt(speaker, e.target.value)}
                                  className="flex-1 bg-zinc-900 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                                  placeholder="TTS style for this speaker..."
                                />
                                <button
                                  onClick={() => handleRemoveSpeakerPrompt(speaker)}
                                  className="p-1 text-zinc-500 hover:text-red-400 transition-colors"
                                  title="Remove"
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Add new speaker prompt */}
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={newSpeakerName}
                            onChange={(e) => setNewSpeakerName(e.target.value)}
                            className="w-[80px] bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="Speaker"
                          />
                          <input
                            type="text"
                            value={newSpeakerPrompt}
                            onChange={(e) => setNewSpeakerPrompt(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAddSpeakerPrompt()}
                            className="flex-1 bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                            placeholder="TTS prompt for this speaker..."
                          />
                          <button
                            onClick={handleAddSpeakerPrompt}
                            disabled={!newSpeakerName.trim()}
                            className="p-1 text-zinc-500 hover:text-brand-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                            title="Add speaker prompt"
                          >
                            <Plus className="w-4 h-4" />
                          </button>
                        </div>
                        <p className="text-[9px] text-zinc-600">
                          Speaker names (e.g., "SPEAKER_00", "SPEAKER_01") will be matched after transcription
                        </p>
                      </div>
                    </div>

                    {/* Audio Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Audio</h4>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Dubbed Vol
                            <InfoTip text="Volume multiplier for translated audio (1.0 = normal)" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            max="2"
                            value={config.dubbedVolume ?? currentPreset.dubbedVolume ?? 1.0}
                            onChange={(e) => onConfigChange({ dubbedVolume: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            BG Vol
                            <InfoTip text="Volume for background when kept (0.56 ≈ -5dB)" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            max="1"
                            value={config.backgroundVolume ?? currentPreset.backgroundVolume ?? 0.56}
                            onChange={(e) => onConfigChange({ backgroundVolume: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                        <label className="flex items-center gap-1.5 cursor-pointer pt-4">
                          <input 
                            type="checkbox" 
                            checked={config.useTwoPassEncoding ?? currentPreset.useTwoPassEncoding ?? true}
                            onChange={(e) => onConfigChange({ useTwoPassEncoding: e.target.checked })}
                            className="w-3 h-3 rounded border-zinc-700 bg-zinc-900 text-brand-600"
                          />
                          <span className="text-[10px] text-zinc-400">2-Pass</span>
                          <InfoTip text="Higher quality video encoding (slower)" />
                        </label>
                      </div>
                      <div className="space-y-1 max-w-[240px]">
                        <label className="text-[10px] text-zinc-500 flex items-center">
                          Video Quality
                          <InfoTip text="Final output resolution cap: Fast (720p), HQ (1080p), Ultra (keep original)." />
                        </label>
                        <select
                          value={config.videoQualityPreset || currentPreset.videoQualityPreset || '1080p'}
                          onChange={(e) => onConfigChange({ videoQualityPreset: e.target.value as VideoQualityPreset })}
                          className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                        >
                          {VIDEO_QUALITY_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </div>
                      <div className="space-y-2 pt-2 border-t border-zinc-800/30">
                        <div className="flex items-center justify-between gap-2">
                          <label className="text-[10px] text-zinc-500 flex items-center gap-1">
                            Keep Original Audio Ranges
                            <InfoTip text="Keep the source speech instead of dubbed audio for selected ranges. Time format: SS, MM:SS, or HH:MM:SS." />
                          </label>
                          <button
                            onClick={handleAddOriginalAudioRange}
                            className="p-1 text-zinc-500 hover:text-brand-400 transition-colors"
                            title="Add original audio range"
                          >
                            <Plus className="w-3.5 h-3.5" />
                          </button>
                        </div>

                        {originalAudioRanges.length > 0 ? (
                          <div className="space-y-1.5">
                            {originalAudioRanges.map((range) => (
                              <div key={range.id} className="flex items-center gap-2 bg-zinc-950/50 rounded p-1.5">
                                <input
                                  type="text"
                                  value={range.start}
                                  onChange={(e) => handleUpdateOriginalAudioRange(range.id, 'start', e.target.value)}
                                  className="w-[110px] bg-zinc-900 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                                  placeholder="00:10"
                                />
                                <span className="text-[10px] text-zinc-600">to</span>
                                <input
                                  type="text"
                                  value={range.end}
                                  onChange={(e) => handleUpdateOriginalAudioRange(range.id, 'end', e.target.value)}
                                  className="w-[110px] bg-zinc-900 border border-zinc-700/50 rounded px-2 py-1 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                                  placeholder="00:24.5"
                                />
                                <button
                                  onClick={() => handleRemoveOriginalAudioRange(range.id)}
                                  className="p-1 text-zinc-500 hover:text-red-400 transition-colors"
                                  title="Remove range"
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-[9px] text-zinc-600">
                            Add ranges where the original audio should stay without translation.
                          </p>
                        )}

                        <p className="text-[9px] text-zinc-600">
                          Examples: 12.5 to 18, 01:10 to 01:24, 00:10:05 to 00:10:22
                        </p>
                      </div>
                    </div>

                    {/* Processing Section */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Processing</h4>
                      <div className="space-y-1 max-w-[120px]">
                        <label className="text-[10px] text-zinc-500 flex items-center">
                          Workers
                          <InfoTip text="Parallel TTS workers (higher = faster, more API usage)" />
                        </label>
                        <input 
                          type="number"
                          min="1"
                          max="16"
                          value={config.maxWorkers ?? currentPreset.maxWorkers ?? 4}
                          onChange={(e) => onConfigChange({ maxWorkers: parseInt(e.target.value) })}
                          className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                        />
                      </div>
                    </div>

                    {/* Segment Settings */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Segments</h4>
                      <div className="grid grid-cols-4 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Merge Gap 1
                            <InfoTip text="Max gap (sec) to merge segments after diarization" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            value={config.postDiarizationMergeGap ?? 0.3}
                            onChange={(e) => onConfigChange({ postDiarizationMergeGap: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Merge Gap 2
                            <InfoTip text="Max gap (sec) to merge segments after translation" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            value={config.postTranslationMergeGap ?? 1.5}
                            onChange={(e) => onConfigChange({ postTranslationMergeGap: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Min Dur
                            <InfoTip text="Minimum segment duration in seconds" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0.1"
                            value={config.minSegmentDuration ?? 0.5}
                            onChange={(e) => onConfigChange({ minSegmentDuration: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Max Dur
                            <InfoTip text="Maximum segment duration in seconds" />
                          </label>
                          <input 
                            type="number"
                            step="1"
                            min="5"
                            value={config.maxSegmentDuration ?? 60}
                            onChange={(e) => onConfigChange({ maxSegmentDuration: parseFloat(e.target.value) })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Audio/Video Sync */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Audio/Video Sync</h4>
                      <div className="space-y-1 max-w-[200px]">
                        <label className="text-[10px] text-zinc-500 flex items-center">
                          Stretch Mode
                          <InfoTip text="How to sync timing: Audio (TTS speed only), Audio+Video (hybrid), Video (video speed only)" />
                        </label>
                        <select 
                          value={config.segmentStretch || 'audio_and_video'}
                          onChange={(e) => onConfigChange({ segmentStretch: e.target.value as 'audio' | 'audio_and_video' | 'video' })}
                          className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                        >
                          <option value="audio">Audio only</option>
                          <option value="audio_and_video">Audio + Video</option>
                          <option value="video">Video only</option>
                        </select>
                      </div>
                      
                      {/* Audio Comfort Zone */}
                      <div className="pt-1">
                        <label className={`text-[10px] font-medium flex items-center gap-1 ${config.segmentStretch === 'video' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                          🔊 Audio Speed Range
                          <InfoTip text="TTS playback speed comfort zone (0.85-1.15 means ±15%)" />
                        </label>
                        <div className="grid grid-cols-2 gap-2 mt-1">
                          <div className="space-y-0.5">
                            <label className={`text-[10px] ${config.segmentStretch === 'video' ? 'text-zinc-600' : 'text-zinc-500'}`}>Slowdown</label>
                            <input 
                              type="number"
                              step="0.05"
                              min="0.5"
                              max="1"
                              disabled={config.segmentStretch === 'video'}
                              value={config.comfortMinAdjustmentRatio ?? 0.85}
                              onChange={(e) => onConfigChange({ comfortMinAdjustmentRatio: parseFloat(e.target.value) })}
                              className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.segmentStretch === 'video' ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                          </div>
                          <div className="space-y-0.5">
                            <label className={`text-[10px] ${config.segmentStretch === 'video' ? 'text-zinc-600' : 'text-zinc-500'}`}>Speedup</label>
                            <input 
                              type="number"
                              step="0.05"
                              min="1"
                              max="2"
                              disabled={config.segmentStretch === 'video'}
                              value={config.comfortMaxAdjustmentRatio ?? 1.15}
                              onChange={(e) => onConfigChange({ comfortMaxAdjustmentRatio: parseFloat(e.target.value) })}
                              className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.segmentStretch === 'video' ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Pause Processing */}
                    <div className="space-y-2">
                      <h4 className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">Pause Processing</h4>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="space-y-1">
                          <label className="text-[10px] text-zinc-500 flex items-center">
                            Mode
                            <InfoTip text="Cut: remove pauses" />
                          </label>
                          <select 
                            value={config.pauseRemoval || 'disabled'}
                            onChange={(e) => onConfigChange({ pauseRemoval: e.target.value as 'cut' | 'disabled' })}
                            className="w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white appearance-none focus:ring-1 focus:ring-brand-500/50 outline-none"
                          >
                            <option value="disabled">Disabled</option>
                            <option value="cut">Cut pauses</option>
                          </select>
                        </div>
                        <div className="space-y-1">
                          <label className={`text-[10px] flex items-center ${config.pauseRemoval !== 'disabled' ? 'text-zinc-500' : 'text-zinc-600'}`}>
                            Min Pause
                            <InfoTip text="Minimum pause duration (sec) to process" />
                          </label>
                          <input 
                            type="number"
                            step="0.5"
                            min="1"
                            disabled={config.pauseRemoval === 'disabled'}
                            value={config.minPauseDuration ?? 3}
                            onChange={(e) => onConfigChange({ minPauseDuration: parseFloat(e.target.value) })}
                            className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.pauseRemoval === 'disabled' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          />
                        </div>
                        <div className="space-y-1">
                          <label className={`text-[10px] flex items-center ${config.pauseRemoval !== 'disabled' ? 'text-zinc-500' : 'text-zinc-600'}`}>
                            Preserve
                            <InfoTip text="Pause duration (sec) to preserve" />
                          </label>
                          <input 
                            type="number"
                            step="0.1"
                            min="0"
                            disabled={config.pauseRemoval === 'disabled'}
                            value={config.preservePauseDuration ?? 1.5}
                            onChange={(e) => onConfigChange({ preservePauseDuration: parseFloat(e.target.value) })}
                            className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.pauseRemoval === 'disabled' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Start Processing Button */}
        <div className="w-full relative" ref={startMenuRef}>
          <div className="w-full flex">
            <button 
              onClick={onNext}
              disabled={!canProceed}
              className="flex-1 py-3.5 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-l-lg font-semibold transition-all shadow-lg shadow-brand-500/20 active:scale-[0.98]"
            >
              {isUploading ? (
                <span className="flex items-center justify-center gap-2 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading... {Math.round(uploadProgress)}%
                </span>
              ) : (
                'Start'
              )}
            </button>
            <button
              type="button"
              onClick={() => setShowStartMenu((prev) => !prev)}
              disabled={!canProceed}
              className="px-4 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-r-lg border-l border-brand-400/40 transition-colors"
              title="More Start Options"
            >
              <ChevronDown className={`w-4 h-4 transition-transform ${showStartMenu ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {showStartMenu && canProceed && !isUploading && (
            <div className="absolute right-0 mt-2 w-64 bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl overflow-hidden z-30">
              <button
                type="button"
                onClick={() => {
                  setShowStartMenu(false);
                  onResetAndStart();
                }}
                className="w-full px-4 py-3 text-left hover:bg-zinc-800 transition-colors"
              >
                <span className="flex items-center gap-2 text-sm text-white font-medium">
                  <RotateCcw className="w-4 h-4 text-cyan-400" />
                  Reset & Start Processing
                </span>
                <span className="block mt-1 text-[11px] text-zinc-400">
                  Clear cache/artifacts and run the whole pipeline from scratch.
                </span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
