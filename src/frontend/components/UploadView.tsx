
import React, { useState, useMemo, useRef, useEffect } from 'react';
import { ChevronDown, ChevronRight, Speaker, Loader2, Info, Brain, Plus, Trash2, Users } from 'lucide-react';
import { AppConfig, Persona, PresetId, LlmProvider } from '../types';
import { LANGUAGES, PRESETS, LLM_PROVIDERS, TTS_PROVIDERS, TRANSCRIPTION_PROVIDERS, WHISPER_MODELS } from '../constants';
import api from '../api';

const TTS_DEFAULT_MODELS: Record<string, string> = {
  gemini: 'gemini-2.5-flash-preview-tts',
  openai: 'gpt-4o-mini-tts',
  minimax: 'speech-02-hd',
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
  projectId: string;
  videoFile: File | null;
  isUploading: boolean;
  uploadProgress: number;
  onConfigChange: (cfg: Partial<AppConfig>) => void;
  onNext: () => void;
  onOpenSettings: () => void;
}

export const UploadView: React.FC<UploadViewProps> = ({ 
  config, 
  personas,
  projectId,
  videoFile,
  isUploading,
  uploadProgress,
  onConfigChange, 
  onNext,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExtra, setShowExtra] = useState(false);
  const [newSpeakerName, setNewSpeakerName] = useState('');
  const [newSpeakerPrompt, setNewSpeakerPrompt] = useState('');
  
  // Use ref to track latest speakerTtsPrompts to avoid race conditions with async state updates
  const speakerPromptsRef = useRef<Record<string, string>>(config.speakerTtsPrompts || {});
  
  useEffect(() => {
    speakerPromptsRef.current = config.speakerTtsPrompts || {};
  }, [config.speakerTtsPrompts]);

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
                  onClick={() => {
                    const updates: Partial<AppConfig> = { 
                      preset: preset.id as PresetId,
                      personaId: preset.id === 'fast' ? 'none' : 'normal',
                      keepBackground: preset.id !== 'fast',
                      llmProvider: preset.llmProvider,
                      llmModelName: preset.llmModelName,
                      llmTemperature: preset.llmTemperature,
                      refinementLlmProvider: preset.refinementLlmProvider,
                      refinementModelName: preset.refinementModelName,
                      refinementTemperature: preset.refinementTemperature,
                      ttsSystem: preset.ttsSystem,
                      ttsModel: preset.ttsModel,
                      ttsPromptPrefix: preset.ttsPromptPrefix,
                      voiceAutoSelection: preset.voiceAutoSelection,
                      enableEmotionEnrichment: preset.enableEmotionEnrichment,
                      dubbedVolume: preset.dubbedVolume,
                      backgroundVolume: preset.backgroundVolume,
                      useTwoPassEncoding: preset.useTwoPassEncoding,
                      maxWorkers: preset.maxWorkers,
                      pauseRemoval: preset.pauseRemoval,
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
                            value={config.ttsSystem || currentPreset.ttsSystem}
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
                        <label className={`text-[10px] flex items-center ${(config.ttsSystem || currentPreset.ttsSystem) === 'gemini' ? 'text-zinc-500' : 'text-zinc-600'}`}>
                          TTS Prompt Prefix
                          <InfoTip text="Global instruction prefix for TTS (Gemini only)" />
                        </label>
                        <input 
                          type="text"
                          value={(config.ttsSystem || currentPreset.ttsSystem) === 'gemini' ? (config.ttsPromptPrefix || currentPreset.ttsPromptPrefix || '') : ''}
                          onChange={(e) => onConfigChange({ ttsPromptPrefix: e.target.value || undefined })}
                          disabled={(config.ttsSystem || currentPreset.ttsSystem) !== 'gemini'}
                          className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${(config.ttsSystem || currentPreset.ttsSystem) !== 'gemini' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          placeholder={(config.ttsSystem || currentPreset.ttsSystem) === 'gemini' ? "Speak with natural conversational energy..." : "Only available for Gemini"}/>
                      </div>

                      {/* Per-Speaker TTS Prompts */}
                      <div className={`space-y-2 pt-2 border-t border-zinc-800/30 ${(config.ttsSystem || currentPreset.ttsSystem) !== 'gemini' ? 'opacity-40 pointer-events-none' : ''}`}>
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
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                const videoMin = config.videoSegmentSpeedMin ?? 0.75;
                                if (val < videoMin) {
                                  onConfigChange({ comfortMinAdjustmentRatio: val, videoSegmentSpeedMin: val - 0.1 });
                                } else {
                                  onConfigChange({ comfortMinAdjustmentRatio: val });
                                }
                              }}
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
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                const videoMax = config.videoSegmentSpeedMax ?? 1.5;
                                if (val > videoMax) {
                                  onConfigChange({ comfortMaxAdjustmentRatio: val, videoSegmentSpeedMax: val + 0.1 });
                                } else {
                                  onConfigChange({ comfortMaxAdjustmentRatio: val });
                                }
                              }}
                              className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.segmentStretch === 'video' ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Video Comfort Zone */}
                      <div className="pt-1">
                        <label className={`text-[10px] font-medium flex items-center gap-1 ${config.segmentStretch === 'audio' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                          🎬 Video Speed Range
                          <InfoTip text="Video playback speed limits. Must be wider than audio range. Uses minterpolate for slowdown beyond 0.75x" />
                        </label>
                        <div className="grid grid-cols-2 gap-2 mt-1">
                          <div className="space-y-0.5">
                            <label className={`text-[10px] ${config.segmentStretch === 'audio' ? 'text-zinc-600' : 'text-zinc-500'}`}>Slowdown</label>
                            <input 
                              type="number"
                              step="0.05"
                              min="0.25"
                              max={config.comfortMinAdjustmentRatio ?? 0.85}
                              disabled={config.segmentStretch === 'audio'}
                              value={config.videoSegmentSpeedMin ?? 0.75}
                              onChange={(e) => onConfigChange({ videoSegmentSpeedMin: parseFloat(e.target.value) })}
                              className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.segmentStretch === 'audio' ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                          </div>
                          <div className="space-y-0.5">
                            <label className={`text-[10px] ${config.segmentStretch === 'audio' ? 'text-zinc-600' : 'text-zinc-500'}`}>Speedup</label>
                            <input 
                              type="number"
                              step="0.05"
                              min={config.comfortMaxAdjustmentRatio ?? 1.15}
                              max="3"
                              disabled={config.segmentStretch === 'audio'}
                              value={config.videoSegmentSpeedMax ?? 1.5}
                              onChange={(e) => onConfigChange({ videoSegmentSpeedMax: parseFloat(e.target.value) })}
                              className={`w-full bg-zinc-950 border border-zinc-700/50 rounded px-2 py-1.5 text-[11px] text-white focus:ring-1 focus:ring-brand-500/50 outline-none ${config.segmentStretch === 'audio' ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                          </div>
                        </div>
                        {/* Validation warning */}
                        {config.segmentStretch !== 'audio' && (
                          (config.videoSegmentSpeedMin ?? 0.75) >= (config.comfortMinAdjustmentRatio ?? 0.85) ||
                          (config.videoSegmentSpeedMax ?? 1.5) <= (config.comfortMaxAdjustmentRatio ?? 1.15)
                        ) && (
                          <p className="text-[9px] text-amber-500 mt-1">
                            ⚠️ Video range must be wider than audio range
                          </p>
                        )}
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
        <button 
          onClick={onNext}
          disabled={!canProceed}
          className="w-full py-3.5 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition-all shadow-lg shadow-brand-500/20 active:scale-[0.98]"
        >
          {isUploading ? (
            <span className="flex items-center justify-center gap-2 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              Uploading... {Math.round(uploadProgress)}%
            </span>
          ) : (
            'Start Processing'
          )}
        </button>
      </div>
    </div>
  );
};
