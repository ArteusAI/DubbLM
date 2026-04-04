
import React, { useState, useEffect } from 'react';
import { X, Plus, Trash2, Save, Globe, Pencil, RotateCcw, CheckCircle, Loader2, Sparkles } from 'lucide-react';
import { AppConfig, Persona } from '../types';
import { LANGUAGES } from '../constants';
import api from '../api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: AppConfig;
  personas: Persona[];
  onUpdateConfig: (cfg: Partial<AppConfig>) => void;
  onUpdatePersonas: (personas: Persona[]) => void;
  isGlobal?: boolean;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  personas,
  onUpdateConfig,
  onUpdatePersonas,
  isGlobal = false
}) => {
  const [activeTab, setActiveTab] = useState<'general' | 'api' | 'personas'>('general');
  const [newPersona, setNewPersona] = useState<Partial<Persona>>({ name: '', description: '', promptTemplate: '' });
  const [editingPersonaId, setEditingPersonaId] = useState<string | null>(null);
  const [apiKeyStatus, setApiKeyStatus] = useState<Record<string, { configured: boolean; saving?: boolean; saved?: boolean }>>({});
  const [apiKeyInputs, setApiKeyInputs] = useState<Record<string, string>>({});

  useEffect(() => {
    if (isOpen && isGlobal) {
      loadApiKeyStatus();
    }
  }, [isOpen, isGlobal]);

  const loadApiKeyStatus = async () => {
    try {
      const status = await api.getApiKeys();
      setApiKeyStatus(
        Object.fromEntries(
          Object.entries(status).map(([k, v]) => [k, { configured: v.configured }])
        )
      );
    } catch (err) {
      console.error('Failed to load API key status:', err);
    }
  };

  const handleSaveApiKey = async (provider: string) => {
    const key = apiKeyInputs[provider];
    if (!key) return;

    setApiKeyStatus(prev => ({
      ...prev,
      [provider]: { ...prev[provider], saving: true }
    }));

    try {
      await api.setApiKey(provider, key);
      setApiKeyStatus(prev => ({
        ...prev,
        [provider]: { configured: true, saving: false, saved: true }
      }));
      setApiKeyInputs(prev => ({ ...prev, [provider]: '' }));
      
      setTimeout(() => {
        setApiKeyStatus(prev => ({
          ...prev,
          [provider]: { ...prev[provider], saved: false }
        }));
      }, 2000);
    } catch (err) {
      console.error('Failed to save API key:', err);
      setApiKeyStatus(prev => ({
        ...prev,
        [provider]: { ...prev[provider], saving: false }
      }));
    }
  };

  if (!isOpen) return null;

  const handleSavePersona = async () => {
    if (newPersona.name && (newPersona.promptTemplate || newPersona.description)) {
      if (editingPersonaId) {
        const updatedPersonas = personas.map(p => 
          p.id === editingPersonaId 
            ? { ...p, ...newPersona, id: editingPersonaId } as Persona 
            : p
        );
        onUpdatePersonas(updatedPersonas);
        setEditingPersonaId(null);
      } else {
        try {
          const created = await api.createPersona({
            name: newPersona.name,
            description: newPersona.description,
            promptTemplate: newPersona.promptTemplate,
          });
          onUpdatePersonas([...personas, {
            id: created.id,
            name: created.name,
            description: created.description || '',
            promptTemplate: created.promptTemplate,
          }]);
        } catch (err) {
          console.error('Failed to create persona:', err);
          const id = newPersona.name.toLowerCase().replace(/\s+/g, '_');
          onUpdatePersonas([...personas, { id, ...newPersona } as Persona]);
        }
      }
      setNewPersona({ name: '', description: '', promptTemplate: '' });
    }
  };

  const handleEditPersona = (persona: Persona) => {
    setNewPersona({ name: persona.name, description: persona.description, promptTemplate: persona.promptTemplate });
    setEditingPersonaId(persona.id);
  };

  const handleCancelEdit = () => {
    setNewPersona({ name: '', description: '', promptTemplate: '' });
    setEditingPersonaId(null);
  };

  const handleDeletePersona = (id: string) => {
    onUpdatePersonas(personas.filter(p => p.id !== id));
    if (editingPersonaId === id) {
      handleCancelEdit();
    }
  };

  const defaultPersonaIds = ['normal', 'casual_manager', 'child', 'academic', 'technical'];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-zinc-900 w-full max-w-2xl rounded-xl border border-zinc-800 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-900">
          <div>
            <h2 className="text-xl font-semibold text-white">{isGlobal ? 'System Settings' : 'Project Settings'}</h2>
            {isGlobal && <p className="text-xs text-zinc-500 mt-1">These settings will apply to new projects.</p>}
          </div>
          <button onClick={onClose} className="p-2 hover:bg-zinc-800 rounded-lg transition-colors">
            <X className="w-5 h-5 text-zinc-400" />
          </button>
        </div>

        <div className="flex border-b border-zinc-800">
          <button
            onClick={() => setActiveTab('general')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              activeTab === 'general' ? 'bg-zinc-800 text-white border-b-2 border-brand-500' : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            General
          </button>
          <button
            onClick={() => setActiveTab('api')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              activeTab === 'api' ? 'bg-zinc-800 text-white border-b-2 border-brand-500' : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            API Keys
          </button>
          <button
            onClick={() => setActiveTab('personas')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              activeTab === 'personas' ? 'bg-zinc-800 text-white border-b-2 border-brand-500' : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Personas
          </button>
        </div>

        <div className="p-6 overflow-y-auto flex-1">
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Globe className="w-4 h-4" />
                  Default Target Language
                </label>
                <p className="text-xs text-zinc-500 mb-2">Select the language you typically translate video content into.</p>
                <div className="relative">
                  <select
                    value={config.defaultTargetLang || config.targetLang}
                    onChange={(e) => onUpdateConfig({ 
                      defaultTargetLang: e.target.value,
                      ...(isGlobal && { targetLang: e.target.value })
                    })}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>{lang.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Pencil className="w-4 h-4" />
                  Additional Translation Prompt
                </label>
                <p className="text-xs text-zinc-500 mb-2">Add custom instructions for the LLM translator (e.g., "Use polite form", "Keep it technical").</p>
                <textarea
                  value={config.translationPromptPrefix || ''}
                  onChange={(e) => onUpdateConfig({ translationPromptPrefix: e.target.value })}
                  placeholder="Enter custom translation instructions..."
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 h-24 resize-none"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  Global TTS Prompt Prefix
                </label>
                <p className="text-xs text-zinc-500 mb-2">Default instructions for all speakers (e.g., "Speak with natural energy").</p>
                <textarea
                  value={config.ttsPromptPrefix || ''}
                  onChange={(e) => onUpdateConfig({ ttsPromptPrefix: e.target.value })}
                  placeholder="Enter default TTS style instructions..."
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 h-24 resize-none"
                />
              </div>

            </div>
          )}

          {activeTab === 'api' && (
            <div className="space-y-6">
              <div className="p-4 bg-amber-500/5 border border-amber-500/10 rounded-lg mb-4">
                <p className="text-xs text-amber-200/80">
                  {isGlobal 
                    ? "API keys are stored securely on the server and used for processing." 
                    : "API keys are stored locally in your browser."}
                </p>
              </div>
              
              {/* OpenAI */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-zinc-300">OpenAI API Key</label>
                  {apiKeyStatus.openai?.configured && (
                    <span className="text-xs text-emerald-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Configured
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={isGlobal ? (apiKeyInputs.openai ?? '') : (config.apiKeys.openai || '')}
                    onChange={(e) => isGlobal 
                      ? setApiKeyInputs(prev => ({ ...prev, openai: e.target.value }))
                      : onUpdateConfig({ apiKeys: { ...config.apiKeys, openai: e.target.value } })
                    }
                    className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                    placeholder={apiKeyStatus.openai?.configured ? "••••••••" : "sk-..."}
                  />
                  {isGlobal && (
                    <button
                      onClick={() => handleSaveApiKey('openai')}
                      disabled={!apiKeyInputs.openai || apiKeyStatus.openai?.saving}
                      className="px-4 py-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      {apiKeyStatus.openai?.saving ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : apiKeyStatus.openai?.saved ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <Save className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>

              {/* Gemini */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-zinc-300">Google Gemini API Key</label>
                  {apiKeyStatus.gemini?.configured && (
                    <span className="text-xs text-emerald-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Configured
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={isGlobal ? (apiKeyInputs.gemini ?? '') : (config.apiKeys.gemini || '')}
                    onChange={(e) => isGlobal 
                      ? setApiKeyInputs(prev => ({ ...prev, gemini: e.target.value }))
                      : onUpdateConfig({ apiKeys: { ...config.apiKeys, gemini: e.target.value } })
                    }
                    className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                    placeholder={apiKeyStatus.gemini?.configured ? "••••••••" : "AIza..."}
                  />
                  {isGlobal && (
                    <button
                      onClick={() => handleSaveApiKey('gemini')}
                      disabled={!apiKeyInputs.gemini || apiKeyStatus.gemini?.saving}
                      className="px-4 py-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      {apiKeyStatus.gemini?.saving ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : apiKeyStatus.gemini?.saved ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <Save className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>

              {/* Minimax */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-zinc-300">Minimax API Key (Optional)</label>
                  {apiKeyStatus.minimax?.configured && (
                    <span className="text-xs text-emerald-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Configured
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={isGlobal ? (apiKeyInputs.minimax ?? '') : (config.apiKeys.minimax || '')}
                    onChange={(e) => isGlobal 
                      ? setApiKeyInputs(prev => ({ ...prev, minimax: e.target.value }))
                      : onUpdateConfig({ apiKeys: { ...config.apiKeys, minimax: e.target.value } })
                    }
                    className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                    placeholder={apiKeyStatus.minimax?.configured ? "••••••••" : "Enter key..."}
                  />
                  {isGlobal && (
                    <button
                      onClick={() => handleSaveApiKey('minimax')}
                      disabled={!apiKeyInputs.minimax || apiKeyStatus.minimax?.saving}
                      className="px-4 py-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      {apiKeyStatus.minimax?.saving ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : apiKeyStatus.minimax?.saved ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <Save className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'personas' && (
            <div className="space-y-6">
              <div className="bg-zinc-950 p-4 rounded-lg border border-zinc-800 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-zinc-300">
                    {editingPersonaId ? 'Edit Persona' : 'Create New Persona'}
                  </h3>
                  {editingPersonaId && (
                    <button 
                      onClick={handleCancelEdit}
                      className="text-xs text-zinc-500 hover:text-zinc-300 flex items-center gap-1"
                    >
                      <RotateCcw className="w-3 h-3" /> Cancel
                    </button>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <input
                    type="text"
                    placeholder="Name (e.g., Tech Guru)"
                    value={newPersona.name}
                    onChange={(e) => setNewPersona({ ...newPersona, name: e.target.value })}
                    className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  />
                  <input
                    type="text"
                    placeholder="Short Description"
                    value={newPersona.description}
                    onChange={(e) => setNewPersona({ ...newPersona, description: e.target.value })}
                    className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  />
                </div>
                <textarea
                  placeholder="System Prompt instructions..."
                  value={newPersona.promptTemplate || ''}
                  onChange={(e) => setNewPersona({ ...newPersona, promptTemplate: e.target.value })}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500 h-24 resize-none"
                />
                <button
                  onClick={handleSavePersona}
                  disabled={!newPersona.name}
                  className="flex items-center gap-2 px-4 py-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium transition-colors w-full justify-center"
                >
                  {editingPersonaId ? <Save className="w-4 h-4" /> : <Plus className="w-4 h-4" />} 
                  {editingPersonaId ? 'Update Persona' : 'Add Persona'}
                </button>
              </div>

              <div className="space-y-3">
                <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">Existing Personas</h3>
                {personas.map((persona) => (
                  <div 
                    key={persona.id} 
                    className={`group bg-zinc-900 border p-4 rounded-lg transition-colors ${editingPersonaId === persona.id ? 'border-brand-500/50 bg-brand-900/10' : 'border-zinc-800 hover:border-zinc-700'}`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-medium text-white flex items-center gap-2">
                          {persona.name}
                          {editingPersonaId === persona.id && <span className="text-[10px] bg-brand-500/20 text-brand-400 px-1.5 py-0.5 rounded">Editing</span>}
                        </h4>
                        <p className="text-xs text-zinc-400 mt-1">{persona.description}</p>
                      </div>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleEditPersona(persona)}
                          className="p-1.5 text-zinc-500 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                          title="Edit"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                        {!defaultPersonaIds.includes(persona.id) && (
                          <button
                            onClick={() => handleDeletePersona(persona.id)}
                            className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="p-4 border-t border-zinc-800 bg-zinc-900 flex justify-end">
          <button 
            onClick={onClose}
            className="flex items-center gap-2 px-6 py-2 bg-white text-black hover:bg-zinc-200 rounded-lg font-medium transition-colors"
          >
            <Save className="w-4 h-4" /> Done
          </button>
        </div>
      </div>
    </div>
  );
};
