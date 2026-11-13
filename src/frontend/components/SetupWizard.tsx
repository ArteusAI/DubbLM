import React, { useState } from 'react';
import { KeyRound, Loader2, ArrowRight, ExternalLink, ShieldCheck } from 'lucide-react';
import api from '../api';

interface SetupWizardProps {
  onComplete: () => void;
}

export const SetupWizard: React.FC<SetupWizardProps> = ({ onComplete }) => {
  const [key, setKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const value = key.trim();
    if (!value) {
      setError('Введите ключ OpenRouter.');
      return;
    }
    if (!value.startsWith('sk-or-')) {
      setError('Ключ OpenRouter должен начинаться с sk-or-.');
      return;
    }

    setIsSaving(true);
    setError(null);
    try {
      await api.saveOpenRouterKey(value);
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось сохранить ключ.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="min-h-full flex items-center justify-center bg-zinc-950 px-4 py-10">
      <div className="w-full max-w-xl rounded-2xl border border-zinc-800 bg-zinc-900/80 shadow-2xl overflow-hidden">
        <div className="p-8 border-b border-zinc-800 bg-gradient-to-br from-brand-600/15 to-transparent">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-11 h-11 rounded-xl bg-brand-600/20 border border-brand-500/30 flex items-center justify-center">
              <KeyRound className="w-5 h-5 text-brand-400" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">Настройка DubbLM</h1>
              <p className="text-sm text-zinc-400">Один ключ OpenRouter для всей системы</p>
            </div>
          </div>
          <p className="text-sm leading-6 text-zinc-400">
            DubbLM использует OpenRouter для транскрибации, диаризации, перевода,
            редактора и озвучки. Вставьте ключ — остальное настроится автоматически.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-8 space-y-5">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="openrouter-key" className="text-sm font-medium text-zinc-200">
                OpenRouter API key
              </label>
              <a
                href="https://openrouter.ai/settings/keys"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-brand-400 hover:text-brand-300 inline-flex items-center gap-1"
              >
                Получить ключ <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            <div className="relative">
              <input
                id="openrouter-key"
                type={showKey ? 'text' : 'password'}
                value={key}
                onChange={(e) => setKey(e.target.value)}
                autoFocus
                autoComplete="off"
                spellCheck={false}
                placeholder="sk-or-v1-..."
                className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-4 py-3 pr-20 text-sm text-white outline-none focus:ring-2 focus:ring-brand-500/50"
              />
              <button
                type="button"
                onClick={() => setShowKey((value) => !value)}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              >
                {showKey ? 'Скрыть' : 'Показать'}
              </button>
            </div>
            <p className="text-xs text-zinc-500">
              Ключ сохраняется на сервере в <span className="text-zinc-400">.env</span> и{' '}
              <span className="text-zinc-400">settings.yml</span> и никогда не показывается полностью.
            </p>
          </div>

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
              {error}
            </div>
          )}

          <div className="flex items-start gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs text-emerald-200/80">
            <ShieldCheck className="w-4 h-4 mt-0.5 shrink-0 text-emerald-400" />
            <span>
              После первого запуска ключ можно поменять в Settings → API Keys.
              Дополнительные ключи не нужны.
            </span>
          </div>

          <button
            type="submit"
            disabled={isSaving || !key.trim()}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-brand-600 px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Проверяем ключ...
              </>
            ) : (
              <>
                Продолжить <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
