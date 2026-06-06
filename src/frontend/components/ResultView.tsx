import React, { useState, useEffect } from 'react';
import { Download, FileText, ArrowLeft, CheckCircle, Clock, DollarSign, ChevronDown, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import api, { ProjectStatsResponse } from '../api';

interface ResultViewProps {
  projectId: string;
  onReset: () => void;
}

interface ReportArtifactLink {
  label: string;
  href: string;
  meta: string;
}

interface ReportContentParts {
  beforeApiCosts: string;
  beforeArtifacts: string;
  artifactLinks: ReportArtifactLink[];
  afterArtifacts: string;
}

const splitReportArtifactLinks = (markdown: string): ReportContentParts => {
  const artifactsMatch = markdown.match(/(^|\n)## Artifacts\s*\n+([\s\S]*?)(?=\n## |\s*$)/);
  if (!artifactsMatch || artifactsMatch.index === undefined) {
    return { beforeApiCosts: '', beforeArtifacts: markdown, artifactLinks: [], afterArtifacts: '' };
  }

  const sectionPrefix = artifactsMatch[1] || '';
  const sectionStart = artifactsMatch.index + sectionPrefix.length;
  const sectionEnd = artifactsMatch.index + artifactsMatch[0].length;
  const artifactsBody = artifactsMatch[2] || '';
  const artifactLinks = artifactsBody
    .split(/\r?\n/)
    .map((line) => {
      const linkMatch = line.match(/^\s*-\s+\[([^\]]+)]\(([^)]+)\)(.*)$/);
      if (!linkMatch) return null;
      return {
        label: linkMatch[1],
        href: linkMatch[2],
        meta: linkMatch[3].replace(/^\s*[—-]\s*/, '').trim(),
      };
    })
    .filter((link): link is ReportArtifactLink => link !== null);

  if (!artifactLinks.length) {
    return { beforeApiCosts: '', beforeArtifacts: markdown, artifactLinks: [], afterArtifacts: '' };
  }

  return {
    beforeApiCosts: '',
    beforeArtifacts: markdown.slice(0, sectionStart).trimEnd(),
    artifactLinks,
    afterArtifacts: markdown.slice(sectionEnd).replace(/^\n+/, ''),
  };
};

interface ReportApiCostRow {
  stageKey: string;
  stageLabel: string;
  categoryLabel: string;
  provider: string;
  model: string;
  usage: Record<string, unknown>;
  rates: Record<string, unknown>;
  pricingSource: string;
  costUsd: number;
}

interface ApiCostGroup {
  key: string;
  label: string;
  rows: ReportApiCostRow[];
  totalCost: number;
}

const STAGE_ORDER = [
  'transcription',
  'context_analysis',
  'translation',
  'emotion_analysis',
  'speech_synthesis',
];

const splitReportContent = (markdown: string, hideApiCosts: boolean): ReportContentParts => {
  if (!hideApiCosts) {
    return splitReportArtifactLinks(markdown);
  }

  const apiCostsMatch = markdown.match(/(^|\n)## API Costs\s*\n+[\s\S]*?(?=\n## |\s*$)/);
  if (!apiCostsMatch || apiCostsMatch.index === undefined) {
    return splitReportArtifactLinks(markdown);
  }

  const sectionPrefix = apiCostsMatch[1] || '';
  const sectionStart = apiCostsMatch.index + sectionPrefix.length;
  const sectionEnd = apiCostsMatch.index + apiCostsMatch[0].length;
  const artifactParts = splitReportArtifactLinks(markdown.slice(sectionEnd).replace(/^\n+/, ''));
  return {
    ...artifactParts,
    beforeApiCosts: markdown.slice(0, sectionStart).trimEnd(),
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

const asNumber = (value: unknown): number => (
  typeof value === 'number' && Number.isFinite(value) ? value : 0
);

const asString = (value: unknown, fallback = '—'): string => {
  if (typeof value !== 'string') return fallback;
  const trimmed = value.trim();
  return trimmed || fallback;
};

const normalizeApiCostRows = (value: unknown): ReportApiCostRow[] => {
  if (!isRecord(value) || !Array.isArray(value.api_costs)) return [];
  return value.api_costs
    .filter(isRecord)
    .map((row) => ({
      stageKey: asString(row.stage_key, 'unknown'),
      stageLabel: asString(row.stage_label, 'Unknown'),
      categoryLabel: asString(row.category_label, asString(row.category, 'Unknown')),
      provider: asString(row.provider),
      model: asString(row.model),
      usage: isRecord(row.usage) ? row.usage : {},
      rates: isRecord(row.rates) ? row.rates : {},
      pricingSource: asString(row.pricing_source, asString(isRecord(row.rates) ? row.rates.pricing_source : undefined, 'unknown')),
      costUsd: asNumber(row.cost_usd),
    }))
    .filter((row) => row.costUsd !== 0 || Object.keys(row.usage).length > 0);
};

const buildApiCostGroups = (rows: ReportApiCostRow[]): ApiCostGroup[] => {
  const groups = new Map<string, ApiCostGroup>();
  rows.forEach((row) => {
    const key = row.stageKey;
    const existing = groups.get(key);
    if (existing) {
      existing.rows.push(row);
      existing.totalCost += row.costUsd;
      return;
    }
    groups.set(key, {
      key,
      label: row.stageLabel,
      rows: [row],
      totalCost: row.costUsd,
    });
  });

  return Array.from(groups.values()).sort((a, b) => {
    const aIdx = STAGE_ORDER.indexOf(a.key);
    const bIdx = STAGE_ORDER.indexOf(b.key);
    const aOrder = aIdx === -1 ? STAGE_ORDER.length : aIdx;
    const bOrder = bIdx === -1 ? STAGE_ORDER.length : bIdx;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.label.localeCompare(b.label);
  });
};

const formatCost = (value: number): string => `$${value.toFixed(4)}`;

const formatCompactNumber = (value: number): string => (
  Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 2 })
);

const formatRateNumber = (value: number): string => (
  value >= 1 ? value.toFixed(2).replace(/\.?0+$/, '') : Number(value.toPrecision(4)).toString()
);

const formatAudioUsage = (seconds: number): string => {
  const hours = seconds / 3600;
  if (hours >= 0.1) return `${hours.toFixed(2)} h audio`;
  return `${(seconds / 60).toFixed(2)} min audio`;
};

const formatApiUsage = (usage: Record<string, unknown>): string => {
  const parts: string[] = [];
  const audioSeconds = asNumber(usage.audio_seconds);
  if (audioSeconds > 0) parts.push(formatAudioUsage(audioSeconds));

  const inputTokens = asNumber(usage.input_tokens);
  const outputTokens = asNumber(usage.output_tokens);
  const reasoningTokens = asNumber(usage.reasoning_tokens);
  const tokenParts: string[] = [];
  if (inputTokens > 0) tokenParts.push(`in ${formatCompactNumber(inputTokens)} tok`);
  if (outputTokens > 0) tokenParts.push(`out ${formatCompactNumber(outputTokens)} tok`);
  if (reasoningTokens > 0) tokenParts.push(`reasoning ${formatCompactNumber(reasoningTokens)} tok`);
  if (tokenParts.length) parts.push(tokenParts.join(', '));

  const inputChars = asNumber(usage.input_characters);
  if (inputChars > 0) parts.push(`${formatCompactNumber(inputChars)} chars`);

  const voiceCount = asNumber(usage.voice_count);
  if (voiceCount > 0) parts.push(`${formatCompactNumber(voiceCount)} voice(s)`);

  return parts.join('; ') || '—';
};

const formatApiRate = (rates: Record<string, unknown>, pricingSource: string): string => {
  const parts: string[] = [];
  const perMin = asNumber(rates.per_min);
  if (perMin > 0) parts.push(`$${formatRateNumber(perMin)}/min`);

  const diarization = asNumber(rates.speaker_diarization_per_min);
  if (diarization > 0) parts.push(`+$${formatRateNumber(diarization)}/min diarization`);

  const inputRate = asNumber(rates.input_per_1m_tokens);
  if (inputRate > 0) parts.push(`$${formatRateNumber(inputRate)}/M input tok`);

  const outputRate = asNumber(rates.output_per_1m_tokens);
  if (outputRate > 0) parts.push(`$${formatRateNumber(outputRate)}/M output tok`);

  const reasoningRate = asNumber(rates.reasoning_per_1m_tokens);
  if (reasoningRate > 0 && reasoningRate !== outputRate) {
    parts.push(`$${formatRateNumber(reasoningRate)}/M reasoning tok`);
  }

  const perAudioMin = asNumber(rates.per_audio_min);
  if (perAudioMin > 0) parts.push(`$${formatRateNumber(perAudioMin)}/audio min`);

  const perChars = asNumber(rates.per_1m_chars);
  if (perChars > 0) parts.push(`$${formatRateNumber(perChars)}/M chars`);

  const audioTokensPerSecond = asNumber(rates.audio_output_tokens_per_second);
  if (audioTokensPerSecond > 0) parts.push(`${formatRateNumber(audioTokensPerSecond)} audio tok/s`);

  const voiceClone = asNumber(rates.voice_clone_per_voice);
  if (voiceClone > 0) parts.push(`$${formatRateNumber(voiceClone)}/voice`);

  const source = pricingSource !== 'unknown' ? ` (${pricingSource})` : '';
  return `${parts.join(', ') || '—'}${source}`;
};

const ApiCostsPanel: React.FC<{ rows: ReportApiCostRow[] }> = ({ rows }) => {
  const groups = buildApiCostGroups(rows);
  const total = groups.reduce((sum, group) => sum + group.totalCost, 0);
  if (!groups.length) return null;

  return (
    <section className="my-5 rounded-2xl border border-zinc-800 bg-black/35">
      <div className="flex items-center justify-between gap-3 border-b border-zinc-800 px-4 py-3">
        <div>
          <h2 className="!my-0 text-lg font-semibold text-white">API Costs</h2>
          <p className="mt-1 text-xs text-zinc-500">Paid API usage grouped by pipeline stage</p>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wide text-zinc-500">Total</div>
          <div className="text-base font-semibold text-white">{formatCost(total)}</div>
        </div>
      </div>

      <div className="divide-y divide-zinc-800">
        {groups.map((group) => (
          <details key={group.key} className="group">
            <summary className="flex cursor-pointer select-none items-center justify-between gap-3 px-4 py-3 hover:bg-zinc-900/70">
              <span className="flex min-w-0 items-center gap-2">
                <ChevronDown className="h-4 w-4 shrink-0 text-zinc-500 transition-transform group-open:rotate-180" />
                <span className="truncate font-semibold text-zinc-100">{group.label}</span>
              </span>
              <span className="flex shrink-0 items-center gap-2 text-xs">
                <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-400">{group.rows.length} rows</span>
                <span className="font-semibold text-white">{formatCost(group.totalCost)}</span>
              </span>
            </summary>
            <div className="overflow-x-auto border-t border-zinc-800">
              <table className="!my-0 min-w-[760px]">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Provider</th>
                    <th>Model</th>
                    <th>Usage</th>
                    <th>Rate</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {group.rows.map((row, index) => (
                    <tr key={`${group.key}-${row.categoryLabel}-${row.provider}-${row.model}-${index}`}>
                      <td>{row.categoryLabel}</td>
                      <td>{row.provider}</td>
                      <td>{row.model}</td>
                      <td>{formatApiUsage(row.usage)}</td>
                      <td>{formatApiRate(row.rates, row.pricingSource)}</td>
                      <td className="whitespace-nowrap text-right font-medium text-zinc-100">{formatCost(row.costUsd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ))}
      </div>
    </section>
  );
};

const formatTime = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hours = Math.floor(mins / 60);
  const remainingMins = mins % 60;
  return `${hours}h ${remainingMins}m`;
};

export const ResultView: React.FC<ResultViewProps> = ({ projectId, onReset }) => {
  const [stats, setStats] = useState<ProjectStatsResponse | null>(null);
  const [reportMd, setReportMd] = useState<string | null>(null);
  const [reportApiCosts, setReportApiCosts] = useState<ReportApiCostRow[]>([]);
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);

  const streamVideoUrl = api.getStreamVideoUrl(projectId);
  const downloadVideoUrl = api.getDownloadVideoUrl(projectId);
  const sourceSrtUrl = api.getDownloadSubtitlesUrl(projectId, 'srt', 'source');
  const targetSrtUrl = api.getDownloadSubtitlesUrl(projectId, 'srt', 'target');
  const downloadReportUrl = api.getDownloadReportUrl(projectId);

  useEffect(() => {
    api.getProjectStats(projectId)
      .then(setStats)
      .catch(() => setStats(null));
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    setReportLoading(true);
    setReportError(null);
    setReportMd(null);
    setReportApiCosts([]);
    setIsReportOpen(false);
    api.getProjectReport(projectId)
      .then(({ markdown, json }) => {
        if (!cancelled) {
          setReportMd(markdown);
          setReportApiCosts(normalizeApiCostRows(json));
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setReportError(err.message || 'Report not available');
      })
      .finally(() => {
        if (!cancelled) setReportLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const handleDownloadReport = () => {
    const link = document.createElement('a');
    link.href = downloadReportUrl;
    link.download = 'dubbing_report.md';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadVideo = () => {
    const link = document.createElement('a');
    link.href = downloadVideoUrl;
    link.download = 'dubbed_video.mp4';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadSubtitles = (lang: 'source' | 'target') => {
    const url = lang === 'source' ? sourceSrtUrl : targetSrtUrl;
    const link = document.createElement('a');
    link.href = url;
    link.download = `subtitles_${lang}.srt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const reportContent = reportMd ? splitReportContent(reportMd, reportApiCosts.length > 0) : null;

  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center space-y-8 p-6">
      <div className="text-center space-y-4 max-w-lg">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-emerald-500/10 mb-4 ring-1 ring-emerald-500/50">
          <CheckCircle className="w-10 h-10 text-emerald-500" />
        </div>
        <h2 className="text-4xl font-bold text-white">Dubbing Complete!</h2>
        <p className="text-zinc-400 text-lg">Your video has been successfully processed, translated, and synthesized.</p>
        
        {stats && (stats.processingTimeSec > 0 || stats.totalCost > 0) && (
          <div className="flex justify-center gap-6 mt-4">
            {stats.processingTimeSec > 0 && (
              <div className="flex items-center gap-2 text-zinc-400">
                <Clock className="w-4 h-4" />
                <span>{formatTime(stats.processingTimeSec)}</span>
              </div>
            )}
            {stats.totalCost > 0 && (
              <div className="flex items-center gap-2 text-zinc-400">
                <DollarSign className="w-4 h-4" />
                <span>${stats.totalCost.toFixed(4)}</span>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="w-full max-w-3xl bg-black rounded-xl overflow-hidden aspect-video border border-zinc-800 shadow-2xl">
        <video 
          src={streamVideoUrl}
          controls
          preload="metadata"
          playsInline
          className="w-full h-full"
        />
      </div>

      <div className="flex flex-row justify-center gap-3">
        <button 
          onClick={handleDownloadVideo}
          className="flex items-center justify-center gap-2 px-6 py-3 bg-brand-600 hover:bg-brand-500 text-white rounded-xl font-bold shadow-lg shadow-brand-500/20 transition-all active:scale-95"
        >
          <Download className="w-5 h-5" />
          Download Video
        </button>
        <button 
          onClick={() => handleDownloadSubtitles('source')}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl font-medium transition-all"
          title="Original language subtitles"
        >
          <FileText className="w-4 h-4" />
          Source .SRT
        </button>
        <button 
          onClick={() => handleDownloadSubtitles('target')}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl font-medium transition-all"
          title="Translated language subtitles"
        >
          <FileText className="w-4 h-4" />
          Target .SRT
        </button>
      </div>

      <button 
        onClick={onReset}
        className="flex items-center gap-2 text-zinc-500 hover:text-zinc-300 transition-colors text-sm mt-8"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Editor
      </button>

      <button
        type="button"
        onClick={() => setIsReportOpen((open) => !open)}
        className="absolute bottom-5 right-5 z-20 flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-black/35 text-zinc-500 backdrop-blur-md transition-all hover:bg-black/60 hover:text-zinc-200 sm:bottom-6 sm:right-6"
        aria-label={isReportOpen ? 'Hide summary report' : 'Show summary report'}
        title="Summary Report"
      >
        <FileText className="h-3.5 w-3.5" />
      </button>

      <div
        className={`absolute inset-y-0 right-0 z-20 w-full border-l border-zinc-800 bg-zinc-950 shadow-[0_0_80px_rgba(0,0,0,0.7)] backdrop-blur-md transition-all duration-200 sm:w-1/2 ${
          isReportOpen
            ? 'pointer-events-auto translate-x-0 opacity-100'
            : 'pointer-events-none translate-x-6 opacity-0'
        }`}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <ChevronDown className={`h-4 w-4 transition-transform ${isReportOpen ? 'rotate-180' : ''}`} />
            <span>Summary Report</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadReport}
              disabled={!reportMd}
              className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs text-white transition-all hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Download className="h-3.5 w-3.5" />
              <span>.md</span>
            </button>
            <button
              type="button"
              onClick={() => setIsReportOpen(false)}
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-zinc-900 text-zinc-400 transition-all hover:bg-zinc-800 hover:text-white"
              aria-label="Close summary report"
              title="Close report"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="h-[calc(100%-57px)] overflow-auto px-4 py-3">
          {reportLoading && (
            <p className="text-sm text-zinc-500">Loading report…</p>
          )}
          {!reportLoading && reportError && (
            <p className="text-sm text-zinc-500">{reportError}</p>
          )}
          {!reportLoading && !reportError && reportContent && (
            <div className="report-markdown text-sm text-zinc-300 leading-relaxed [&_h1]:mt-0 [&_h1]:mb-3 [&_h1]:text-xl [&_h1]:font-bold [&_h1]:text-white [&_h2]:mt-6 [&_h2]:mb-2 [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:text-white [&_h3]:mt-4 [&_h3]:mb-2 [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-zinc-200 [&_a]:text-brand-400 [&_a]:underline hover:[&_a]:text-brand-300 [&_code]:rounded [&_code]:bg-zinc-800 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_hr]:my-4 [&_hr]:border-zinc-800 [&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-zinc-800 [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-zinc-700 [&_th]:bg-zinc-800 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6">
              {reportContent.beforeArtifacts && (
                <>
                  {reportContent.beforeApiCosts && (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent.beforeApiCosts}</ReactMarkdown>
                  )}
                  <ApiCostsPanel rows={reportApiCosts} />
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent.beforeArtifacts}</ReactMarkdown>
                </>
              )}
              {!reportContent.beforeArtifacts && reportContent.beforeApiCosts && (
                <>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent.beforeApiCosts}</ReactMarkdown>
                  <ApiCostsPanel rows={reportApiCosts} />
                </>
              )}
              {!reportContent.beforeArtifacts && !reportContent.beforeApiCosts && reportApiCosts.length > 0 && (
                <ApiCostsPanel rows={reportApiCosts} />
              )}
              {reportContent.artifactLinks.length > 0 && (
                <details className="group my-5 rounded-xl border border-zinc-800 bg-black/35">
                  <summary className="flex cursor-pointer select-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-zinc-100 transition-colors hover:bg-zinc-900/80">
                    <span className="flex items-center gap-2">
                      <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
                      Artifact links
                    </span>
                    <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs font-medium text-zinc-400">
                      {reportContent.artifactLinks.length} files
                    </span>
                  </summary>
                  <ul className="!my-0 max-h-72 !list-none divide-y divide-zinc-800 overflow-auto border-t border-zinc-800 !pl-0">
                    {reportContent.artifactLinks.map((artifact, index) => (
                      <li key={`${artifact.href}-${index}`} className="px-4 py-2.5">
                        <a
                          href={artifact.href}
                          target="_blank"
                          rel="noreferrer"
                          className="break-all text-brand-400 underline transition-colors hover:text-brand-300"
                        >
                          {artifact.label}
                        </a>
                        {artifact.meta && (
                          <span className="ml-2 whitespace-nowrap text-xs text-zinc-500">{artifact.meta}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {reportContent.afterArtifacts && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent.afterArtifacts}</ReactMarkdown>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
