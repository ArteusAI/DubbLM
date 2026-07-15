import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Plus, FolderOpen, Clock, Trash2, Video, FileText, Upload, Sparkles, StopCircle, CheckSquare, Square, Settings, AlertCircle, Loader2, RotateCcw, Link2, Download, Cookie, X, ChevronDown, HelpCircle } from 'lucide-react';
import { Project, ProjectStatus, PresetId } from '../types';
import { PRESETS, LANGUAGES } from '../constants';
import { api } from '../api';

const LANG_FLAGS: Record<string, string> = {
  auto: '🔮',
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

const formatBytes = (bytes?: number | null): string => {
  if (bytes == null || bytes <= 0) return '';
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1).replace(/\.0$/, '')} ${units[unit]}`;
};

const formatResolution = (w?: number | null, h?: number | null): string => {
  if (!w || !h) return '';
  return `${w}×${h}`;
};

const formatDuration = (seconds?: number | null): string => {
  if (seconds == null || seconds <= 0) return '';
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
};

const buildProjectMetaLine = (project: Project): string => {
  const parts: string[] = [];
  const resolution = formatResolution(project.sourceWidth, project.sourceHeight)
    || formatResolution(project.resultWidth, project.resultHeight);
  if (resolution) parts.push(resolution);
  const duration = formatDuration(project.sourceDuration)
    || formatDuration(project.resultDuration);
  if (duration) parts.push(duration);
  const source = formatBytes(project.sourceSize);
  if (source) parts.push(`исх ${source}`);
  const result = formatBytes(project.resultSize);
  if (result) parts.push(`рез ${result}`);
  const total = formatBytes(project.totalSize);
  if (total) parts.push(`всего ${total}`);
  return parts.join(' · ');
};

interface ProjectListViewProps {
  projects: Project[];
  onSelectProject: (id: string) => void;
  onDeleteProject: (id: string) => void;
  onBatchUpload: (files: File[]) => void;
  onDownloadFromUrl: (url: string, quality: 'best' | '1080p' | '720p' | '480p') => void;
  onRetryDownload: (id: string) => void;
  onAutoProcess: (ids: string[]) => void;
  onStopProcess: (ids: string[]) => void;
  onResetAndRestart: (id: string) => void;
  onOpenSettings: () => void;
  onChangePreset: (projectId: string, preset: PresetId) => void;
  onChangeLanguages: (projectId: string, sourceLang: string, targetLang: string) => void;
  onProgressUpdate: (projectId: string, progress: number, stage: string) => void;
  onDownloadProgressUpdate: (projectId: string, progress: number, stage: string) => void;
  onStatusComplete: (projectId: string, status: ProjectStatus, error?: string) => void;
}

interface CompactLangSelectorProps {
  value: string;
  onChange: (value: string) => void;
  isSource?: boolean;
  disabled?: boolean;
}

const CompactLangSelector: React.FC<CompactLangSelectorProps> = ({ value, onChange, isSource, disabled }) => {
  const options = isSource 
    ? LANGUAGES 
    : LANGUAGES.filter(l => l.code !== 'auto');
  
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="appearance-none bg-transparent text-[11px] text-zinc-400 cursor-pointer hover:text-zinc-200 focus:text-white outline-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed uppercase font-medium"
    >
      {options.map(lang => (
        <option key={lang.code} value={lang.code} className="bg-zinc-900 text-zinc-300">
          {LANG_FLAGS[lang.code]} {lang.code.toUpperCase()}
        </option>
      ))}
    </select>
  );
};

// Project Card with dynamic video background from backend
interface ProjectCardProps {
  project: Project;
  isSelected: boolean;
  isExtensionInstalled: boolean;
  onSelect: () => void;
  onToggleSelect: () => void;
  onDelete: () => void;
  onAutoProcess: () => void;
  onStopProcess: () => void;
  onResetAndRestart: () => void;
  onRetryDownload: () => void;
  onChangePreset: (preset: PresetId) => void;
  onChangeLanguages: (source: string, target: string) => void;
  onProgressUpdate: (progress: number, stage: string) => void;
  onDownloadProgressUpdate: (progress: number, stage: string) => void;
  onStatusComplete: (status: ProjectStatus, error?: string) => void;
}

const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  isSelected,
  isExtensionInstalled,
  onSelect,
  onToggleSelect,
  onDelete,
  onAutoProcess,
  onStopProcess,
  onResetAndRestart,
  onRetryDownload,
  onChangePreset,
  onChangeLanguages,
  onProgressUpdate,
  onDownloadProgressUpdate,
  onStatusComplete
}) => {
  const [frameLoaded, setFrameLoaded] = useState(false);
  const [thumbLoaded, setThumbLoaded] = useState(false);
  const [localProgress, setLocalProgress] = useState(0);
  const [localStage, setLocalStage] = useState<string | undefined>();
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [isDraggingCookie, setIsDraggingCookie] = useState(false);
  const [isUploadingCookie, setIsUploadingCookie] = useState(false);
  const [hasAttemptedAutoSync, setHasAttemptedAutoSync] = useState(false);

  // Automatically trigger cookie sync when a cookie error is shown and the extension is installed
  useEffect(() => {
    if (project.status === 'error' && isCookieError(project.error)) {
      if (!hasAttemptedAutoSync) {
        if (isExtensionInstalled) {
          console.log('Cookie error detected, requesting automatic cookie sync from extension...');
          setHasAttemptedAutoSync(true);
          window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_COOKIES', { 
            detail: { serverUrl: window.location.origin } 
          }));
        }
      }
    } else {
      // Reset the auto-sync attempt flag once status is no longer 'error'
      setHasAttemptedAutoSync(false);
    }
  }, [project.status, project.error, hasAttemptedAutoSync, isExtensionInstalled]);

  // Listen to successful cookie synchronization from the extension and auto-retry download
  useEffect(() => {
    const handleSyncSuccess = () => {
      if (project.status === 'error' && isCookieError(project.error)) {
        console.log('Extension synced cookies successfully! Retrying download automatically...');
        onRetryDownload();
      }
    };

    window.addEventListener('DUBBLM_SYNC_SUCCESS', handleSyncSuccess);
    return () => {
      window.removeEventListener('DUBBLM_SYNC_SUCCESS', handleSyncSuccess);
    };
  }, [project.status, project.error, onRetryDownload]);

  const isCookieError = (error?: string) => {
    if (!error) return false;
    const err = error.toLowerCase();
    return err.includes('cookie') || err.includes('sign-in') || err.includes('sign in') || err.includes('age-restricted');
  };

  const handleCookieDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCookie(true);
  };

  const handleCookieDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCookie(false);
  };

  const handleCookieDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCookie(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.endsWith('.txt')) {
        setIsUploadingCookie(true);
        try {
          await api.uploadCookies(file);
          onRetryDownload();
        } catch (err) {
          alert(err instanceof Error ? err.message : 'Failed to upload cookies');
        } finally {
          setIsUploadingCookie(false);
        }
      } else {
        alert('Please drop a valid .txt cookies file.');
      }
    }
  };
  
  // Store callbacks in refs to avoid dependency issues
  const onProgressRef = useRef(onProgressUpdate);
  const onDownloadProgressRef = useRef(onDownloadProgressUpdate);
  const onCompleteRef = useRef(onStatusComplete);
  onProgressRef.current = onProgressUpdate;
  onDownloadProgressRef.current = onDownloadProgressUpdate;
  onCompleteRef.current = onStatusComplete;
  
  const processing = project.status === 'transcribing' || project.status === 'dubbing' || project.isAutoProcessing;
  const downloading = project.status === 'downloading' || project.isDownloading;
  const uploading = project.isUploading;
  
  // Subscribe to SSE for progress updates when processing or downloading
  useEffect(() => {
    if ((!processing && !downloading) || uploading) {
      setLocalProgress(0);
      setLocalStage(undefined);
      return;
    }
    
    const unsubscribe = api.subscribeToStatus(project.id, {
      onProgress: (percent, step) => {
        setLocalProgress(percent);
        setLocalStage(step);
        if (downloading) {
          onDownloadProgressRef.current(percent, step);
        } else {
          onProgressRef.current(percent, step);
        }
      },
      onComplete: (status, error) => {
        onCompleteRef.current(status as ProjectStatus, error);
      },
      onError: () => {
        // Silent fail - will retry on next render
      }
    });
    
    return unsubscribe;
  }, [project.id, processing, downloading, uploading]);
  
  // Use local progress for real-time updates, fallback to project progress
  const progress = downloading
    ? (localProgress || project.downloadProgress || 0)
    : (localProgress || project.processProgress || 0);
  const stage = localStage || (downloading ? project.downloadStage : project.processStage);
  
  // Get frame URL from backend (buckets at 5% intervals)
  const frameUrl = useMemo(() => {
    if (!processing || uploading || downloading) return null;
    return api.getVideoFrameUrl(project.id, progress);
  }, [project.id, processing, uploading, downloading, progress]);
  
  // Get static thumbnail URL from backend
  const thumbnailUrl = useMemo(() => {
    // Always available after upload
    if (uploading || downloading) return null;
    return api.getVideoThumbnailUrl(project.id);
  }, [project.id, uploading, downloading]);
  
  // Preload frame image
  useEffect(() => {
    if (!frameUrl) {
      setFrameLoaded(false);
      return;
    }
    setFrameLoaded(false);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setFrameLoaded(true);
    };
    img.onerror = () => {
      console.warn('Failed to load frame:', frameUrl);
      setFrameLoaded(false);
    };
    img.src = frameUrl;
  }, [frameUrl]);
  
  // Preload thumbnail
  useEffect(() => {
    if (!thumbnailUrl) {
      setThumbLoaded(false);
      return;
    }
    setThumbLoaded(false);
    const img = new Image();
    img.onload = () => setThumbLoaded(true);
    img.onerror = () => {
      console.warn('Failed to load thumbnail:', thumbnailUrl);
      setThumbLoaded(false);
    };
    img.src = thumbnailUrl;
  }, [thumbnailUrl]);
  
  // Choose background: dynamic frame when processing, otherwise thumbnail
  const backgroundImage = processing && frameLoaded && frameUrl 
    ? frameUrl 
    : (thumbLoaded && thumbnailUrl ? thumbnailUrl : null);
  
  // Show loading state while waiting for frame/thumb
  const isLoadingFrame = (processing && frameUrl && !frameLoaded) || (!uploading && !downloading && thumbnailUrl && !thumbLoaded);
  
  const statusInfo = getStatusInfo(project.status, uploading, downloading, stage);

  return (
    <div 
      className={`
        group relative flex flex-col justify-between overflow-hidden h-[260px]
        bg-gradient-to-b from-zinc-900 to-zinc-950 
        border rounded-xl p-6 transition-all duration-200 cursor-pointer 
        ${isSelected ? 'border-brand-500/50 shadow-lg shadow-brand-500/5' : 'border-zinc-800 hover:border-zinc-600 hover:shadow-2xl'}
        ${project.status === 'error' ? 'border-red-500/30' : ''}
      `}
      onClick={onSelect}
    >
      {/* Video Frame Background */}
      {(backgroundImage || isLoadingFrame || processing || downloading) && (
        <>
          {/* Actual frame image or loading placeholder */}
          {backgroundImage ? (
            <div 
              className="absolute inset-0 bg-cover bg-center transition-opacity duration-500"
              style={{ 
                backgroundImage: `url(${backgroundImage})`,
                opacity: processing ? 0.4 : 0.25,
                filter: 'blur(3px) saturate(1.3) brightness(0.9)',
                transform: 'scale(1.05)'
              }}
            />
          ) : (
            /* Placeholder gradient while loading */
            <div 
              className="absolute inset-0 transition-opacity duration-500"
              style={{ 
                background: processing 
                  ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%)'
                  : downloading
                    ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.05) 100%)'
                    : 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(16, 185, 129, 0.05) 100%)',
                opacity: 0.5
              }}
            />
          )}
          {/* Animated glow during processing or downloading */}
          {(processing || downloading) && (
            <div 
              className="absolute inset-0 pointer-events-none transition-all duration-300"
              style={{
                background: `radial-gradient(ellipse at ${Math.min(progress, 100)}% 50%, 
                  ${downloading ? 'rgba(59, 130, 246, 0.25)' : 'rgba(245, 158, 11, 0.25)'} 0%, 
                  transparent 50%)`
              }}
            />
          )}
          {/* Gradient overlay for readability */}
          <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-zinc-950/60 to-zinc-950/20" />
        </>
      )}

      {/* Content (relative to stay above background) */}
      <div className="relative z-10 flex flex-col h-full justify-between flex-1">
        {project.status === 'error' && isCookieError(project.error) ? (
          <div className="flex flex-col h-full justify-between gap-4 flex-1">
            {/* Simple header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div 
                  className="z-20 cursor-pointer" 
                  onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
                >
                  {isSelected ? (
                    <div className="w-5 h-5 bg-brand-500 rounded border border-brand-400 flex items-center justify-center text-white shadow-sm">
                      <CheckSquare className="w-3.5 h-3.5" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 border border-zinc-600 rounded bg-zinc-900/50 hover:border-zinc-400 transition-colors" />
                  )}
                </div>
                <h3 className="text-sm font-bold text-white truncate max-w-[180px]">{project.name}</h3>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(); }}
                className="p-1.5 text-zinc-600 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors opacity-0 group-hover:opacity-100 backdrop-blur-sm z-20"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {/* Full-size Drag & Drop Zone */}
            <div 
              onDragOver={handleCookieDragOver}
              onDragLeave={handleCookieDragLeave}
              onDrop={handleCookieDrop}
              onClick={(e) => e.stopPropagation()}
              className={`
                flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 transition-all duration-200 text-center cursor-default select-none my-1 min-h-[130px]
                ${isDraggingCookie 
                  ? 'border-brand-500 bg-brand-500/10 text-brand-400' 
                  : 'border-amber-500/30 bg-amber-500/5 text-zinc-400 hover:border-amber-500/50'}
              `}
            >
              {isUploadingCookie ? (
                <div className="flex flex-col items-center gap-2 py-2">
                  <Loader2 className="w-6 h-6 text-amber-500 animate-spin" />
                  <span className="text-[11px] font-semibold text-zinc-300">Uploading cookies...</span>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <Cookie className="w-7 h-7 text-amber-500 mx-auto animate-pulse" />
                  <p className="text-[11px] font-bold text-zinc-200">
                    YouTube Authentication Required
                  </p>
                  <p className="text-[9px] text-zinc-500 leading-normal max-w-[240px] mx-auto flex items-center justify-center flex-wrap gap-x-1">
                    <span>Drag & drop a <span className="text-zinc-300 font-semibold">cookies.txt</span> here, or install our</span>
                    <a href="/api/v1/settings/extension/download" onClick={() => { alert("To install the extension:\n1. Download and save the extension ZIP file.\n2. Extract/Unzip the ZIP file into a folder.\n3. Open Chrome or Brave Extension Settings (chrome://extensions).\n4. Enable Developer Mode (toggle in top-right).\n5. Click 'Load unpacked' (button in top-left).\n6. Select the extracted folder."); }} download className="text-brand-400 hover:underline font-semibold">Chrome Extension</a>
                    <span>to sync automatically.</span>
                    <span className="relative inline-block group/tooltip ml-0.5 align-middle">
                      <HelpCircle className="w-3.5 h-3.5 text-zinc-500 hover:text-zinc-300 cursor-help" />
                      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 bg-zinc-950 border border-zinc-800 text-zinc-300 text-[10px] p-3 rounded-lg shadow-xl opacity-0 invisible group-hover/tooltip:opacity-100 group-hover/tooltip:visible transition-all duration-200 z-50 text-left leading-relaxed">
                        <span className="block font-bold text-zinc-100 mb-1">Extension Installation Guide:</span>
                        <ol className="list-decimal list-inside space-y-1">
                          <li>Download & save the ZIP file.</li>
                          <li>Extract the ZIP into a folder.</li>
                          <li>Open <span className="text-brand-400 font-semibold">chrome://extensions</span></li>
                          <li>Enable <b>Developer Mode</b> (top-right).</li>
                          <li>Click <b>Load unpacked</b> (top-left).</li>
                          <li>Select the extracted folder.</li>
                        </ol>
                      </span>
                    </span>
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Top Row: Checkbox, Icon, Action */}
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-start gap-4">
                {/* Selection Checkbox */}
                <div 
                  className="mt-1 z-20 cursor-pointer" 
                  onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
                >
                  {isSelected ? (
                    <div className="w-5 h-5 bg-brand-500 rounded border border-brand-400 flex items-center justify-center text-white shadow-sm">
                      <CheckSquare className="w-3.5 h-3.5" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 border border-zinc-600 rounded bg-zinc-900/50 hover:border-zinc-400 transition-colors" />
                  )}
                </div>

                {/* Icon Box */}
                <div className={`
                  w-10 h-10 rounded-lg flex items-center justify-center border backdrop-blur-sm
                  ${uploading || downloading ? 'bg-blue-500/10 border-blue-500/20 text-blue-500' :
                    project.status === 'dubbed' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500' : 
                    project.status === 'transcribed' ? 'bg-brand-500/10 border-brand-500/20 text-brand-500' : 
                    project.status === 'error' ? 'bg-red-500/10 border-red-500/20 text-red-500' :
                    processing ? 'bg-amber-500/10 border-amber-500/20 text-amber-500' :
                    'bg-zinc-800/50 border-zinc-700/50 text-zinc-400'}
                `}>
                  {uploading || downloading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : project.status === 'error' ? (
                    <AlertCircle className="w-5 h-5" />
                  ) : processing ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : project.status === 'draft' ? (
                    <Video className="w-5 h-5" />
                  ) : (
                    <FileText className="w-5 h-5" />
                  )}
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex items-center gap-1">
                {!processing && project.status !== 'dubbed' && project.status !== 'error' && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onAutoProcess(); }}
                    className={`p-2 rounded-lg transition-colors backdrop-blur-sm ${
                      project.config?.autoProcess
                        ? 'text-amber-400 bg-amber-500/10 hover:text-amber-300 hover:bg-amber-500/20 animate-pulse'
                        : 'text-zinc-400 hover:text-white hover:bg-zinc-800/80'
                    }`}
                    title={project.config?.autoProcess ? "Auto-processing queued" : "Feeling Lucky (Auto-Process)"}
                  >
                    <Sparkles className="w-4 h-4" />
                  </button>
                )}
                
                {(processing || downloading || (uploading && project.config?.autoProcess)) && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onStopProcess(); }}
                    className="p-2 text-amber-400 hover:text-white hover:bg-amber-500/20 rounded-lg transition-colors animate-pulse backdrop-blur-sm"
                    title="Stop Processing"
                  >
                    <StopCircle className="w-4 h-4" />
                  </button>
                )}

                <button
                  onClick={(e) => { e.stopPropagation(); onDelete(); }}
                  className="p-2 text-zinc-600 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors opacity-0 group-hover:opacity-100 backdrop-blur-sm"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Content Body */}
            <div className="mb-6 space-y-1.5">
              <h3 className="text-base font-bold text-white truncate pr-2">{project.name}</h3>
              
              {uploading ? (
                <div className="space-y-2 mt-2">
                  <div className="flex justify-between text-[10px] font-medium uppercase tracking-wider">
                    <span className="text-blue-400">Uploading video...</span>
                    <span className="text-zinc-500">{Math.round(project.uploadProgress || 0)}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-800/80 rounded-full overflow-hidden backdrop-blur-sm">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300 shadow-lg shadow-blue-500/30"
                      style={{ width: `${project.uploadProgress || 0}%` }}
                    />
                  </div>
                </div>
              ) : downloading ? (
                <div className="space-y-2 mt-2">
                  <div className="flex justify-between text-[10px] font-medium uppercase tracking-wider">
                    <span className="text-blue-400">{stage || 'Downloading video...'}</span>
                    <span className="text-zinc-500">{Math.round(progress)}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-800/80 rounded-full overflow-hidden backdrop-blur-sm">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300 shadow-lg shadow-blue-500/30"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : processing ? (
                <div className="space-y-2 mt-2">
                  <div className="flex justify-between text-[10px] font-medium uppercase tracking-wider">
                    <span className="text-brand-400">{stage || 'Processing...'}</span>
                    <span className="text-zinc-500">{Math.round(progress)}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-800/80 rounded-full overflow-hidden backdrop-blur-sm">
                    <div 
                      className="h-full bg-gradient-to-r from-brand-600 to-brand-400 transition-all duration-300 shadow-lg shadow-brand-500/30"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : project.status === 'error' ? (
                <div className="mt-2 group">
                  <div className="text-xs text-red-400 line-clamp-2 leading-relaxed" title={project.error || 'An error occurred'}>
                    {project.error || 'An error occurred'}
                  </div>
                  {project.error && project.error.length > 80 && (
                    <div className="hidden group-hover:block absolute z-50 max-w-sm mt-1 p-2 text-[11px] text-red-200 bg-red-950/90 border border-red-800 rounded shadow-lg">
                      {project.error}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-zinc-500">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{project.updatedAt.toLocaleDateString()} {project.updatedAt.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                  </div>
                  {(() => {
                    const meta = buildProjectMetaLine(project);
                    if (!meta) return null;
                    return (
                      <div className="text-[10px] text-zinc-500 leading-tight truncate" title={meta}>
                        {meta}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>

            {/* Footer: Badges */}
            <div className="mt-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className={`
                  px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider border backdrop-blur-sm
                  ${statusInfo.color}
                `}>
                  {statusInfo.label}
                </span>
                
                {/* Language Selectors */}
                {!processing && !uploading && project.status !== 'dubbed' ? (
                  <div 
                    className="flex items-center gap-1 z-20"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <CompactLangSelector
                      value={project.config.sourceLang}
                      onChange={(val) => onChangeLanguages(val, project.config.targetLang)}
                      isSource
                    />
                    <span className="text-zinc-600 text-xs">→</span>
                    <CompactLangSelector
                      value={project.config.targetLang}
                      onChange={(val) => onChangeLanguages(project.config.sourceLang, val)}
                    />
                  </div>
                ) : (
                  <span className="text-xs font-medium text-zinc-500 uppercase">
                    {LANG_FLAGS[project.config.sourceLang]} {project.config.sourceLang} <span className="text-zinc-600">→</span> {LANG_FLAGS[project.config.targetLang]} {project.config.targetLang}
                  </span>
                )}
              </div>
              
              {/* Preset Selector */}
              {!processing && !uploading && project.status !== 'dubbed' && (
                <div 
                  className="flex items-center gap-1 z-20"
                  onClick={(e) => e.stopPropagation()}
                >
                  {PRESETS.map((preset) => {
                    const isPresetSelected = (project.config.preset || 'hq') === preset.id;
                    return (
                      <button
                        key={preset.id}
                        onClick={() => onChangePreset(preset.id)}
                        className={`
                          w-7 h-7 rounded-md flex items-center justify-center text-sm transition-all backdrop-blur-sm
                          ${isPresetSelected 
                            ? 'bg-brand-500/20 border border-brand-500/40 shadow-sm' 
                            : 'bg-zinc-800/50 border border-transparent hover:bg-zinc-800 hover:border-zinc-700'}
                        `}
                        title={`${preset.name} - ${preset.description}`}
                      >
                        {preset.icon}
                      </button>
                    );
                  })}
                </div>
              )}
              
              {/* Show current preset badge when processing or done */}
              {(processing || uploading || project.status === 'dubbed') && (
                <span className="text-xs text-zinc-500 flex items-center gap-1">
                  {PRESETS.find(p => p.id === (project.config.preset || 'hq'))?.icon}
                  {PRESETS.find(p => p.id === (project.config.preset || 'hq'))?.name}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const getStatusInfo = (status: ProjectStatus, isUploading?: boolean, isDownloading?: boolean, processStage?: string) => {
  if (isUploading) {
    return { label: 'Uploading', color: 'bg-blue-500/10 text-blue-400 border-blue-500/20' };
  }

  if (isDownloading) {
    return { label: 'Downloading', color: 'bg-blue-500/10 text-blue-400 border-blue-500/20' };
  }
  
  // When processing, show more accurate stage based on processStage
  if (status === 'transcribing' && processStage) {
    const stageLower = processStage.toLowerCase();
    if (stageLower.includes('translat')) {
      return { label: 'Translating', color: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20' };
    }
    if (stageLower.includes('refin')) {
      return { label: 'Refining', color: 'bg-teal-500/10 text-teal-400 border-teal-500/20' };
    }
    if (stageLower.includes('editor')) {
      return { label: 'Editing', color: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' };
    }
  }
  
  switch (status) {
    case 'draft':
      return { label: 'Draft', color: 'bg-zinc-800 text-zinc-400 border-zinc-700' };
    case 'transcribing':
      return { label: 'Transcribing', color: 'bg-amber-500/10 text-amber-400 border-amber-500/20' };
    case 'transcribed':
      return { label: 'Ready', color: 'bg-brand-500/10 text-brand-400 border-brand-500/20' };
    case 'dubbing':
      return { label: 'Dubbing', color: 'bg-purple-500/10 text-purple-400 border-purple-500/20' };
    case 'dubbed':
      return { label: 'Done', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' };
    case 'error':
      return { label: 'Error', color: 'bg-red-500/10 text-red-400 border-red-500/20' };
    default:
      return { label: status, color: 'bg-zinc-800 text-zinc-400 border-zinc-700' };
  }
};

export const ProjectListView: React.FC<ProjectListViewProps> = ({
  projects,
  onSelectProject,
  onDeleteProject,
  onBatchUpload,
  onDownloadFromUrl,
  onAutoProcess,
  onStopProcess,
  onResetAndRestart,
  onOpenSettings,
  onChangePreset,
  onChangeLanguages,
  onProgressUpdate,
  onDownloadProgressUpdate,
  onStatusComplete
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [urlInput, setUrlInput] = useState('');
  const [downloadQuality, setDownloadQuality] = useState<'best' | '1080p' | '720p' | '480p'>('best');
  const [isSubmittingUrl, setIsSubmittingUrl] = useState(false);
  const [isAddMediaOpen, setIsAddMediaOpen] = useState(false);
  const [isExtensionInstalled, setIsExtensionInstalled] = useState(false);
  const [cookiesStatus, setCookiesStatus] = useState<{ configured: boolean; source: string } | null>(null);
  const [isCookiesLoading, setIsCookiesLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dynamically detect when the Chrome extension is installed/active
  useEffect(() => {
    const checkPresence = () => {
      const installed = document.getElementById('dubblm-extension-installed') !== null;
      setIsExtensionInstalled(installed);
    };
    checkPresence();

    const handleInstalled = () => {
      console.log('Extension presence detected!');
      setIsExtensionInstalled(true);
    };

    window.addEventListener('DUBBLM_EXTENSION_INSTALLED', handleInstalled);
    window.dispatchEvent(new CustomEvent('DUBBLM_PING_EXTENSION'));

    return () => {
      window.removeEventListener('DUBBLM_EXTENSION_INSTALLED', handleInstalled);
    };
  }, []);

  const syncCookies = (): Promise<void> => {
    return new Promise((resolve) => {
      const timeoutId = setTimeout(() => {
        console.warn('Cookie sync timed out, proceeding anyway...');
        cleanup();
        resolve();
      }, 5000); // 5-second safety timeout

      const cleanup = () => {
        clearTimeout(timeoutId);
        window.removeEventListener('DUBBLM_SYNC_SUCCESS', handleSuccess);
        window.removeEventListener('DUBBLM_SYNC_ERROR', handleError);
      };

      const handleSuccess = () => {
        console.log('Cookie sync succeeded!');
        cleanup();
        resolve();
      };

      const handleError = (e: Event) => {
        const err = (e as CustomEvent).detail?.error;
        console.warn('Cookie sync failed:', err);
        cleanup();
        resolve(); // Resolve anyway so that we don't block the user's download
      };

      window.addEventListener('DUBBLM_SYNC_SUCCESS', handleSuccess);
      window.addEventListener('DUBBLM_SYNC_ERROR', handleError);
      
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_COOKIES', { 
        detail: { serverUrl: window.location.origin } 
      }));
    });
  };

  // Helper to validate whether a string is a video URL
  const isValidVideoUrl = (text: string): boolean => {
    try {
      const url = new URL(text);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch {
      return false;
    }
  };

  // Automatically read clipboard and pre-fill URL input if it contains a valid video URL
  useEffect(() => {
    if (isAddMediaOpen) {
      if (navigator.clipboard && typeof navigator.clipboard.readText === 'function') {
        navigator.clipboard.readText()
          .then((text) => {
            const trimmed = text.trim();
            if (isValidVideoUrl(trimmed)) {
              console.log('Auto-detected valid video URL in clipboard, filling input:', trimmed);
              setUrlInput(trimmed);
            }
          })
          .catch((err) => {
            console.warn('Clipboard read permissions denied or unsupported:', err);
          });
      }
    }
  }, [isAddMediaOpen]);
  const cookiesInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const loadCookiesStatus = async () => {
      try {
        setCookiesStatus(await api.getCookiesStatus());
      } catch {
        setCookiesStatus(null);
      }
    };
    loadCookiesStatus();
  }, []);

  const handleBatchSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onBatchUpload(Array.from(e.target.files));
      e.target.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const videoFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'));
      if (videoFiles.length > 0) {
        onBatchUpload(videoFiles);
      }
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!urlInput.trim() || isSubmittingUrl) return;

    setIsSubmittingUrl(true);
    try {
      if (isExtensionInstalled) {
        console.log('Requesting automatic cookie sync before download...');
        await syncCookies();
      }
      await onDownloadFromUrl(urlInput.trim(), downloadQuality);
      setUrlInput('');
    } finally {
      setIsSubmittingUrl(false);
    }
  };

  const handleCookiesSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsCookiesLoading(true);
    try {
      await api.uploadCookies(file);
      setCookiesStatus(await api.getCookiesStatus());
    } catch (err) {
      console.error('Failed to upload cookies:', err);
      alert(err instanceof Error ? err.message : 'Failed to upload cookies');
    } finally {
      setIsCookiesLoading(false);
      e.target.value = '';
    }
  };

  const handleDeleteCookies = async () => {
    setIsCookiesLoading(true);
    try {
      await api.deleteCookies();
      setCookiesStatus(await api.getCookiesStatus());
    } catch (err) {
      console.error('Failed to delete cookies:', err);
      alert(err instanceof Error ? err.message : 'Failed to delete cookies');
    } finally {
      setIsCookiesLoading(false);
    }
  };

  const toggleSelection = (id: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === projects.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(projects.map(p => p.id)));
    }
  };

  const handleBulkAutoProcess = () => {
    onAutoProcess(Array.from(selectedIds));
    setSelectedIds(new Set());
  };

  const handleBulkStop = () => {
    onStopProcess(Array.from(selectedIds));
    setSelectedIds(new Set());
  };

  const handleBulkDelete = () => {
    Array.from(selectedIds).forEach(id => onDeleteProject(id));
    setSelectedIds(new Set());
  };

  const isProcessing = (status: ProjectStatus) => 
    status === 'transcribing' || status === 'dubbing';

  return (
    <div 
      className="h-full overflow-y-auto bg-zinc-950 p-10 relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-brand-500/10 backdrop-blur-sm z-50 flex items-center justify-center border-4 border-dashed border-brand-500/50 m-6 rounded-3xl pointer-events-none">
          <div className="text-center animate-bounce">
            <Upload className="w-16 h-16 text-brand-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white">Drop videos to add media</h2>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto space-y-10 pb-32">
        
        {/* Header */}
        <div className="flex items-end justify-between border-b border-zinc-800/50 pb-6">
          <div className="flex items-center gap-6">
            <img 
                src="https://github.com/ArteusAI/DubbLM/blob/master/logo.png?raw=true" 
                alt="DubbLM Logo" 
                className="w-12 h-12 rounded-xl"
            />
            <div>
              <h1 className="text-4xl font-extrabold text-white tracking-tight mb-2">DubbLM</h1>
              <p className="text-zinc-400">Manage your dubbing projects</p>
            </div>
            {projects.length > 0 && (
              <button 
                onClick={toggleSelectAll}
                className="flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-zinc-300 transition-colors ml-4 mb-1"
              >
                {selectedIds.size === projects.length && projects.length > 0 ? <CheckSquare className="w-4 h-4 text-brand-500" /> : <Square className="w-4 h-4" />}
                Select All
              </button>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={onOpenSettings}
              className="p-2.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors border border-zinc-800 hover:border-zinc-700"
              title="Global Settings"
            >
              <Settings className="w-5 h-5" />
            </button>
            <input 
                type="file"
                multiple
                accept="video/*"
                className="hidden"
                ref={fileInputRef}
                onChange={handleBatchSelect}
            />
            
            <div className="relative">
              <button
                onClick={() => setIsAddMediaOpen(!isAddMediaOpen)}
                className="flex items-center gap-2 px-5 py-2.5 bg-brand-600 hover:bg-brand-500 text-white rounded-lg font-bold transition-all shadow-lg shadow-brand-500/20 active:scale-95"
              >
                <Plus className="w-5 h-5" />
                Add Media
                <ChevronDown className="w-4 h-4 ml-0.5 opacity-80" />
              </button>
              
              {isAddMediaOpen && (
                <>
                  <div className="fixed inset-0 z-20" onClick={() => setIsAddMediaOpen(false)} />
                  
                  <div className="absolute right-0 mt-2 w-80 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl z-30 p-4 space-y-4 animate-in fade-in slide-in-from-top-2 duration-150">
                    <div>
                      <span className="block text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2">Local Files</span>
                      <button
                        onClick={() => {
                          setIsAddMediaOpen(false);
                          fileInputRef.current?.click();
                        }}
                        className="flex items-center gap-3 w-full p-2.5 rounded-xl bg-zinc-950/40 border border-zinc-800/60 hover:border-brand-500/50 hover:bg-brand-500/5 transition-all text-left group"
                      >
                        <div className="w-10 h-10 rounded-lg bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400 group-hover:bg-brand-500/20 transition-all">
                          <Upload className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="font-semibold text-sm text-zinc-200">Upload from Device</div>
                          <div className="text-[10px] text-zinc-500">Video & audio files</div>
                        </div>
                      </button>
                    </div>
                    
                    <div className="border-t border-zinc-800/80" />
                    
                    <div>
                      <span className="block text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2">From Link</span>
                      <form                       onSubmit={async (e) => {
                        e.preventDefault();
                        if (!urlInput.trim()) return;
                        
                        setIsSubmittingUrl(true);
                        setIsAddMediaOpen(false);
                        
                        try {
                          if (isExtensionInstalled) {
                            console.log('Requesting automatic cookie sync before download...');
                            await syncCookies();
                          }
                          await onDownloadFromUrl(urlInput.trim(), downloadQuality);
                          setUrlInput('');
                        } catch (err) {
                          console.error(err);
                        } finally {
                          setIsSubmittingUrl(false);
                        }
                      }} className="space-y-3">
                        <div className="flex items-center bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 focus-within:border-brand-500/50 focus-within:ring-1 focus-within:ring-brand-500/20 transition-all">
                          <Link2 className="w-4 h-4 text-zinc-500 mr-2 shrink-0" />
                          <input
                            type="url"
                            required
                            placeholder="Paste YouTube, Vimeo URL..."
                            value={urlInput}
                            onChange={(e) => setUrlInput(e.target.value)}
                            className="bg-transparent border-none outline-none text-xs text-white placeholder-zinc-600 w-full"
                          />
                        </div>
                        
                        <div className="flex gap-2">
                          <select
                            value={downloadQuality}
                            onChange={(e) => setDownloadQuality(e.target.value as 'best' | '1080p' | '720p' | '480p')}
                            className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl px-2.5 py-1.5 text-xs text-zinc-300 outline-none focus:border-brand-500/50"
                            title="Download quality"
                          >
                            <option value="best">Best Quality</option>
                            <option value="1080p">1080p HD</option>
                            <option value="720p">720p HD</option>
                            <option value="480p">480p SD</option>
                          </select>
                          
                          <button
                            type="submit"
                            disabled={!urlInput.trim() || isSubmittingUrl}
                            className="flex items-center justify-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold text-xs transition-colors shadow-lg shadow-blue-500/20"
                          >
                            {isSubmittingUrl ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Download className="w-3.5 h-3.5" />
                            )}
                            Get
                          </button>
                        </div>
                      </form>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Project Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {projects.length === 0 ? (
            <div className="col-span-full flex flex-col items-center justify-center py-32 text-zinc-500 border-2 border-dashed border-zinc-800 rounded-3xl bg-zinc-900/30">
              <div className="w-20 h-20 bg-zinc-900 rounded-full flex items-center justify-center mb-6 shadow-inner">
                <FolderOpen className="w-10 h-10 text-zinc-600" />
              </div>
              <p className="text-xl font-semibold text-zinc-300">No projects yet</p>
              <p className="text-sm mt-2 max-w-sm text-center">Get started by clicking "Add Media", dragging your video files directly onto this area, or pasting a video URL above.</p>
            </div>
          ) : (
            projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                isSelected={selectedIds.has(project.id)}
                isExtensionInstalled={isExtensionInstalled}
                onSelect={() => onSelectProject(project.id)}
                onToggleSelect={() => toggleSelection(project.id)}
                onDelete={() => onDeleteProject(project.id)}
                onAutoProcess={() => onAutoProcess([project.id])}
                onStopProcess={() => onStopProcess([project.id])}
                onResetAndRestart={() => onResetAndRestart(project.id)}
                onRetryDownload={() => onRetryDownload(project.id)}
                onChangePreset={(preset) => onChangePreset(project.id, preset)}
                onChangeLanguages={(source, target) => onChangeLanguages(project.id, source, target)}
                onProgressUpdate={(progress, stage) => onProgressUpdate(project.id, progress, stage)}
                onDownloadProgressUpdate={(progress, stage) => onDownloadProgressUpdate(project.id, progress, stage)}
                onStatusComplete={(status, error) => onStatusComplete(project.id, status, error)}
              />
            ))
          )}
        </div>
      </div>

      {/* Bulk Action Bar */}
      {selectedIds.size > 0 && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-zinc-950/95 backdrop-blur-xl border border-zinc-900 rounded-2xl shadow-[0_0_40px_rgba(0,0,0,0.95)] px-3 py-2 flex items-center gap-2 z-50 animate-in slide-in-from-bottom-6 duration-300 ring-1 ring-black/40">
          <div className="pl-3 pr-2 py-1.5 border-r border-zinc-800">
            <span className="text-xs font-bold text-white">{selectedIds.size} selected</span>
          </div>
          
          <button 
            onClick={handleBulkAutoProcess}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-zinc-300 hover:text-white hover:bg-white/5 transition-colors"
          >
            <Sparkles className="w-4 h-4 text-brand-400" />
            Feeling Lucky
          </button>

          <button 
            onClick={handleBulkStop}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-zinc-300 hover:text-white hover:bg-white/5 transition-colors"
          >
            <StopCircle className="w-4 h-4 text-amber-400" />
            Stop
          </button>

          <div className="w-px h-6 bg-zinc-800 mx-1"></div>

          <button 
            onClick={handleBulkDelete}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-zinc-300 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        </div>
      )}
    </div>
  );
};
