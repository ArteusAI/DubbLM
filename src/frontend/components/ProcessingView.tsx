import React, { useEffect, useRef, useMemo } from 'react';
import { Terminal, CheckCircle2, AlertCircle } from 'lucide-react';
import { ProcessingLog } from '../types';
import { BlackHoleProgress } from './BlackHoleProgress';

interface ProcessingViewProps {
  title: string;
  description: string;
  projectId: string;
  onComplete: () => void;
  logs: ProcessingLog[];
  progress: number;
  currentStep: string;
}

// Parse progress from message like "Processing background audio: 4/289"
const parseProgressFromMessage = (message: string): { label: string; current: number; total: number } | null => {
  const match = message.match(/^(.+?):\s*(\d+)\s*\/\s*(\d+)\s*[s]?$/);
  if (match) {
    return {
      label: match[1].trim(),
      current: parseInt(match[2], 10),
      total: parseInt(match[3], 10),
    };
  }
  const percentMatch = message.match(/^(.+?):\s*(\d+)%$/);
  if (percentMatch) {
    return {
      label: percentMatch[1].trim(),
      current: parseInt(percentMatch[2], 10),
      total: 100,
    };
  }
  return null;
};

const InlineProgressBar: React.FC<{ current: number; total: number; label: string }> = ({ current, total, label }) => {
  const percent = Math.min(100, Math.round((current / total) * 100));

  return (
    <div className="flex items-center gap-3 flex-1 min-w-0">
      <span className="text-zinc-300 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden min-w-[100px] max-w-[200px]">
        <div
          className="h-full bg-gradient-to-r from-brand-500 to-brand-400 transition-all duration-300 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="text-zinc-500 text-xs shrink-0 tabular-nums w-16 text-right">
        {current}/{total}
      </span>
    </div>
  );
};

interface ProcessedLog extends ProcessingLog {
  progressInfo?: { label: string; current: number; total: number };
}

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  title,
  description,
  projectId,
  onComplete,
  logs,
  progress,
  currentStep,
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const hasCompleted = useRef(false);

  const processedLogs = useMemo(() => {
    const result: ProcessedLog[] = [];

    logs.forEach((log) => {
      const progressInfo = parseProgressFromMessage(log.message);

      if (progressInfo) {
        const existingIndex = result.findIndex(
          l => l.progressInfo?.label === progressInfo.label ||
               l.message === progressInfo.label ||
               l.message.startsWith(progressInfo.label + ':')
        );

        const processedLog: ProcessedLog = { ...log, progressInfo };

        if (existingIndex >= 0) {
          result[existingIndex] = processedLog;
        } else {
          result.push(processedLog);
        }
      } else {
        const willBeReplacedIndex = result.findIndex(
          l => l.progressInfo?.label === log.message
        );

        if (willBeReplacedIndex >= 0) {
          return;
        }

        result.push({ ...log });
      }
    });

    return result;
  }, [logs]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [processedLogs]);

  useEffect(() => {
    if (progress >= 100 && !hasCompleted.current) {
      hasCompleted.current = true;
      setTimeout(onComplete, 800);
    }
  }, [progress, onComplete]);

  const isComplete = progress >= 100;

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-4xl mx-auto px-6">
      {/* Responsive CSS for small screens */}
      <style>{`
        .processing-spacer { height: 130px; }
        .processing-logs { height: 16rem; }
        @media (max-height: 700px) {
          .processing-spacer { height: 90px; }
          .processing-logs { height: 10rem; }
        }
        @media (max-height: 550px) {
          .processing-spacer { height: 60px; }
          .processing-logs { height: 8rem; }
        }
      `}</style>
      <div className="w-full relative">

        {/* Circle layer — behind everything, top-aligned, grows downward */}
        {!isComplete && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 0,
              pointerEvents: 'none',
            }}
          >
            <BlackHoleProgress
              progress={progress}
              growable={true}
              minSize={115}
              maxSize={520}
            />
          </div>
        )}

        {/* Content layer — on top of circle */}
        <div className="relative" style={{ zIndex: 1 }}>
          {/* Space for the visible top of the circle */}
          {!isComplete && <div className="processing-spacer" />}

          <div className="space-y-4" style={{ position: 'relative', zIndex: 1 }}>
            {isComplete && (
              <div className="text-center mb-4 pt-6">
                <div className="inline-flex items-center justify-center w-[115px] h-[115px] rounded-full bg-emerald-500/10">
                  <CheckCircle2 className="w-10 h-10 text-emerald-500" />
                </div>
              </div>
            )}

            <div className="text-center pt-2">
              <h2 className="text-3xl font-bold text-white tracking-tight">{title}</h2>
            </div>

            <div className="rounded-xl border border-zinc-800 overflow-hidden shadow-inner" style={{ position: 'relative', zIndex: 3, background: 'rgba(9,9,11,0.95)' }}>
              <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800" style={{ background: 'rgba(24,24,27,0.5)' }}>
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-zinc-500" />
                  <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider">System Logs</span>
                </div>
                <span className="text-xs font-medium text-brand-400 uppercase tracking-wider">{currentStep || 'Processing...'}</span>
              </div>
              <div className="processing-logs overflow-y-auto p-4 font-mono text-sm space-y-2">
                {processedLogs.length === 0 ? (
                  <div className="text-zinc-600 text-center py-8">
                    Waiting for logs...
                  </div>
                ) : (
                  processedLogs.map((log) => (
                    <div key={log.id} className="flex items-start gap-3 animate-in slide-in-from-left-2 duration-300 min-w-0">
                      <span className="text-zinc-600 text-xs shrink-0 pt-1">
                        {log.timestamp.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' })}
                      </span>
                      <span className={`flex items-start gap-2 min-w-0 flex-1 ${
                        log.type === 'error' ? 'text-red-400' :
                        log.type === 'success' ? 'text-emerald-400' :
                        'text-zinc-300'
                      }`}>
                        {log.type === 'success' && <CheckCircle2 className="w-3 h-3 shrink-0 mt-1" />}
                        {log.type === 'error' && <AlertCircle className="w-3 h-3 shrink-0 mt-1" />}
                        {log.progressInfo ? (
                          <InlineProgressBar
                            label={log.progressInfo.label}
                            current={log.progressInfo.current}
                            total={log.progressInfo.total}
                          />
                        ) : (
                          <span className="min-w-0 flex-1">
                            <span>{log.message}</span>
                            {log.text && (
                              <span className="text-sky-400/80 italic ml-1 truncate inline-block max-w-[400px] align-bottom">
                                "{log.text}"
                              </span>
                            )}
                          </span>
                        )}
                      </span>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
