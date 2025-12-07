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
  // Also match "Pause removal: 50%" format
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

// Inline progress bar component
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

// Process logs to group progress updates
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

  // Process and deduplicate progress logs - group by label, keep only latest
  const processedLogs = useMemo(() => {
    const result: ProcessedLog[] = [];
    
    logs.forEach((log) => {
      const progressInfo = parseProgressFromMessage(log.message);
      
      if (progressInfo) {
        // Find existing progress log OR plain message that matches the label
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
        // Check if this plain message will be replaced by a progress message
        const willBeReplacedIndex = result.findIndex(
          l => l.progressInfo?.label === log.message
        );
        
        if (willBeReplacedIndex >= 0) {
          // Skip - there's already a progress bar for this
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

  // Auto-complete when progress reaches 100%
  useEffect(() => {
    if (progress >= 100 && !hasCompleted.current) {
      hasCompleted.current = true;
      setTimeout(onComplete, 800);
    }
  }, [progress, onComplete]);

  const isComplete = progress >= 100;

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-4xl mx-auto px-6">
      <div className="w-full space-y-8">
        <div className="text-center space-y-6">
          <div className="inline-flex items-center justify-center mb-4">
            {isComplete ? (
              <div className="w-[115px] h-[115px] flex items-center justify-center rounded-full bg-emerald-500/10">
                <CheckCircle2 className="w-10 h-10 text-emerald-500" />
              </div>
            ) : (
              <BlackHoleProgress progress={progress} size={115} />
            )}
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">{title}</h2>
          <p className="text-zinc-400 text-lg">{description}</p>
        </div>

        <div className="bg-zinc-950 rounded-xl border border-zinc-800 overflow-hidden shadow-inner">
          <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/50 border-b border-zinc-800">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-zinc-500" />
              <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider">System Logs</span>
            </div>
            <span className="text-xs font-medium text-brand-400 uppercase tracking-wider">{currentStep || 'Processing...'}</span>
          </div>
          <div className="h-64 overflow-y-auto p-4 font-mono text-sm space-y-2">
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
  );
};
