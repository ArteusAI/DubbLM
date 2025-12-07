import React, { useState, useEffect } from 'react';
import { Download, FileText, ArrowLeft, CheckCircle, Clock, DollarSign } from 'lucide-react';
import api, { ProjectStatsResponse } from '../api';

interface ResultViewProps {
  projectId: string;
  onReset: () => void;
}

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
  
  const videoUrl = api.getDownloadVideoUrl(projectId);
  const sourceSrtUrl = api.getDownloadSubtitlesUrl(projectId, 'srt', 'source');
  const targetSrtUrl = api.getDownloadSubtitlesUrl(projectId, 'srt', 'target');

  useEffect(() => {
    api.getProjectStats(projectId)
      .then(setStats)
      .catch(() => setStats(null));
  }, [projectId]);

  const handleDownloadVideo = () => {
    const link = document.createElement('a');
    link.href = videoUrl;
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

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-8 p-6">
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

      <div className="w-full max-w-3xl bg-black rounded-xl overflow-hidden aspect-video border border-zinc-800 shadow-2xl relative group">
        <video 
          src={videoUrl}
          controls
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
    </div>
  );
};
