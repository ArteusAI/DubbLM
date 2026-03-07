
import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, Pause, Wand2, Volume2, VolumeX, 
  ChevronRight, RefreshCw, ArrowRight, Loader2, Speaker, Sparkles, X, Clock, Pencil, Check
} from 'lucide-react';
import { Segment, AppConfig } from '../types';
import { VoiceResponse } from '../api';
import api from '../api';

interface EditorViewProps {
  projectId: string;
  segments: Segment[];
  videoFile: File | null;
  voices: VoiceResponse[];
  onUpdateSegments: (segments: Segment[]) => void;
  onUpdateConfig?: (cfg: Partial<AppConfig>) => void;
  onContinue: () => void;
  activeProject?: any; // Add this to access config
}

export const EditorView: React.FC<EditorViewProps> = ({ 
  projectId,
  segments, 
  videoFile, 
  voices,
  onUpdateSegments, 
  onUpdateConfig,
  onContinue,
  activeProject
}) => {
  const [reTranslatePrompt, setReTranslatePrompt] = useState<string>('');
  const [showPromptInputId, setShowPromptInputId] = useState<string | null>(null);
  const [playingSegmentId, setPlayingSegmentId] = useState<string | null>(null);
  const [showTtsPromptId, setShowTtsPromptId] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  
  // Speaker Logic State
  const [editingSpeakerInstance, setEditingSpeakerInstance] = useState<{ segmentId: string, originalName: string } | null>(null);
  const [renameValue, setRenameValue] = useState<string>('');
  const [showSpeakerDropdownId, setShowSpeakerDropdownId] = useState<string | null>(null);
  
  // Resizable Layout State
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    const saved = localStorage.getItem('dubb_sidebar_width');
    return saved ? parseFloat(saved) : 70;
  });
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  // Use ref to track latest speakerTtsPrompts to avoid race conditions with async state updates
  const speakerPromptsRef = useRef<Record<string, string>>(activeProject?.config.speakerTtsPrompts || {});
  
  useEffect(() => {
    speakerPromptsRef.current = activeProject?.config.speakerTtsPrompts || {};
  }, [activeProject?.config.speakerTtsPrompts]);

  // Get unique speakers for the reassignment dropdown
  const uniqueSpeakers: { name: string; color: string }[] = Array.from(
    new Set(segments.map(s => s.speaker))
  ).map((name: string) => {
    const segment = segments.find(s => s.speaker === name);
    return {
      name,
      color: segment?.speakerColor || 'bg-zinc-700 text-zinc-300'
    };
  }).sort((a, b) => a.name.localeCompare(b.name));

  // Map Tailwind color names to CSS glow colors
  const getGlowColor = (speakerColor: string | undefined): string => {
    const colorMap: Record<string, string> = {
      'amber': 'rgba(245, 158, 11, 0.6)',
      'emerald': 'rgba(16, 185, 129, 0.6)',
      'sky': 'rgba(14, 165, 233, 0.6)',
      'violet': 'rgba(139, 92, 246, 0.6)',
      'rose': 'rgba(244, 63, 94, 0.6)',
      'cyan': 'rgba(6, 182, 212, 0.6)',
      'orange': 'rgba(249, 115, 22, 0.6)',
      'pink': 'rgba(236, 72, 153, 0.6)',
      'lime': 'rgba(132, 204, 22, 0.6)',
      'indigo': 'rgba(99, 102, 241, 0.6)',
      'teal': 'rgba(20, 184, 166, 0.6)',
      'fuchsia': 'rgba(217, 70, 239, 0.6)',
      'blue': 'rgba(59, 130, 246, 0.6)',
      'green': 'rgba(34, 197, 94, 0.6)',
      'red': 'rgba(239, 68, 68, 0.6)',
      'yellow': 'rgba(234, 179, 8, 0.6)',
      'purple': 'rgba(168, 85, 247, 0.6)',
    };
    
    if (speakerColor) {
      for (const [name, rgba] of Object.entries(colorMap)) {
        if (speakerColor.includes(name)) return rgba;
      }
    }
    return 'rgba(139, 92, 246, 0.6)'; // violet as default
  };

  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      // Use API URL if no local file
      setVideoUrl(api.getVideoUrl(projectId));
    }
  }, [videoFile, projectId]);

  // Check for cached audio on initial load
  useEffect(() => {
    const checkCachedAudio = async () => {
      const segmentsToCheck = segments.filter(s => !s.audioUrl && s.voiceId);
      if (segmentsToCheck.length === 0) return;

      const updates: { segmentId: string; audioUrl: string }[] = [];
      
      await Promise.all(segmentsToCheck.map(async (segment) => {
        try {
          const status = await api.getPreviewStatus(projectId, segment.id);
          if (status.status === 'completed' && status.audioUrl) {
            updates.push({ segmentId: segment.id, audioUrl: status.audioUrl });
          }
        } catch {
          // Ignore errors
        }
      }));

      if (updates.length > 0) {
        onUpdateSegments(segments.map(s => {
          const update = updates.find(u => u.segmentId === s.id);
          return update ? { ...s, audioUrl: update.audioUrl } : s;
        }));
      }
    };

    checkCachedAudio();
  }, [projectId]); // Only run on project load

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setShowSpeakerDropdownId(null);
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  // Handle Dragging Logic
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
        if (!isDragging || !containerRef.current) return;
        
        const containerRect = containerRef.current.getBoundingClientRect();
        const newWidthPercent = ((e.clientX - containerRect.left) / containerRect.width) * 100;
        
        if (newWidthPercent > 20 && newWidthPercent < 80) {
            setSidebarWidth(newWidthPercent);
        }
    };

    const handleMouseUp = () => {
        if (isDragging) {
            setIsDragging(false);
            localStorage.setItem('dubb_sidebar_width', sidebarWidth.toString());
        }
    };

    if (isDragging) {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = 'col-resize';
    }

    return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
    };
  }, [isDragging, sidebarWidth]);

  const handleUpdateSegment = (id: string, updates: Partial<Segment>) => {
    onUpdateSegments(segments.map(s => s.id === id ? { ...s, ...updates } : s));
  };

  const handleGlobalVoiceChange = async (speakerName: string, newVoiceId: string) => {
    const voice = voices.find(v => v.id === newVoiceId);
    if (!voice) return;

    // Keep project-level speaker voice mappings in sync with editor voice changes.
    const currentMappings = activeProject?.config?.speakerVoiceMappings || {};
    onUpdateConfig?.({
      speakerVoiceMappings: {
        ...currentMappings,
        [speakerName]: voice.id,
      },
    });

    // Get affected segments
    const affectedSegmentIds = segments
      .filter(s => s.speaker === speakerName)
      .map(s => s.id);

    // First update voice and clear audioUrl
    const updatedSegments = segments.map(s => {
      if (s.speaker === speakerName) {
        return {
          ...s,
          voiceId: voice.id,
          provider: voice.provider,
          audioUrl: undefined
        };
      }
      return s;
    });
    onUpdateSegments(updatedSegments);

    // Sync to API
    try {
      await api.updateSpeakerVoice(projectId, speakerName, voice.id, voice.provider);
      
      // Check if any segments have cached audio for the new voice
      const statusChecks = affectedSegmentIds.map(async (segmentId) => {
        try {
          const status = await api.getPreviewStatus(projectId, segmentId);
          if (status.status === 'completed' && status.audioUrl) {
            return { segmentId, audioUrl: status.audioUrl };
          }
        } catch {
          // Ignore errors, segment just won't have audioUrl
        }
        return null;
      });
      
      const results = await Promise.all(statusChecks);
      const cachedSegments = results.filter(Boolean) as { segmentId: string; audioUrl: string }[];
      
      if (cachedSegments.length > 0) {
        // Update segments with cached audio
        onUpdateSegments(segments.map(s => {
          const cached = cachedSegments.find(c => c.segmentId === s.id);
          if (cached) {
            return { ...s, voiceId: voice.id, provider: voice.provider, audioUrl: cached.audioUrl };
          }
          if (s.speaker === speakerName) {
            return { ...s, voiceId: voice.id, provider: voice.provider, audioUrl: undefined };
          }
          return s;
        }));
      }
    } catch (err) {
      console.error('Failed to update speaker voice:', err);
    }
  };

  const handleReTranslate = async (id: string) => {
    const segment = segments.find(s => s.id === id);
    if (!segment) return;

    handleUpdateSegment(id, { isLocked: true });
    
    try {
      const result = await api.rephraseSegment(projectId, id, reTranslatePrompt || undefined);
      handleUpdateSegment(id, { 
        translatedText: result.translatedText,
        isLocked: false,
        audioUrl: undefined
      });
    } catch (err) {
      console.error('Failed to rephrase:', err);
      handleUpdateSegment(id, { isLocked: false });
    }
    
    setShowPromptInputId(null);
    setReTranslatePrompt('');
  };

  // --- Speaker Rename Handlers (Global) ---
  const handleStartRename = (segmentId: string, currentName: string) => {
    setEditingSpeakerInstance({ segmentId, originalName: currentName });
    setRenameValue(currentName);
    setShowSpeakerDropdownId(null);
  };

  const handleCommitRename = async () => {
    if (editingSpeakerInstance && renameValue.trim() && renameValue !== editingSpeakerInstance.originalName) {
      const updatedSegments = segments.map(s => 
        s.speaker === editingSpeakerInstance.originalName
          ? { ...s, speaker: renameValue.trim() }
          : s
      );
      onUpdateSegments(updatedSegments);

      try {
        await api.renameSpeaker(projectId, editingSpeakerInstance.originalName, renameValue.trim());
      } catch (err) {
        console.error('Failed to rename speaker:', err);
      }
    }
    setEditingSpeakerInstance(null);
    setRenameValue('');
  };

  // --- Speaker Reassign Handler (Local) ---
  const handleSegmentReassign = (segmentId: string, newSpeakerName: string) => {
    const targetSpeaker = uniqueSpeakers.find(s => s.name === newSpeakerName);
    handleUpdateSegment(segmentId, { 
        speaker: newSpeakerName,
        speakerColor: targetSpeaker?.color
    });
    setShowSpeakerDropdownId(null);
  };

  const handleSpeakerPromptChange = (speakerName: string, prompt: string) => {
    const currentPrompts = { ...speakerPromptsRef.current };
    if (prompt) {
      currentPrompts[speakerName] = prompt;
    } else {
      delete currentPrompts[speakerName];
    }
    speakerPromptsRef.current = currentPrompts;
    
    if (onUpdateConfig) {
      onUpdateConfig({ speakerTtsPrompts: currentPrompts });
    }
  };

  const handleSegmentClick = (segment: Segment) => {
    setSelectedSegmentId(segment.id);
    if (videoRef.current) {
        videoRef.current.currentTime = segment.startTime;
    }
  };

  const handlePlayAudio = (segment: Segment) => {
    if (playingSegmentId) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
      if (videoRef.current) {
        videoRef.current.muted = false;
        videoRef.current.pause();
      }
      
      if (playingSegmentId === segment.id) {
        setPlayingSegmentId(null);
        return;
      }
    }

    if (segment.audioUrl) {
      playSyncedAudio(segment, segment.audioUrl);
    }
  };

  const handleGeneratePreview = async (segment: Segment, forceRegenerate: boolean = false) => {
    handleUpdateSegment(segment.id, { isSynthesizing: true });
    
    try {
      const result = await api.previewSegment(projectId, segment.id, forceRegenerate);
      
      if (result.status === 'completed' && result.audioUrl) {
        handleUpdateSegment(segment.id, { 
          isSynthesizing: false,
          audioUrl: result.audioUrl 
        });
      } else if (result.status === 'processing' && result.jobId) {
        // Poll for completion
        pollPreviewStatus(segment.id, result.jobId);
      } else {
        // Fallback: check status after delay
        setTimeout(() => pollPreviewStatus(segment.id), 1000);
      }
    } catch (err) {
      console.error('Failed to generate preview:', err);
      handleUpdateSegment(segment.id, { isSynthesizing: false });
    }
  };

  const pollPreviewStatus = async (segmentId: string, jobId?: string) => {
    const maxAttempts = 60;  // 60 seconds max
    let attempts = 0;
    
    const poll = async () => {
      try {
        const status = await api.getPreviewStatus(projectId, segmentId, jobId);
        
        if (status.status === 'completed' && status.audioUrl) {
          handleUpdateSegment(segmentId, { 
            isSynthesizing: false,
            audioUrl: status.audioUrl 
          });
          return;
        } else if (status.status === 'failed') {
          console.error('Preview generation failed:', status.error);
          handleUpdateSegment(segmentId, { isSynthesizing: false });
          return;
        } else if (status.status === 'processing' || status.status === 'none') {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 1000);
          } else {
            console.error('Preview generation timeout');
            handleUpdateSegment(segmentId, { isSynthesizing: false });
          }
        } else {
          // Unknown status - stop synthesizing
          handleUpdateSegment(segmentId, { isSynthesizing: false });
        }
      } catch (err) {
        console.error('Failed to poll preview status:', err);
        handleUpdateSegment(segmentId, { isSynthesizing: false });
      }
    };
    
    setTimeout(poll, 500);
  };

  const playSyncedAudio = (segment: Segment, audioUrl: string) => {
    setPlayingSegmentId(segment.id);

    // Sync Video
    if (videoRef.current) {
        videoRef.current.currentTime = segment.startTime;
        videoRef.current.muted = true;
        videoRef.current.play().catch(e => console.log('Video play interrupted', e));
    }

    // Play audio from API with voice_id for correct cached file
    const audioSrc = api.getPreviewAudioUrl(projectId, segment.id, segment.voiceId);
    if (audioRef.current) {
      audioRef.current.src = audioSrc;
      audioRef.current.onended = () => {
        setPlayingSegmentId(null);
        if (videoRef.current) {
          videoRef.current.pause();
          videoRef.current.muted = false;
        }
      };
      audioRef.current.play().catch(e => console.log('Audio play failed', e));
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} className="hidden" />

      {/* Editor Content */}
      <div 
        ref={containerRef} 
        className={`flex-1 flex overflow-hidden ${isDragging ? 'select-none' : ''}`}
      >
        {/* Left Column: Segments List */}
        <div 
            style={{ width: `${sidebarWidth}%` }}
            className="border-r border-zinc-800 overflow-y-auto bg-zinc-950/50 scroll-smooth flex flex-col"
        >
            <div className="p-4 space-y-4 pb-20">
                {segments.map((segment) => (
                <div 
                    key={segment.id} 
                    onClick={() => handleSegmentClick(segment)}
                    className={`
                    relative bg-zinc-900 rounded-xl border transition-all duration-200 cursor-pointer group/card
                    ${selectedSegmentId === segment.id ? 'border-white/30' : playingSegmentId === segment.id ? 'border-brand-500/50' : 'border-zinc-800/50 hover:border-zinc-700/70'}
                    ${segment.isMuted ? 'opacity-60' : ''}
                    `}
                    style={{
                      boxShadow: selectedSegmentId === segment.id
                        ? `0 0 15px ${getGlowColor(segment.speakerColor).replace('0.6)', '0.9)')}, 0 0 30px ${getGlowColor(segment.speakerColor).replace('0.6)', '0.5)')}, 0 0 45px ${getGlowColor(segment.speakerColor).replace('0.6)', '0.25)')}`
                        : `0 0 8px ${getGlowColor(segment.speakerColor)}, 0 0 16px ${getGlowColor(segment.speakerColor).replace('0.6)', '0.25)')}`,
                    }}
                >
                    {/* Header Row: Speaker, Time, Mute */}
                    <div className="flex items-center justify-between p-3 border-b border-zinc-800/50 bg-zinc-950/30 rounded-t-xl">
                    <div className="flex items-center gap-3">
                        
                        {/* Speaker Control Area */}
                        <div className="flex items-center gap-1.5 relative">
                            {editingSpeakerInstance?.segmentId === segment.id ? (
                                <div className="flex items-center gap-1">
                                    <input
                                        type="text"
                                        value={renameValue}
                                        onClick={(e) => e.stopPropagation()}
                                        onChange={(e) => setRenameValue(e.target.value)}
                                        onBlur={handleCommitRename}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') handleCommitRename();
                                            if (e.key === 'Escape') setEditingSpeakerInstance(null);
                                        }}
                                        autoFocus
                                        className="px-2 py-1 rounded text-xs font-mono font-medium bg-zinc-800 text-white border border-brand-500 focus:outline-none w-32 shadow-lg"
                                    />
                                    <button 
                                        onMouseDown={(e) => { e.preventDefault(); handleCommitRename(); }}
                                        className="p-1 bg-brand-600/20 text-brand-400 rounded hover:bg-brand-600/40"
                                    >
                                        <Check className="w-3 h-3" />
                                    </button>
                                </div>
                            ) : (
                                <>
                                    <button 
                                        onClick={(e) => { 
                                            e.stopPropagation(); 
                                            setShowSpeakerDropdownId(showSpeakerDropdownId === segment.id ? null : segment.id); 
                                        }}
                                        className={`
                                        px-2 py-1 rounded text-xs font-mono font-medium cursor-pointer hover:ring-1 hover:ring-white/20 transition-all select-none border border-transparent
                                        flex items-center gap-1
                                        ${segment.speakerColor || 'bg-zinc-700 text-zinc-300'}
                                        `}
                                        title="Click to reassign speaker"
                                    >
                                        {segment.speaker}
                                        <ChevronRight className={`w-3 h-3 transition-transform ${showSpeakerDropdownId === segment.id ? 'rotate-90' : ''}`} />
                                    </button>

                                    <button
                                        onClick={(e) => { e.stopPropagation(); handleStartRename(segment.id, segment.speaker); }}
                                        className="p-1 text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 rounded opacity-0 group-hover/card:opacity-100 transition-opacity"
                                        title="Rename this speaker globally"
                                    >
                                        <Pencil className="w-3 h-3" />
                                    </button>

                                    {/* Reassign Dropdown */}
                                    {showSpeakerDropdownId === segment.id && (
                                        <div className="absolute top-full left-0 mt-1 z-50 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl py-1 min-w-[200px] animate-in fade-in zoom-in-95 duration-150">
                                            <div className="px-3 py-2 border-b border-zinc-800">
                                                <div className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider mb-2">Speaker Style</div>
                                                <input 
                                                    type="text"
                                                    placeholder="Global style for this speaker..."
                                                    value={activeProject?.config.speakerTtsPrompts?.[segment.speaker] || ''}
                                                    onClick={(e) => e.stopPropagation()}
                                                    onChange={(e) => handleSpeakerPromptChange(segment.speaker, e.target.value)}
                                                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                                                />
                                                <p className="text-[9px] text-zinc-500 mt-1">Applies to all segments of {segment.speaker}</p>
                                            </div>
                                            <div className="px-3 py-1.5 text-[10px] uppercase font-bold text-zinc-500 tracking-wider">Switch Speaker</div>
                                            <div className="max-h-48 overflow-y-auto">
                                                {uniqueSpeakers.map((spk) => (
                                                    <button
                                                        key={spk.name}
                                                        onClick={(e) => { e.stopPropagation(); handleSegmentReassign(segment.id, spk.name); }}
                                                        className={`w-full text-left px-3 py-1.5 text-xs hover:bg-zinc-800 transition-colors flex items-center gap-2 ${spk.name === segment.speaker ? 'bg-brand-500/10 text-brand-400' : 'text-zinc-300'}`}
                                                    >
                                                        <span className={`w-2 h-2 rounded-full ${spk.color.split(' ')[0].replace('/20', '')}`}></span>
                                                        {spk.name}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                        
                        <div className="flex items-center gap-1.5 text-xs text-zinc-400 font-mono bg-zinc-950 rounded px-2 py-1 border border-zinc-800">
                        <Clock className="w-3 h-3 text-zinc-600 shrink-0" />
                        <input 
                            type="number" 
                            value={segment.startTime}
                            step={0.1}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => handleUpdateSegment(segment.id, { startTime: parseFloat(e.target.value) })}
                            className="bg-transparent w-14 text-center focus:outline-none focus:text-white [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        />
                        <span className="text-zinc-600">–</span>
                        <input 
                            type="number" 
                            value={segment.endTime}
                            step={0.1}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => handleUpdateSegment(segment.id, { endTime: parseFloat(e.target.value) })}
                            className="bg-transparent w-14 text-center focus:outline-none focus:text-white [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        />
                        </div>
                    </div>

                    <button 
                        onClick={(e) => { e.stopPropagation(); handleUpdateSegment(segment.id, { isMuted: !segment.isMuted }); }}
                        className={`p-1.5 rounded-md transition-colors ${
                        segment.isMuted 
                            ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20' 
                            : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
                        }`}
                        title={segment.isMuted ? "Unmute segment" : "Mute segment"}
                    >
                        {segment.isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                    </button>
                    </div>

                    <div className="p-3 grid gap-3">
                    {/* Text Comparison */}
                    <div className="grid grid-cols-1 gap-2">
                        <div className="text-sm text-zinc-400 leading-snug">
                            {segment.originalText}
                        </div>

                        <div className="relative group/textarea">
                            <textarea 
                                value={segment.translatedText}
                                onClick={(e) => e.stopPropagation()}
                                onChange={(e) => handleUpdateSegment(segment.id, { 
                                    translatedText: e.target.value,
                                    audioUrl: undefined
                                })}
                                disabled={segment.isLocked}
                                className="w-full min-h-[80px] p-2.5 pr-8 bg-zinc-950 rounded-lg border border-zinc-800 text-zinc-100 text-sm leading-relaxed focus:outline-none focus:ring-1 focus:ring-brand-500/50 focus:border-brand-500/50 resize-y"
                            />
                            {segment.isLocked && <div className="absolute inset-0 bg-zinc-950/80 flex items-center justify-center rounded-lg"><Loader2 className="w-5 h-5 text-brand-500 animate-spin" /></div>}
                            
                            {/* Re-translation Tool */}
                            <div className="absolute top-2 right-2 z-10">
                                {showPromptInputId === segment.id ? (
                                    <div className="absolute right-0 top-0 z-20 flex items-center gap-1 bg-zinc-900 border border-zinc-700 p-1.5 rounded-lg shadow-xl animate-in fade-in zoom-in-95 duration-200">
                                    <input 
                                        type="text" 
                                        value={reTranslatePrompt}
                                        onClick={(e) => e.stopPropagation()}
                                        onChange={(e) => setReTranslatePrompt(e.target.value)}
                                        placeholder="Refinement..."
                                        className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1 text-xs text-white w-40 focus:outline-none focus:border-brand-500"
                                        autoFocus
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') {
                                                e.stopPropagation();
                                                handleReTranslate(segment.id);
                                            }
                                        }}
                                    />
                                    <button 
                                        onClick={(e) => { e.stopPropagation(); handleReTranslate(segment.id); }}
                                        className="p-1 bg-brand-600 hover:bg-brand-500 text-white rounded-md"
                                    >
                                        <Wand2 className="w-3 h-3" />
                                    </button>
                                    <button 
                                        onClick={(e) => { e.stopPropagation(); setShowPromptInputId(null); setReTranslatePrompt(''); }}
                                        className="p-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 rounded-md"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                    </div>
                                ) : (
                                    <button 
                                    onClick={(e) => { e.stopPropagation(); setShowPromptInputId(segment.id); }}
                                    className="p-1.5 text-zinc-500 hover:text-brand-400 hover:bg-brand-500/10 rounded-md transition-colors opacity-0 group-hover/textarea:opacity-100 focus:opacity-100"
                                    title="AI Re-phrase"
                                    >
                                    <Wand2 className="w-3.5 h-3.5" />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Controls Footer */}
                    <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-zinc-800/50">
                        {/* Voice Selection */}
                        <div className="flex-1 min-w-[120px]">
                        <select 
                            value={segment.voiceId}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => handleGlobalVoiceChange(segment.speaker, e.target.value)}
                            className="w-full bg-zinc-950 border border-zinc-800 text-zinc-300 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-zinc-700"
                        >
                            {voices.map(voice => (
                            <option key={voice.id} value={voice.id}>
                                {voice.name} ({voice.provider})
                            </option>
                            ))}
                        </select>
                        </div>

                        {/* TTS Prompt / Style */}
                        <div className="relative">
                        {showTtsPromptId === segment.id ? (
                            <div className="flex items-center gap-2 animate-in fade-in zoom-in-95 duration-200 absolute right-0 bottom-full mb-2 bg-zinc-900 border border-zinc-700 p-1.5 rounded-lg shadow-xl z-30">
                            <input 
                                type="text" 
                                value={segment.ttsPrompt || ''}
                                onClick={(e) => e.stopPropagation()}
                                onChange={(e) => handleUpdateSegment(segment.id, { 
                                ttsPrompt: e.target.value,
                                audioUrl: undefined
                                })}
                                placeholder="e.g. Happy, Fast"
                                className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1 text-xs text-white w-32 focus:outline-none focus:ring-1 focus:ring-brand-500"
                                autoFocus
                            />
                            <button 
                                onClick={(e) => { e.stopPropagation(); setShowTtsPromptId(null); }}
                                className="p-1 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md"
                            >
                                <X className="w-3 h-3" />
                            </button>
                            </div>
                        ) : (
                            <button 
                            onClick={(e) => { e.stopPropagation(); setShowTtsPromptId(segment.id); }}
                            className={`p-1.5 rounded-md transition-colors border ${
                                segment.ttsPrompt 
                                ? 'bg-brand-500/10 text-brand-400 border-brand-500/30' 
                                : 'text-zinc-500 border-transparent hover:bg-zinc-800 hover:text-zinc-300'
                            }`}
                            title="Set TTS Style"
                            >
                            <Sparkles className="w-3.5 h-3.5" />
                            </button>
                        )}
                        </div>

                        <div className="h-4 w-px bg-zinc-800 mx-1"></div>

                        {/* Audio Controls */}
                        <div className="flex items-center gap-1">
                        {segment.audioUrl ? (
                            <>
                            {/* Play Button */}
                            <button
                                onClick={(e) => { e.stopPropagation(); handlePlayAudio(segment); }}
                                disabled={segment.isLocked}
                                className={`
                                    flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors
                                    ${playingSegmentId === segment.id 
                                        ? 'bg-brand-500/20 text-brand-300 border border-brand-500/30'
                                        : 'bg-emerald-600/10 text-emerald-400 hover:bg-emerald-600/20 border border-emerald-500/20'}
                                `}
                            >
                                {playingSegmentId === segment.id ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                                {playingSegmentId === segment.id ? 'Stop' : 'Play'}
                            </button>

                            {/* Regenerate Button */}
                            <button
                                onClick={(e) => { e.stopPropagation(); handleGeneratePreview(segment, true); }}
                                disabled={segment.isSynthesizing || segment.isLocked}
                                className="p-1.5 rounded-md bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white border border-zinc-700 transition-colors disabled:opacity-50"
                                title="Regenerate Audio"
                            >
                                {segment.isSynthesizing ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                    <RefreshCw className="w-3 h-3" />
                                )}
                            </button>
                            </>
                        ) : (
                            /* Generate Button - always force regenerate to handle prompt changes */
                            <button
                                onClick={(e) => { e.stopPropagation(); handleGeneratePreview(segment, true); }}
                                disabled={segment.isSynthesizing || segment.isLocked}
                                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors bg-indigo-600/10 text-indigo-400 hover:bg-indigo-600/20 border border-indigo-500/20 disabled:opacity-50"
                            >
                                {segment.isSynthesizing ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                    <>
                                        <Speaker className="w-3 h-3" />
                                        Gen
                                    </>
                                )}
                            </button>
                        )}
                        </div>
                    </div>
                    </div>
                </div>
                ))}
            </div>
        </div>

        {/* Resizer Handle */}
        <div
            className="w-1.5 bg-zinc-900 border-x border-zinc-800 hover:border-brand-500/50 hover:bg-brand-500/10 cursor-col-resize transition-colors z-20 flex items-center justify-center group/resizer"
            onMouseDown={() => setIsDragging(true)}
        >
            <div className="h-8 w-0.5 bg-zinc-700 group-hover/resizer:bg-brand-500 rounded-full transition-colors"></div>
        </div>

        {/* Right Column: Video Player */}
        <div 
            style={{ width: `${100 - sidebarWidth}%` }}
            className={`bg-black flex items-center justify-center p-8 relative overflow-hidden ${isDragging ? 'pointer-events-none' : ''}`}
        >
            {videoUrl ? (
                <div className="relative w-full h-full flex items-center justify-center">
                    <video 
                        ref={videoRef}
                        src={videoUrl}
                        controls
                        className="max-h-full max-w-full rounded-lg shadow-2xl outline-none"
                    />
                </div>
            ) : (
                <div className="text-zinc-600 flex flex-col items-center gap-3">
                    <div className="w-16 h-16 rounded-full bg-zinc-900 flex items-center justify-center">
                        <Play className="w-6 h-6 text-zinc-700" />
                    </div>
                    <p>No video loaded for this project</p>
                </div>
            )}
        </div>
      </div>
      
      {/* Bottom Action Bar */}
      <div className="shrink-0 p-4 border-t border-zinc-800 bg-zinc-900 flex justify-end items-center z-10 gap-4">
        <div className="text-sm text-zinc-400 mr-auto">
            {segments.filter(s => s.audioUrl).length} / {segments.length} segments dubbed
        </div>
        <button 
          onClick={onContinue}
          className="group flex items-center gap-2 px-6 py-2.5 bg-brand-600 hover:bg-brand-500 rounded-lg text-white font-semibold transition-all shadow-lg shadow-brand-500/20"
        >
          Start Final Dubbing
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </button>
      </div>
    </div>
  );
};
