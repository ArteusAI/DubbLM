
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { UploadView } from './components/UploadView';
import { ProcessingView } from './components/ProcessingView';
import { EditorView } from './components/EditorView';
import { ResultView } from './components/ResultView';
import { ProjectListView } from './components/ProjectListView';
import { SettingsModal } from './components/SettingsModal';
import { AppStep, AppConfig, Segment, ProcessingLog, Persona, Project, ProjectStatus, PresetId, LlmProvider } from './types';
import { DEFAULT_PERSONAS, PRESETS } from './constants';
import { ChevronRight, LayoutDashboard, Settings, AlertTriangle, Loader2, X } from 'lucide-react';
import api, { ProjectResponse, SegmentResponse, VoiceResponse, PersonaResponse } from './api';

const DEFAULT_PRESET = PRESETS.find(p => p.id === 'hq')!;

const INITIAL_CONFIG: AppConfig = {
  sourceLang: 'auto',
  targetLang: 'ru',
  defaultTargetLang: 'ru',
  personaId: 'normal',
  keepBackground: DEFAULT_PRESET.keepBackground ?? false,
  preset: 'hq',
  apiKeys: {},
  // Apply default preset settings
  llmProvider: DEFAULT_PRESET.llmProvider,
  llmModelName: DEFAULT_PRESET.llmModelName,
  llmTemperature: DEFAULT_PRESET.llmTemperature,
  refinementLlmProvider: DEFAULT_PRESET.refinementLlmProvider,
  refinementModelName: DEFAULT_PRESET.refinementModelName,
  refinementTemperature: DEFAULT_PRESET.refinementTemperature,
  ttsSystem: DEFAULT_PRESET.ttsSystem,
  ttsModel: DEFAULT_PRESET.ttsModel,
  ttsPromptPrefix: DEFAULT_PRESET.ttsPromptPrefix,
  voiceAutoSelection: DEFAULT_PRESET.voiceAutoSelection,
  enableEmotionEnrichment: DEFAULT_PRESET.enableEmotionEnrichment,
  dubbedVolume: DEFAULT_PRESET.dubbedVolume,
  backgroundVolume: DEFAULT_PRESET.backgroundVolume,
  useTwoPassEncoding: DEFAULT_PRESET.useTwoPassEncoding,
  videoQualityPreset: DEFAULT_PRESET.videoQualityPreset,
  maxWorkers: DEFAULT_PRESET.maxWorkers,
  pauseRemoval: DEFAULT_PRESET.pauseRemoval,
  translationPromptPrefix: '',
  speakerTtsPrompts: {},
  speakerVoiceMappings: {},
};

const mapSegmentFromApi = (
  seg: SegmentResponse,
  fallback?: { speakerVoiceMappings?: Record<string, string>; ttsSystem?: string }
): Segment => ({
  id: seg.id,
  projectId: seg.projectId,
  speaker: seg.speaker,
  startTime: seg.startTime,
  endTime: seg.endTime,
  originalText: seg.originalText,
  translatedText: seg.translatedText || '',
  isMuted: seg.isMuted,
  voiceId: seg.voiceId || fallback?.speakerVoiceMappings?.[seg.speaker] || 'alloy',
  provider: seg.provider || fallback?.ttsSystem || 'openai',
  audioUrl: seg.audioUrl,
  ttsPrompt: seg.ttsPrompt,
  speakerColor: seg.speakerColor,
});

const mapProjectFromApi = (p: ProjectResponse): Project => {
  const presetId = (p.config.preset as PresetId) || 'hq';
  const preset = PRESETS.find(pr => pr.id === presetId) || DEFAULT_PRESET;
  const cfg = p.config;
  
  return {
    id: p.id,
    name: p.name,
    createdAt: new Date(p.createdAt),
    updatedAt: new Date(p.updatedAt),
    status: p.status as ProjectStatus,
    config: {
      sourceLang: cfg.sourceLang || 'auto',
      targetLang: cfg.targetLang || 'ru',
      personaId: cfg.personaId || 'normal',
      speakerCount: cfg.speakerCount,
      keepBackground: cfg.keepBackground ?? (preset.keepBackground ?? false),
      preset: presetId,
      apiKeys: {},
      // Load saved settings, fallback to preset defaults
      startTime: cfg.startTime,
      duration: cfg.duration,
      transcriptionSystem: cfg.transcriptionSystem as 'assemblyai' | 'openai' | 'whisperx' | undefined,
      whisperModel: cfg.whisperModel,
      llmProvider: (cfg.llmProvider as LlmProvider) || preset.llmProvider,
      llmModelName: cfg.llmModelName || preset.llmModelName,
      llmTemperature: cfg.llmTemperature ?? preset.llmTemperature,
      refinementLlmProvider: (cfg.refinementLlmProvider as LlmProvider) || preset.refinementLlmProvider,
      refinementModelName: cfg.refinementModelName || preset.refinementModelName,
      refinementTemperature: cfg.refinementTemperature ?? preset.refinementTemperature,
      translationPromptPrefix: cfg.translationPromptPrefix,
      speakerTtsPrompts: cfg.speakerTtsPrompts || {},
      speakerVoiceMappings: cfg.speakerVoiceMappings || {},
      ttsSystem: cfg.ttsSystem || preset.ttsSystem,
      ttsModel: cfg.ttsModel || preset.ttsModel,
      ttsPromptPrefix: cfg.ttsPromptPrefix ?? preset.ttsPromptPrefix,
      voiceAutoSelection: cfg.voiceAutoSelection ?? preset.voiceAutoSelection,
      enableEmotionEnrichment: cfg.enableEmotionEnrichment ?? preset.enableEmotionEnrichment,
      dubbedVolume: cfg.dubbedVolume ?? preset.dubbedVolume,
      backgroundVolume: cfg.backgroundVolume ?? preset.backgroundVolume,
      useTwoPassEncoding: cfg.useTwoPassEncoding ?? preset.useTwoPassEncoding,
      videoQualityPreset: cfg.videoQualityPreset ?? preset.videoQualityPreset,
      maxWorkers: cfg.maxWorkers ?? preset.maxWorkers,
      pauseRemoval: cfg.pauseRemoval || preset.pauseRemoval,
      postDiarizationMergeGap: cfg.postDiarizationMergeGap,
      postTranslationMergeGap: cfg.postTranslationMergeGap,
      maxSegmentDuration: cfg.maxSegmentDuration,
      minSegmentDuration: cfg.minSegmentDuration,
      comfortMinAdjustmentRatio: cfg.comfortMinAdjustmentRatio,
      comfortMaxAdjustmentRatio: cfg.comfortMaxAdjustmentRatio,
      segmentStretch: cfg.segmentStretch,
      minPauseDuration: cfg.minPauseDuration,
      preservePauseDuration: cfg.preservePauseDuration,
    },
    segments: p.segments?.map(seg => mapSegmentFromApi(seg, {
      speakerVoiceMappings: cfg.speakerVoiceMappings || {},
      ttsSystem: cfg.ttsSystem || preset.ttsSystem,
    })) || [],
    videoFile: null,
    sourceFilename: p.sourceFilename,
    sourceSize: p.sourceSize,
    processProgress: 0,
  };
};

// URL path helpers for browser history
const buildPath = (step: AppStep, projectId: string | null): string => {
  if (step === AppStep.PROJECTS || !projectId) return '/';
  const stepPaths: Record<AppStep, string> = {
    [AppStep.PROJECTS]: '/',
    [AppStep.UPLOAD]: 'upload',
    [AppStep.PROCESSING_TRANSCRIPTION]: 'processing',
    [AppStep.EDITOR]: 'editor',
    [AppStep.PROCESSING_DUBBING]: 'dubbing',
    [AppStep.RESULT]: 'result',
  };
  return `/project/${projectId}/${stepPaths[step]}`;
};

const parsePath = (path: string): { step: AppStep; projectId: string | null } => {
  const match = path.match(/^\/project\/([^/]+)\/(.+)$/);
  if (!match) return { step: AppStep.PROJECTS, projectId: null };
  
  const [, projectId, stepPath] = match;
  const pathToStep: Record<string, AppStep> = {
    'upload': AppStep.UPLOAD,
    'processing': AppStep.PROCESSING_TRANSCRIPTION,
    'editor': AppStep.EDITOR,
    'dubbing': AppStep.PROCESSING_DUBBING,
    'result': AppStep.RESULT,
  };
  return { step: pathToStep[stepPath] || AppStep.UPLOAD, projectId };
};

const App: React.FC = () => {
  // Global State
  const [globalConfig, setGlobalConfig] = useState<AppConfig>(INITIAL_CONFIG);
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Navigation State
  const [step, setStep] = useState<AppStep>(AppStep.PROJECTS);
  const isPopStateNavigation = useRef(false);
  const isInitialLoad = useRef(true);
  
  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [personas, setPersonas] = useState<Persona[]>(DEFAULT_PERSONAS);
  const [voices, setVoices] = useState<VoiceResponse[]>([]);

  // Stop Processing Dialog
  const [showStopDialog, setShowStopDialog] = useState(false);
  const [pendingNavigationStep, setPendingNavigationStep] = useState<AppStep | null>(null);
  const [isStopping, setIsStopping] = useState(false);

  // Delete Confirmation Dialog
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Reset Confirmation Dialog
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [pendingResetId, setPendingResetId] = useState<string | null>(null);
  const [isResetting, setIsResetting] = useState(false);

  // Processing State
  const [processingLogs, setProcessingLogs] = useState<ProcessingLog[]>([]);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState('');

  // Derived State
  const activeProject = projects.find(p => p.id === activeProjectId) || null;
  const pendingResetProject = pendingResetId
    ? projects.find((p) => p.id === pendingResetId) || null
    : null;

  // --- Browser History Sync ---
  
  // Handle browser back/forward
  useEffect(() => {
    const handlePopState = async () => {
      const { step: newStep, projectId } = parsePath(window.location.pathname);
      isPopStateNavigation.current = true;
      
      if (!projectId) {
        setActiveProjectId(null);
        setStep(AppStep.PROJECTS);
        return;
      }
      
      if (projectId !== activeProjectId) {
        try {
          const projectData = await api.getProject(projectId);
          const apiProject = mapProjectFromApi(projectData);
          setProjects(prev => prev.map(p => p.id === projectId ? { ...apiProject, videoFile: p.videoFile } : p));
          setActiveProjectId(projectId);
          setStep(newStep);
        } catch {
          setActiveProjectId(null);
          setStep(AppStep.PROJECTS);
          window.history.replaceState(null, '', '/');
        }
      } else {
        setStep(newStep);
      }
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [activeProjectId]);

  // Sync URL with navigation state
  useEffect(() => {
    if (isLoading) return;
    
    if (isPopStateNavigation.current) {
      isPopStateNavigation.current = false;
      return;
    }
    
    const currentPath = buildPath(step, activeProjectId);
    if (window.location.pathname !== currentPath) {
      if (isInitialLoad.current) {
        // First sync - use replaceState to not add history entry
        isInitialLoad.current = false;
        window.history.replaceState({ step, projectId: activeProjectId }, '', currentPath);
      } else {
        window.history.pushState({ step, projectId: activeProjectId }, '', currentPath);
      }
    } else if (isInitialLoad.current) {
      isInitialLoad.current = false;
    }
  }, [step, activeProjectId, isLoading]);

  // --- Initial Data Load ---
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setIsLoading(true);
        const [projectsData, voicesData, personasData, settingsData] = await Promise.all([
          api.listProjects(),
          api.getVoices(),
          api.getPersonas(),
          api.getSettings(),
        ]);
        
        setProjects(projectsData.map(p => mapProjectFromApi(p as ProjectResponse)));
        setVoices(voicesData);
        
        if (settingsData && settingsData.defaults) {
          setGlobalConfig(prev => ({ ...prev, ...(settingsData.defaults as Partial<AppConfig>) }));
        }
        
        if (personasData.length > 0) {
          setPersonas(personasData.map((p: PersonaResponse) => ({
            id: p.id,
            name: p.name,
            description: p.description || '',
            promptTemplate: p.promptTemplate,
          })));
        }
        
        // If URL had a project, load it after initial data
        const { projectId, step: urlStep } = parsePath(window.location.pathname);
        if (projectId) {
          const projectExists = projectsData.some(p => p.id === projectId);
          if (projectExists) {
            try {
              const projectData = await api.getProject(projectId);
              const apiProject = mapProjectFromApi(projectData);
              setProjects(prev => prev.map(p => p.id === projectId ? apiProject : p));
              setActiveProjectId(projectId);
              setStep(urlStep);
            } catch {
              // Project not found, go to projects list
              setActiveProjectId(null);
              setStep(AppStep.PROJECTS);
              window.history.replaceState(null, '', '/');
            }
          } else {
            // Project doesn't exist
            setActiveProjectId(null);
            setStep(AppStep.PROJECTS);
            window.history.replaceState(null, '', '/');
          }
        }
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // --- SSE Status Subscription ---
  // Track processing status separately to avoid re-subscribing on every project update
  const activeProjectStatus = activeProject?.status;
  
  useEffect(() => {
    if (!activeProjectId) return;
    
    // Only subscribe if in processing state
    if (!activeProjectStatus || !['transcribing', 'dubbing'].includes(activeProjectStatus)) return;

    const unsubscribe = api.subscribeToStatus(activeProjectId, {
      onProgress: (percent, currentStep) => {
        setProcessingProgress(percent);
        setProcessingStep(currentStep);
        
        setProjects(prev => prev.map(p => 
          p.id === activeProjectId
            ? { ...p, processProgress: percent, processStage: currentStep }
            : p
        ));
      },
      onLog: (log) => {
        // Deduplicate logs by ID
        setProcessingLogs(prev => {
          if (prev.some(l => l.id === log.id)) return prev;
          return [...prev, {
            id: log.id,
            message: log.message,
            type: log.type,
            timestamp: new Date(log.timestamp),
            text: log.text,
          }];
        });
      },
      onComplete: async (status, errorMsg) => {
        if (errorMsg) {
          setError(errorMsg);
          setProjects(prev => prev.map(p =>
            p.id === activeProjectId
              ? { ...p, status: 'error' as ProjectStatus, error: errorMsg }
              : p
          ));
          setStep(AppStep.UPLOAD);
          return;
        }

        // Reload project to get updated data
        try {
          const updatedProject = await api.getProject(activeProjectId);
          const mappedProject = mapProjectFromApi(updatedProject);
          
          setProjects(prev => prev.map(p =>
            p.id === activeProjectId ? mappedProject : p
          ));

          // Navigate based on status
          if (status === 'transcribed') {
            setStep(AppStep.EDITOR);
          } else if (status === 'dubbed') {
            // Clear autoProcess flag when complete
            await api.updateProjectConfig(activeProjectId, { autoProcess: false });
            setStep(AppStep.RESULT);
          }
        } catch (err) {
          console.error('Failed to reload project:', err);
        }
      },
      onError: (err) => {
        console.error('SSE error:', err);
      },
    });

    return unsubscribe;
  }, [activeProjectId, activeProjectStatus]);

  // --- Project Management ---

  const handleBatchUpload = async (files: File[]) => {
    try {
      const newProjects: Project[] = [];
      
      for (const file of files) {
        const created = await api.createProject(file.name);
        const project = mapProjectFromApi(created);
        
        // Sync default config to backend immediately
        await api.updateProjectConfig(project.id, {
          sourceLang: project.config.sourceLang,
          targetLang: project.config.targetLang,
          personaId: project.config.personaId,
          preset: project.config.preset,
          keepBackground: project.config.keepBackground,
          pauseRemoval: project.config.pauseRemoval,
          videoQualityPreset: project.config.videoQualityPreset,
        });
        
        project.videoFile = file;
        project.isUploading = true;
        project.uploadProgress = 0;
        newProjects.push(project);
      }

      setProjects(prev => [...newProjects, ...prev]);

      // Start background uploads for all new projects
      for (const project of newProjects) {
        if (project.videoFile) {
          uploadVideoInBackground(project.id, project.videoFile);
        }
      }

      if (newProjects.length === 1) {
        setActiveProjectId(newProjects[0].id);
        setStep(AppStep.UPLOAD);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
    }
  };

  const uploadVideoInBackground = async (projectId: string, file: File) => {
    try {
      await api.uploadVideo(projectId, file, (percent) => {
        setProjects(prev => prev.map(p =>
          p.id === projectId
            ? { ...p, uploadProgress: percent }
            : p
        ));
      });
      
      // Upload complete
      setProjects(prev => prev.map(p =>
        p.id === projectId
          ? { ...p, isUploading: false, uploadProgress: 100, sourceFilename: file.name, sourceSize: file.size }
          : p
      ));
    } catch (err) {
      console.error(`Failed to upload video for project ${projectId}:`, err);
      setProjects(prev => prev.map(p =>
        p.id === projectId
          ? { ...p, isUploading: false, uploadProgress: 0, error: err instanceof Error ? err.message : 'Upload failed' }
          : p
      ));
    }
  };

  const handleAutoProcess = async (ids: string[]) => {
    for (const id of ids) {
      const project = projects.find(p => p.id === id);
      if (!project) continue;

      // Skip if still uploading
      if (project.isUploading) {
        console.log(`Project ${id} is still uploading, skipping auto-process`);
        continue;
      }

      try {
        // Update full config with autoProcess flag
        const cfg = project.config;
        await api.updateProjectConfig(id, {
          sourceLang: cfg.sourceLang,
          targetLang: cfg.targetLang,
          personaId: cfg.personaId,
          keepBackground: cfg.keepBackground,
          pauseRemoval: cfg.pauseRemoval,
          preset: cfg.preset,
          llmProvider: cfg.llmProvider,
          llmModelName: cfg.llmModelName,
          llmTemperature: cfg.llmTemperature,
          speakerTtsPrompts: cfg.speakerTtsPrompts || {},
          speakerVoiceMappings: cfg.speakerVoiceMappings || {},
          refinementLlmProvider: cfg.refinementLlmProvider,
          refinementModelName: cfg.refinementModelName,
          refinementTemperature: cfg.refinementTemperature,
          ttsSystem: cfg.ttsSystem,
          ttsModel: cfg.ttsModel,
          ttsPromptPrefix: cfg.ttsPromptPrefix,
          voiceAutoSelection: cfg.voiceAutoSelection,
          enableEmotionEnrichment: cfg.enableEmotionEnrichment,
          dubbedVolume: cfg.dubbedVolume,
          backgroundVolume: cfg.backgroundVolume,
          useTwoPassEncoding: cfg.useTwoPassEncoding,
          videoQualityPreset: cfg.videoQualityPreset,
          maxWorkers: cfg.maxWorkers,
          autoProcess: true,  // Backend will auto-start dubbing after transcription
        });

        // Start transcription
        await api.startTranscription(id);
        
        setProjects(prev => prev.map(p =>
          p.id === id
            ? { ...p, status: 'transcribing' as ProjectStatus, isAutoProcessing: true, processProgress: 0, processStage: 'Starting...' }
            : p
        ));
      } catch (err) {
        console.error(`Failed to start processing for ${id}:`, err);
        setProjects(prev => prev.map(p =>
          p.id === id
            ? { ...p, status: 'error' as ProjectStatus, error: err instanceof Error ? err.message : 'Processing failed' }
            : p
        ));
      }
    }
  };

  const handleStopProcess = async (ids: string[]) => {
    for (const id of ids) {
      const project = projects.find(p => p.id === id);
      if (!project) continue;

      // Only stop if actually processing
      if (project.status === 'transcribing' || project.status === 'dubbing' || project.isAutoProcessing) {
        try {
          // Call API to stop processing
          await api.stopProcessing(id);
          
          // Update local state
          setProjects(prev => prev.map(p => {
            if (p.id === id) {
              const newStatus: ProjectStatus = p.status === 'dubbing' ? 'transcribed' : 'draft';
              return {
                ...p,
                status: newStatus,
                isAutoProcessing: false,
                processProgress: 0,
                processStage: undefined,
              };
            }
            return p;
          }));
        } catch (err) {
          console.error(`Failed to stop processing for ${id}:`, err);
          setError(err instanceof Error ? err.message : 'Failed to stop processing');
        }
      }
    }
  };

  const handleResetAndRestart = (id: string) => {
    const project = projects.find(p => p.id === id);
    if (!project) return;

    if (project.isUploading) {
      setError('Please wait for video upload to complete');
      return;
    }

    setPendingResetId(id);
    setShowResetDialog(true);
  };

  const handleConfirmResetAndRestart = async () => {
    if (!pendingResetId) return;
    const id = pendingResetId;
    const project = projects.find(p => p.id === id);
    if (!project) {
      setShowResetDialog(false);
      setPendingResetId(null);
      return;
    }

    setIsResetting(true);
    try {
      setProjects(prev => prev.map(p =>
        p.id === id
          ? {
              ...p,
              status: 'transcribing' as ProjectStatus,
              segments: [],
              processProgress: 0,
              processStage: 'Resetting cache and starting...',
              isAutoProcessing: true,
              error: undefined,
            }
          : p
      ));

      const job = await api.restartProcessing(id);

      setProjects(prev => prev.map(p =>
        p.id === id
          ? {
              ...p,
              status: 'transcribing' as ProjectStatus,
              segments: [],
              processProgress: 0,
              processStage: job.currentStep || 'pending',
              currentJobId: job.jobId,
              isAutoProcessing: true,
              error: undefined,
            }
          : p
      ));
      setShowResetDialog(false);
      setPendingResetId(null);
    } catch (err) {
      setProjects(prev => prev.map(p => (p.id === id ? project : p)));
      console.error(`Failed to reset and restart project ${id}:`, err);
      setError(err instanceof Error ? err.message : 'Failed to reset and restart project');
    } finally {
      setIsResetting(false);
    }
  };

  const handleCancelResetDialog = () => {
    if (isResetting) return;
    setShowResetDialog(false);
    setPendingResetId(null);
  };

  const handleSelectProject = async (id: string) => {
    try {
      const projectData = await api.getProject(id);
      const apiProject = mapProjectFromApi(projectData);
      
      // Preserve local state (videoFile, upload progress) when updating from API
      setProjects(prev => prev.map(p => {
        if (p.id === id) {
          return {
            ...apiProject,
            videoFile: p.videoFile,
            isUploading: p.isUploading,
            uploadProgress: p.uploadProgress,
          };
        }
        return p;
      }));
      setActiveProjectId(id);
      
      if (apiProject.status === 'dubbed') {
        setStep(AppStep.RESULT);
      } else if (apiProject.status === 'transcribed') {
        setStep(AppStep.EDITOR);
      } else if (['transcribing', 'dubbing'].includes(apiProject.status)) {
        setStep(apiProject.status === 'transcribing' ? AppStep.PROCESSING_TRANSCRIPTION : AppStep.PROCESSING_DUBBING);
      } else {
        setStep(AppStep.UPLOAD);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load project');
    }
  };

  const handleDeleteProject = async (id: string) => {
    const project = projects.find(p => p.id === id);
    
    // Check if project is processing
    if (project && (project.status === 'transcribing' || project.status === 'dubbing' || project.isAutoProcessing)) {
      setPendingDeleteId(id);
      setShowDeleteDialog(true);
      return;
    }

    // Direct delete if not processing
    await performDelete(id);
  };

  const performDelete = async (id: string) => {
    try {
      await api.deleteProject(id);
      setProjects(prev => prev.filter(p => p.id !== id));
      
      if (activeProjectId === id) {
        setActiveProjectId(null);
        setStep(AppStep.PROJECTS);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete project');
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDeleteId) return;

    setIsDeleting(true);
    try {
      // Stop processing first
      await api.stopProcessing(pendingDeleteId);
      
      // Then delete
      await performDelete(pendingDeleteId);
      
      setShowDeleteDialog(false);
      setPendingDeleteId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete project');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCancelDeleteDialog = () => {
    setShowDeleteDialog(false);
    setPendingDeleteId(null);
  };

  const handleChangePreset = async (projectId: string, presetId: PresetId) => {
    try {
      const preset = PRESETS.find(p => p.id === presetId) || DEFAULT_PRESET;
      const presetFromApi = await api.getPreset(presetId).catch(() => null);
      const configUpdate = {
        preset: presetId,
        personaId: presetId === 'fast' ? 'none' : 'normal',
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
      
      await api.updateProjectConfig(projectId, configUpdate);
      setProjects(prev => prev.map(p =>
        p.id === projectId
          ? { ...p, config: { ...p.config, ...configUpdate } }
          : p
      ));
    } catch (err) {
      console.error('Failed to update preset:', err);
      setError(err instanceof Error ? err.message : 'Failed to update preset');
    }
  };

  const handleChangeLanguages = async (projectId: string, sourceLang: string, targetLang: string) => {
    try {
      const configUpdate = { sourceLang, targetLang };
      await api.updateProjectConfig(projectId, configUpdate);
      setProjects(prev => prev.map(p =>
        p.id === projectId
          ? { ...p, config: { ...p.config, ...configUpdate } }
          : p
      ));
    } catch (err) {
      console.error('Failed to update languages:', err);
      setError(err instanceof Error ? err.message : 'Failed to update languages');
    }
  };

  const updateActiveProject = useCallback((updates: Partial<Project>) => {
    if (!activeProjectId) return;
    setProjects(prev => prev.map(p => 
      p.id === activeProjectId 
        ? { ...p, ...updates, updatedAt: new Date() } 
        : p
    ));
    
    // Save config to API if it was updated
    if (updates.config) {
      const cfg = updates.config;
      api.updateProjectConfig(activeProjectId, {
        sourceLang: cfg.sourceLang,
        targetLang: cfg.targetLang,
        personaId: cfg.personaId,
        speakerCount: cfg.speakerCount,
        keepBackground: cfg.keepBackground,
        pauseRemoval: cfg.pauseRemoval,
        preset: cfg.preset,
        // Extra settings
        startTime: cfg.startTime,
        duration: cfg.duration,
        transcriptionSystem: cfg.transcriptionSystem,
        whisperModel: cfg.whisperModel,
        llmProvider: cfg.llmProvider,
        llmModelName: cfg.llmModelName,
        llmTemperature: cfg.llmTemperature,
        speakerTtsPrompts: cfg.speakerTtsPrompts || {},
        speakerVoiceMappings: cfg.speakerVoiceMappings || {},
        refinementLlmProvider: cfg.refinementLlmProvider,
        refinementModelName: cfg.refinementModelName,
        refinementTemperature: cfg.refinementTemperature,
        translationPromptPrefix: cfg.translationPromptPrefix,
        ttsSystem: cfg.ttsSystem,
        ttsModel: cfg.ttsModel,
        ttsPromptPrefix: cfg.ttsPromptPrefix,
        voiceAutoSelection: cfg.voiceAutoSelection,
        enableEmotionEnrichment: cfg.enableEmotionEnrichment,
        dubbedVolume: cfg.dubbedVolume,
        backgroundVolume: cfg.backgroundVolume,
        useTwoPassEncoding: cfg.useTwoPassEncoding,
        videoQualityPreset: cfg.videoQualityPreset,
        maxWorkers: cfg.maxWorkers,
        postDiarizationMergeGap: cfg.postDiarizationMergeGap,
        postTranslationMergeGap: cfg.postTranslationMergeGap,
        maxSegmentDuration: cfg.maxSegmentDuration,
        minSegmentDuration: cfg.minSegmentDuration,
        comfortMinAdjustmentRatio: cfg.comfortMinAdjustmentRatio,
        comfortMaxAdjustmentRatio: cfg.comfortMaxAdjustmentRatio,
        segmentStretch: cfg.segmentStretch,
        minPauseDuration: cfg.minPauseDuration,
        preservePauseDuration: cfg.preservePauseDuration,
      }).catch(err => {
        console.error('Failed to save config:', err);
      });
    }
  }, [activeProjectId]);

  const updateGlobalConfig = (updates: Partial<AppConfig>) => {
    setGlobalConfig(prev => {
      const newConfig = { ...prev, ...updates };
      // Save to backend
      api.updateSettings({ defaults: newConfig }).catch(err => {
        console.error('Failed to save global settings:', err);
      });
      return newConfig;
    });
  };

  // --- Step Handlers ---

  const startProcessing = async () => {
    if (!activeProject) return;

    // Don't start if upload is still in progress
    if (activeProject.isUploading) {
      setError('Please wait for video upload to complete');
      return;
    }

    try {
      setProcessingLogs([]);
      setProcessingProgress(0);
      setProcessingStep('Preparing...');
      setStep(AppStep.PROCESSING_TRANSCRIPTION);

      // Save full config (and auto-process end-to-end).
      const cfg = activeProject.config;
      await api.updateProjectConfig(activeProject.id, {
        sourceLang: cfg.sourceLang,
        targetLang: cfg.targetLang,
        personaId: cfg.personaId,
        speakerCount: cfg.speakerCount,
        keepBackground: cfg.keepBackground,
        pauseRemoval: cfg.pauseRemoval,
        preset: cfg.preset,
        // Extra settings
        startTime: cfg.startTime,
        duration: cfg.duration,
        transcriptionSystem: cfg.transcriptionSystem,
        whisperModel: cfg.whisperModel,
        llmProvider: cfg.llmProvider,
        llmModelName: cfg.llmModelName,
        llmTemperature: cfg.llmTemperature,
        speakerTtsPrompts: cfg.speakerTtsPrompts || {},
        speakerVoiceMappings: cfg.speakerVoiceMappings || {},
        refinementLlmProvider: cfg.refinementLlmProvider,
        refinementModelName: cfg.refinementModelName,
        refinementTemperature: cfg.refinementTemperature,
        translationPromptPrefix: cfg.translationPromptPrefix,
        ttsSystem: cfg.ttsSystem,
        ttsModel: cfg.ttsModel,
        ttsPromptPrefix: cfg.ttsPromptPrefix,
        voiceAutoSelection: cfg.voiceAutoSelection,
        enableEmotionEnrichment: cfg.enableEmotionEnrichment,
        dubbedVolume: cfg.dubbedVolume,
        backgroundVolume: cfg.backgroundVolume,
        useTwoPassEncoding: cfg.useTwoPassEncoding,
        videoQualityPreset: cfg.videoQualityPreset,
        maxWorkers: cfg.maxWorkers,
        postDiarizationMergeGap: cfg.postDiarizationMergeGap,
        postTranslationMergeGap: cfg.postTranslationMergeGap,
        maxSegmentDuration: cfg.maxSegmentDuration,
        minSegmentDuration: cfg.minSegmentDuration,
        comfortMinAdjustmentRatio: cfg.comfortMinAdjustmentRatio,
        comfortMaxAdjustmentRatio: cfg.comfortMaxAdjustmentRatio,
        segmentStretch: cfg.segmentStretch,
        minPauseDuration: cfg.minPauseDuration,
        preservePauseDuration: cfg.preservePauseDuration,
        autoProcess: true,
      });

      // Start transcription
      await api.startTranscription(activeProject.id);
      
      updateActiveProject({ 
        status: 'transcribing',
        processProgress: 0,
        processStage: 'Transcribing...'
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start processing');
      setStep(AppStep.UPLOAD);
    }
  };

  const handleTranscriptionComplete = async () => {
    if (!activeProjectId) return;
    
    try {
      const projectData = await api.getProject(activeProjectId);
      const project = mapProjectFromApi(projectData);
      
      setProjects(prev => prev.map(p => p.id === activeProjectId ? project : p));
      setStep(project.status === 'dubbed' ? AppStep.RESULT : AppStep.EDITOR);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load transcription results');
    }
  };

  const startDubbing = async () => {
    if (!activeProject) return;

    try {
      setProcessingLogs([]);
      setProcessingProgress(0);
      setStep(AppStep.PROCESSING_DUBBING);

      // Sync full config to backend before starting dubbing
      const cfg = activeProject.config;
      await api.updateProjectConfig(activeProject.id, {
        sourceLang: cfg.sourceLang,
        targetLang: cfg.targetLang,
        personaId: cfg.personaId,
        speakerCount: cfg.speakerCount,
        keepBackground: cfg.keepBackground,
        pauseRemoval: cfg.pauseRemoval,
        preset: cfg.preset,
        // TTS settings
        ttsSystem: cfg.ttsSystem,
        ttsModel: cfg.ttsModel,
        voiceAutoSelection: cfg.voiceAutoSelection,
        enableEmotionEnrichment: cfg.enableEmotionEnrichment,
        ttsPromptPrefix: cfg.ttsPromptPrefix,
        // Audio settings
        dubbedVolume: cfg.dubbedVolume,
        backgroundVolume: cfg.backgroundVolume,
        useTwoPassEncoding: cfg.useTwoPassEncoding,
        videoQualityPreset: cfg.videoQualityPreset,
        // Processing settings
        maxWorkers: cfg.maxWorkers,
        // LLM settings (in case they matter for dubbing)
        llmProvider: cfg.llmProvider,
        llmModelName: cfg.llmModelName,
        llmTemperature: cfg.llmTemperature,
        speakerTtsPrompts: cfg.speakerTtsPrompts || {},
        speakerVoiceMappings: cfg.speakerVoiceMappings || {},
        refinementLlmProvider: cfg.refinementLlmProvider,
        refinementModelName: cfg.refinementModelName,
        refinementTemperature: cfg.refinementTemperature,
      });

      await api.startDubbing(activeProject.id);
      
      updateActiveProject({ 
        status: 'dubbing',
        processProgress: 0,
        processStage: 'Preparing...'
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start dubbing');
      setStep(AppStep.EDITOR);
    }
  };

  const handleDubbingComplete = () => {
    updateActiveProject({ status: 'dubbed' });
    setStep(AppStep.RESULT);
  };

  const resetToDashboard = () => {
    setActiveProjectId(null);
    setStep(AppStep.PROJECTS);
    setProcessingLogs([]);
    setProcessingProgress(0);
  };

  // --- Navigation Helpers ---

  const isProcessingStep = (s: AppStep) => 
    s === AppStep.PROCESSING_TRANSCRIPTION || s === AppStep.PROCESSING_DUBBING;

  const navigateToStep = (targetStep: AppStep) => {
    if (!activeProject) return;

    // Prevent navigation to Editor if not yet transcribed
    if (targetStep === AppStep.EDITOR && 
        (activeProject.status === 'draft' || activeProject.status === 'transcribing')) {
      return;
    }

    // If currently processing and trying to navigate away, show stop dialog
    if (isProcessingStep(step) && !isProcessingStep(targetStep)) {
      setPendingNavigationStep(targetStep);
      setShowStopDialog(true);
      return;
    }

    setStep(targetStep);
  };

  const handleStopAndNavigate = async () => {
    if (!activeProjectId || !pendingNavigationStep) return;

    setIsStopping(true);
    try {
      await api.stopProcessing(activeProjectId);
      
      // Update project status locally
      const newStatus: ProjectStatus = step === AppStep.PROCESSING_DUBBING ? 'transcribed' : 'draft';
      updateActiveProject({ 
        status: newStatus,
        processProgress: 0,
        processStage: undefined,
        isAutoProcessing: false,
      });

      setShowStopDialog(false);
      setStep(pendingNavigationStep);
      setPendingNavigationStep(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop processing');
    } finally {
      setIsStopping(false);
    }
  };

  const handleCancelStopDialog = () => {
    setShowStopDialog(false);
    setPendingNavigationStep(null);
  };

  // --- Segment Updates ---

  const handleUpdateSegments = async (newSegments: Segment[]) => {
    if (!activeProjectId) return;
    
    updateActiveProject({ segments: newSegments });

    // Sync changes to API (debounced in real implementation)
    for (const segment of newSegments) {
      try {
        await api.updateSegment(activeProjectId, segment.id, {
          translatedText: segment.translatedText,
          isMuted: segment.isMuted,
          voiceId: segment.voiceId,
          provider: segment.provider,
          ttsPrompt: segment.ttsPrompt,
        });
      } catch (err) {
        console.error('Failed to update segment:', err);
      }
    }
  };

  // --- Render Logic ---

  const settingsConfig = activeProject ? activeProject.config : globalConfig;
  const handleSettingsUpdate = activeProject 
    ? (newCfg: Partial<AppConfig>) => updateActiveProject({ config: { ...activeProject.config, ...newCfg } })
    : updateGlobalConfig;

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950 text-zinc-100">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-zinc-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-zinc-950 text-zinc-100">
      
      {/* Error Toast */}
      {error && (
        <div className="fixed top-4 right-4 z-[9999] bg-red-600 text-white px-4 py-3 rounded-lg shadow-2xl flex items-start gap-3 max-w-md border border-red-400">
          <span className="flex-1 break-words">{error}</span>
          <button onClick={() => setError(null)} className="text-white/70 hover:text-white shrink-0 mt-0.5">✕</button>
        </div>
      )}

      {/* Stop Processing Confirmation Dialog */}
      {showStopDialog && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl max-w-md w-full p-6 animate-in zoom-in-95 duration-200">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-6 h-6 text-amber-500" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Stop Processing?</h3>
                <p className="text-sm text-zinc-400">
                  Processing is currently in progress. Do you want to stop it and go back to settings?
                  {step === AppStep.PROCESSING_TRANSCRIPTION && (
                    <span className="block mt-2 text-zinc-500">
                      Note: All progress will be lost and you'll need to start over.
                    </span>
                  )}
                </p>
              </div>
              <button 
                onClick={handleCancelStopDialog}
                className="text-zinc-500 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCancelStopDialog}
                disabled={isStopping}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                Continue Processing
              </button>
              <button
                onClick={handleStopAndNavigate}
                disabled={isStopping}
                className="flex-1 px-4 py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isStopping ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Stopping...
                  </>
                ) : (
                  'Stop & Go Back'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Processing Project Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl max-w-md w-full p-6 animate-in zoom-in-95 duration-200">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-6 h-6 text-red-500" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Delete Processing Project?</h3>
                <p className="text-sm text-zinc-400">
                  This project is currently being processed. Deleting it will stop all processing and permanently remove the project.
                </p>
                <p className="text-sm text-zinc-500 mt-2">
                  This action cannot be undone.
                </p>
              </div>
              <button 
                onClick={handleCancelDeleteDialog}
                className="text-zinc-500 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCancelDeleteDialog}
                disabled={isDeleting}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                disabled={isDeleting}
                className="flex-1 px-4 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  'Stop & Delete'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Reset & Restart Confirmation Dialog */}
      {showResetDialog && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl max-w-md w-full p-6 animate-in zoom-in-95 duration-200">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-6 h-6 text-cyan-400" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Reset Cache & Restart?</h3>
                <p className="text-sm text-zinc-400">
                  This will clear cache/artifacts/results
                  {pendingResetProject ? ` for "${pendingResetProject.name}"` : ''} and start processing from scratch.
                </p>
                <p className="text-sm text-zinc-500 mt-2">
                  Uploaded source video and project settings will be preserved.
                </p>
              </div>
              <button
                onClick={handleCancelResetDialog}
                className="text-zinc-500 hover:text-white transition-colors"
                disabled={isResetting}
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCancelResetDialog}
                disabled={isResetting}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmResetAndRestart}
                disabled={isResetting}
                className="flex-1 px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isResetting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Restarting...
                  </>
                ) : (
                  'Reset & Restart'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Top Navigation Bar */}
      {step !== AppStep.PROJECTS && activeProject && (
        <header className="h-14 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm flex items-center justify-between px-4 shrink-0 z-50">
          <div className="flex items-center gap-2 text-sm">
            <button 
              onClick={resetToDashboard}
              className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors"
            >
              <LayoutDashboard className="w-4 h-4" />
              <span className="hidden sm:inline">Projects</span>
            </button>
            <ChevronRight className="w-4 h-4 text-zinc-600" />
            <span className="font-semibold text-white">{activeProject.name}</span>
            
            {/* Project Workflow Tabs */}
            <div className="ml-6 flex bg-zinc-950/50 rounded-lg p-1 border border-zinc-800">
              <button
                onClick={() => navigateToStep(AppStep.UPLOAD)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  step === AppStep.UPLOAD 
                    ? 'bg-zinc-800 text-white shadow-sm' 
                    : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Dubbing Props
              </button>
              <button
                onClick={() => navigateToStep(AppStep.EDITOR)}
                disabled={activeProject.status === 'draft' || activeProject.status === 'transcribing'}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  step === AppStep.EDITOR
                    ? 'bg-zinc-800 text-white shadow-sm' 
                    : (activeProject.status === 'draft' || activeProject.status === 'transcribing')
                      ? 'text-zinc-700 cursor-not-allowed' 
                      : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Editor
              </button>
              <button
                onClick={() => navigateToStep(AppStep.RESULT)}
                disabled={activeProject.status !== 'dubbed'}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  step === AppStep.RESULT
                    ? 'bg-zinc-800 text-white shadow-sm' 
                    : activeProject.status !== 'dubbed'
                      ? 'text-zinc-700 cursor-not-allowed' 
                      : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Result
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
             <button 
                onClick={() => setIsSettingsOpen(true)}
                className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
                title="Project Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
          </div>
        </header>
      )}

      {/* Main Content Area */}
      <main className="flex-1 relative overflow-hidden h-full">
        {step === AppStep.PROJECTS && (
          <ProjectListView 
            projects={projects}
            onSelectProject={handleSelectProject}
            onDeleteProject={handleDeleteProject}
            onBatchUpload={handleBatchUpload}
            onAutoProcess={handleAutoProcess}
            onStopProcess={handleStopProcess}
            onResetAndRestart={handleResetAndRestart}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onChangePreset={handleChangePreset}
            onChangeLanguages={handleChangeLanguages}
            onProgressUpdate={(projectId, progress, stage) => {
              setProjects(prev => prev.map(p =>
                p.id === projectId
                  ? { ...p, processProgress: progress, processStage: stage }
                  : p
              ));
            }}
            onStatusComplete={async (projectId, newStatus) => {
              try {
                const updatedProject = await api.getProject(projectId);
                setProjects(prev => prev.map(p =>
                  p.id === projectId
                    ? {
                        ...mapProjectFromApi(updatedProject),
                        isAutoProcessing: false,
                        processProgress: 0,
                        processStage: undefined,
                      }
                    : p
                ));
              } catch {
                setProjects(prev => prev.map(p =>
                  p.id === projectId
                    ? { ...p, status: newStatus, isAutoProcessing: false }
                    : p
                ));
              }
            }}
          />
        )}

        {step === AppStep.UPLOAD && activeProject && (
          <UploadView 
            config={activeProject.config}
            personas={personas}
            voices={voices}
            projectId={activeProject.id}
            videoFile={activeProject.videoFile}
            isUploading={activeProject.isUploading || false}
            uploadProgress={activeProject.uploadProgress || 0}
            onConfigChange={(newCfg) => updateActiveProject({ config: { ...activeProject.config, ...newCfg } })}
            onNext={startProcessing}
            onResetAndStart={() => handleResetAndRestart(activeProject.id)}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />
        )}

        {step === AppStep.PROCESSING_TRANSCRIPTION && activeProject && (
          <ProcessingView 
            title="Processing Video"
            description="Transcribing, translating, synthesizing voices, and rendering the final dubbed video."
            projectId={activeProject.id}
            onComplete={handleTranscriptionComplete}
            logs={processingLogs}
            progress={processingProgress}
            currentStep={processingStep}
          />
        )}

        {step === AppStep.EDITOR && activeProject && (
          <EditorView 
            projectId={activeProject.id}
            segments={activeProject.segments}
            videoFile={activeProject.videoFile}
            voices={voices}
            onUpdateSegments={handleUpdateSegments}
            onUpdateConfig={handleSettingsUpdate}
            onContinue={startDubbing}
            activeProject={activeProject}
          />
        )}

        {step === AppStep.PROCESSING_DUBBING && activeProject && (
          <ProcessingView 
            title="Generating Dubbed Video"
            description="Synthesizing voice audio, mixing tracks, and rendering the final output."
            projectId={activeProject.id}
            onComplete={handleDubbingComplete}
            logs={processingLogs}
            progress={processingProgress}
            currentStep={processingStep}
          />
        )}

        {step === AppStep.RESULT && activeProject && (
          <ResultView 
            projectId={activeProject.id}
            onReset={() => setStep(AppStep.EDITOR)} 
          />
        )}
      </main>

      {/* Global Modals */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)}
        config={settingsConfig}
        personas={personas}
        onUpdateConfig={handleSettingsUpdate}
        onUpdatePersonas={setPersonas}
        isGlobal={!activeProject}
      />
    </div>
  );
};

export default App;
