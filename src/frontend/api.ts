/**
 * API client for DubbLM backend.
 */

const API_BASE = '/api/v1';

interface ApiError {
  detail: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`,
      }));
      throw new Error(error.detail);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  // --- Projects ---

  async listProjects(): Promise<ProjectListResponse[]> {
    return this.request<ProjectListResponse[]>('/projects');
  }

  async createProject(name: string): Promise<ProjectResponse> {
    return this.request<ProjectResponse>('/projects', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  }

  async getProject(projectId: string): Promise<ProjectResponse> {
    return this.request<ProjectResponse>(`/projects/${projectId}`);
  }

  async deleteProject(projectId: string): Promise<void> {
    return this.request<void>(`/projects/${projectId}`, {
      method: 'DELETE',
    });
  }

  async updateProjectConfig(
    projectId: string,
    config: Partial<ProjectConfig>
  ): Promise<ProjectResponse> {
    return this.request<ProjectResponse>(`/projects/${projectId}/config`, {
      method: 'PATCH',
      body: JSON.stringify(config),
    });
  }

  // --- Upload ---

  async uploadVideo(
    projectId: string,
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/projects/${projectId}/upload`;

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);

      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        });
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || 'Upload failed'));
          } catch {
            reject(new Error(`Upload failed: ${xhr.statusText}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.send(formData);
    });
  }

  async downloadVideoFromUrl(
    projectId: string,
    url: string,
    quality: DownloadQuality = 'best'
  ): Promise<VideoDownloadResponse> {
    return this.request<VideoDownloadResponse>(`/projects/${projectId}/download-url`, {
      method: 'POST',
      body: JSON.stringify({ url, quality }),
    });
  }

  async getVideoInfo(
    url: string,
    quality: DownloadQuality = 'best'
  ): Promise<VideoInfoResponse> {
    return this.request<VideoInfoResponse>(`/projects/video-info`, {
      method: 'POST',
      body: JSON.stringify({ url, quality }),
    });
  }

  // --- Processing ---

  async startTranscription(projectId: string): Promise<JobResponse> {
    return this.request<JobResponse>(
      `/projects/${projectId}/process/transcribe`,
      { method: 'POST' }
    );
  }

  async queueAutoProcess(projectId: string): Promise<QueueAutoResponse> {
    return this.request<QueueAutoResponse>(
      `/projects/${projectId}/process/queue-auto`,
      { method: 'POST' }
    );
  }

  async retranslateProject(projectId: string): Promise<JobResponse> {
    return this.request<JobResponse>(
      `/projects/${projectId}/process/retranslate`,
      { method: 'POST' }
    );
  }

  async startDubbing(projectId: string): Promise<JobResponse> {
    return this.request<JobResponse>(`/projects/${projectId}/process/dub`, {
      method: 'POST',
    });
  }

  async getJobs(projectId: string): Promise<JobResponse[]> {
    return this.request<JobResponse[]>(`/projects/${projectId}/jobs`);
  }

  async getJobStatus(projectId: string, jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(
      `/projects/${projectId}/jobs/${jobId}/status`
    );
  }

  async stopProcessing(projectId: string): Promise<StopProcessingResponse> {
    return this.request<StopProcessingResponse>(
      `/projects/${projectId}/process/stop`,
      { method: 'POST' }
    );
  }

  async restartProcessing(projectId: string): Promise<JobResponse> {
    return this.request<JobResponse>(
      `/projects/${projectId}/process/restart`,
      { method: 'POST' }
    );
  }

  async resetTtsCache(projectId: string): Promise<{
    project_id: string;
    cache_files_removed: number;
    artifact_files_removed: number;
    segments_cleared: number;
  }> {
    return this.request(
      `/projects/${projectId}/process/reset-tts-cache`,
      { method: 'POST' }
    );
  }

  // --- Segments ---

  async getSegments(projectId: string): Promise<SegmentResponse[]> {
    return this.request<SegmentResponse[]>(`/projects/${projectId}/segments`);
  }

  async updateSegment(
    projectId: string,
    segmentId: string,
    data: SegmentUpdate
  ): Promise<SegmentResponse> {
    return this.request<SegmentResponse>(
      `/projects/${projectId}/segments/${segmentId}`,
      {
        method: 'PATCH',
        body: JSON.stringify(data),
      }
    );
  }

  async renameSpeaker(
    projectId: string,
    oldName: string,
    newName: string
  ): Promise<{ message: string }> {
    return this.request<{ message: string }>(
      `/projects/${projectId}/speakers/rename`,
      {
        method: 'POST',
        body: JSON.stringify({ oldName, newName }),
      }
    );
  }

  async updateSpeakerVoice(
    projectId: string,
    speakerName: string,
    voiceId: string,
    provider: string
  ): Promise<{ message: string }> {
    return this.request<{ message: string }>(
      `/projects/${projectId}/speakers/voice`,
      {
        method: 'POST',
        body: JSON.stringify({ speakerName, voiceId, provider }),
      }
    );
  }

  async updateSpeakerGender(
    projectId: string,
    speakerName: string,
    overrideGender: SpeakerGender | null
  ): Promise<{ message: string }> {
    return this.request<{ message: string }>(
      `/projects/${projectId}/speakers/gender`,
      {
        method: 'POST',
        body: JSON.stringify({ speakerName, overrideGender }),
      }
    );
  }

  async rephraseSegment(
    projectId: string,
    segmentId: string,
    prompt?: string
  ): Promise<{ translatedText: string }> {
    return this.request<{ translatedText: string }>(
      `/projects/${projectId}/segments/${segmentId}/rephrase`,
      {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      }
    );
  }

  async previewSegment(
    projectId: string,
    segmentId: string,
    forceRegenerate = false
  ): Promise<PreviewJobResponse> {
    return this.request<PreviewJobResponse>(
      `/projects/${projectId}/segments/${segmentId}/preview`,
      {
        method: 'POST',
        body: JSON.stringify({ forceRegenerate }),
      }
    );
  }

  async getPreviewStatus(
    projectId: string,
    segmentId: string,
    jobId?: string
  ): Promise<PreviewStatusResponse> {
    const query = jobId ? `?job_id=${jobId}` : '';
    return this.request<PreviewStatusResponse>(
      `/projects/${projectId}/segments/${segmentId}/preview/status${query}`
    );
  }

  // --- Resources ---

  async getVoices(provider?: string): Promise<VoiceResponse[]> {
    const query = provider ? `?provider=${provider}` : '';
    return this.request<VoiceResponse[]>(`/resources/voices${query}`);
  }

  async getPersonas(): Promise<PersonaResponse[]> {
    return this.request<PersonaResponse[]>('/resources/personas');
  }

  async createPersona(data: PersonaCreate): Promise<PersonaResponse> {
    return this.request<PersonaResponse>('/resources/personas', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getLanguages(): Promise<LanguageResponse[]> {
    return this.request<LanguageResponse[]>('/resources/languages');
  }

  // --- Settings ---

  async getSettings(): Promise<SettingsResponse> {
    return this.request<SettingsResponse>('/settings');
  }

  async updateSettings(data: SettingsUpdate): Promise<SettingsResponse> {
    return this.request<SettingsResponse>('/settings', {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async getApiKeys(): Promise<Record<string, ApiKeyStatus>> {
    return this.request<Record<string, ApiKeyStatus>>('/settings/api-keys');
  }

  async getPresets(): Promise<Record<string, PresetDefaultsResponse>> {
    return this.request<Record<string, PresetDefaultsResponse>>('/settings/presets');
  }

  async getPreset(preset: string): Promise<PresetDefaultsResponse> {
    return this.request<PresetDefaultsResponse>(`/settings/presets/${preset}`);
  }

  async setApiKey(
    provider: string,
    key: string
  ): Promise<ApiKeyStatus> {
    return this.request<ApiKeyStatus>(`/settings/api-keys/${provider}`, {
      method: 'PUT',
      body: JSON.stringify({ key }),
    });
  }

  async deleteApiKey(provider: string): Promise<void> {
    return this.request<void>(`/settings/api-keys/${provider}`, {
      method: 'DELETE',
    });
  }

  async uploadCookies(file: File): Promise<{ message: string; filename: string; size: number }> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/settings/cookies`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`,
      }));
      throw new Error(error.detail || 'Failed to upload cookies');
    }

    return response.json();
  }

  async getCookiesStatus(): Promise<CookiesStatusResponse> {
    return this.request<CookiesStatusResponse>('/settings/cookies');
  }

  async deleteCookies(): Promise<{ message: string; deleted: boolean }> {
    return this.request<{ message: string; deleted: boolean }>('/settings/cookies', {
      method: 'DELETE',
    });
  }

  // --- Downloads ---

  getVideoUrl(projectId: string): string {
    return `${this.baseUrl}/projects/${projectId}/video`;
  }

  getDownloadVideoUrl(projectId: string): string {
    return `${this.baseUrl}/projects/${projectId}/download/video`;
  }

  getStreamVideoUrl(projectId: string): string {
    return `${this.baseUrl}/projects/${projectId}/stream/video`;
  }

  getDownloadSubtitlesUrl(projectId: string, format: 'srt' | 'vtt' = 'srt', lang: 'source' | 'target' = 'target'): string {
    return `${this.baseUrl}/projects/${projectId}/download/subtitles?format=${format}&lang=${lang}`;
  }

  getPreviewAudioUrl(projectId: string, segmentId: string, voiceId?: string): string {
    const base = `${this.baseUrl}/projects/${projectId}/segments/${segmentId}/preview/audio`;
    return voiceId ? `${base}?voice_id=${encodeURIComponent(voiceId)}` : base;
  }

  getVideoFrameUrl(projectId: string, progress: number): string {
    const bucket = Math.round(progress / 5) * 5;
    return `${this.baseUrl}/projects/${projectId}/frame?progress=${bucket}`;
  }

  getVideoThumbnailUrl(projectId: string): string {
    return `${this.baseUrl}/projects/${projectId}/thumbnail`;
  }

  async getProjectStats(projectId: string): Promise<ProjectStatsResponse> {
    return this.request<ProjectStatsResponse>(`/projects/${projectId}/stats`);
  }

  async getProjectReport(projectId: string): Promise<ProjectReportResponse> {
    return this.request<ProjectReportResponse>(`/projects/${projectId}/report`);
  }

  getDownloadReportUrl(projectId: string): string {
    return `${this.baseUrl}/projects/${projectId}/download/report`;
  }

  // --- SSE Status Stream ---

  subscribeToStatus(
    projectId: string,
    handlers: StatusEventHandlers
  ): () => void {
    let eventSource: EventSource | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 2000;
    let isClosedManually = false;

    const connect = () => {
      const url = `${this.baseUrl}/projects/${projectId}/status`;
      eventSource = new EventSource(url);

      eventSource.addEventListener('progress', (e) => {
        reconnectAttempts = 0; // Reset on successful message
        const data = JSON.parse(e.data);
        handlers.onProgress?.(data.percent, data.step);
      });

      eventSource.addEventListener('log', (e) => {
        reconnectAttempts = 0;
        const data = JSON.parse(e.data);
        handlers.onLog?.(data);
      });

      eventSource.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data);
        handlers.onComplete?.(data.status, data.error);
        isClosedManually = true;
        eventSource?.close();
      });

      eventSource.addEventListener('timeout', () => {
        handlers.onTimeout?.();
        isClosedManually = true;
        eventSource?.close();
      });

      eventSource.onerror = () => {
        eventSource?.close();
        
        if (isClosedManually) return;

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = reconnectDelay * Math.pow(1.5, reconnectAttempts - 1);
          console.log(`SSE reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          setTimeout(() => {
            if (!isClosedManually) {
              connect();
            }
          }, delay);
        } else {
          // After max attempts, check project status directly
          console.log('SSE max reconnect attempts reached, checking project status directly');
          this.getProject(projectId)
            .then(project => {
              if (project.status === 'transcribed' || project.status === 'dubbed') {
                handlers.onComplete?.(project.status, undefined);
              } else if (project.status === 'error') {
                handlers.onComplete?.('error', 'Connection lost');
              } else {
                handlers.onError?.(new Error('SSE connection lost'));
              }
            })
            .catch(() => {
              handlers.onError?.(new Error('SSE connection error'));
            });
        }
      };
    };

    connect();

    return () => {
      isClosedManually = true;
      eventSource?.close();
    };
  }
}

// --- Types ---

export interface ProjectConfig {
  sourceLang?: string;
  targetLang?: string;
  personaId?: string;
  speakerCount?: number;
  keepBackground?: boolean;
  pauseRemoval?: 'cut' | 'disabled';
  preset?: string;
  apiKeys?: Record<string, string>;
  autoProcess?: boolean;  // Auto-continue to dubbing after transcription
  // Extra settings
  startTime?: number;
  duration?: number;
  transcriptionSystem?: string;
  whisperModel?: string;
  llmProvider?: string;
  llmModelName?: string;
  llmTemperature?: number;
  enableLlmEditor?: boolean;
  enableLlmTextAdjustment?: boolean;
  editorLlmProvider?: string;
  editorModelName?: string;
  editorTemperature?: number;
  editorReasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'none';
  speakerTtsPrompts?: Record<string, string>;
  speakerVoiceMappings?: Record<string, string>;
  enableSpeakerGenderInference?: boolean;
  speakerMetadata?: Record<string, SpeakerMetadata>;
  refinementLlmProvider?: string;
  refinementModelName?: string;
  refinementTemperature?: number;
  translationPromptPrefix?: string;
  ttsSystem?: string;
  ttsModel?: string;
  ttsStyle?: 'podcast' | 'lecture' | 'gothic' | 'news' | 'custom' | 'auto';
  ttsPromptPrefix?: string;
  resolvedTtsStyle?: 'podcast' | 'lecture' | 'gothic' | 'news';
  voiceAutoSelection?: boolean;
  enableEmotionEnrichment?: boolean;
  enableContentValidation?: boolean;
  contentValidatorProvider?: 'whisper' | 'assemblyai';
  contentValidatorWhisperModel?: string;
  contentValidatorWhisperComputeType?: string;
  contentValidatorWhisperCpuThreads?: number;
  contentValidatorSpeechModel?: string;
  dubbedVolume?: number;
  backgroundVolume?: number;
  keepOriginalAudioRanges?: string[];
  useTwoPassEncoding?: boolean;
  videoQualityPreset?: '720p' | '1080p' | 'original';
  maxWorkers?: number;
  postDiarizationMergeGap?: number;
  postTranslationMergeGap?: number;
  maxSegmentDuration?: number;
  minSegmentDuration?: number;
  comfortMinAdjustmentRatio?: number;
  comfortMaxAdjustmentRatio?: number;
  segmentStretch?: 'audio' | 'audio_and_video' | 'video';
  minPauseDuration?: number;
  preservePauseDuration?: number;
}

  export interface CookiesStatusResponse {
  configured: boolean;
  path: string | null;
  source: 'uploaded' | 'env';
}

export interface ProjectListResponse {
  id: string;
  name: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  config: ProjectConfig;
  speakerGenderTranslationStale?: boolean;
}

export interface ProjectResponse extends ProjectListResponse {
  sourceFile?: string;
  sourceFilename?: string;
  sourceSize?: number;
  segments?: SegmentResponse[];
}

export type SpeakerGender = 'male' | 'female' | 'unknown';

export interface SpeakerMetadata {
  inferredGender: SpeakerGender;
  inferredConfidence: number;
  rawLabel?: string | null;
  modelId?: string | null;
  overrideGender?: SpeakerGender | null;
}

export interface SegmentResponse {
  id: string;
  projectId: string;
  speaker: string;
  speakerColor: string;
  startTime: number;
  endTime: number;
  originalText: string;
  translatedText?: string;
  voiceId?: string;
  provider?: string;
  ttsPrompt?: string;
  isMuted: boolean;
  audioUrl?: string;
}

export interface SegmentUpdate {
  startTime?: number;
  endTime?: number;
  translatedText?: string;
  isMuted?: boolean;
  voiceId?: string;
  provider?: string;
  ttsPrompt?: string;
}

export interface UploadResponse {
  url: string;
  filename: string;
  size: number;
}

export interface VideoDownloadResponse {
  url: string;
  filename: string;
  size: number;
  title?: string | null;
  jobId: string;
  status: string;
  projectId: string;
}

export interface VideoInfoResponse {
  url: string;
  title?: string | null;
  duration?: number | null;
  uploader?: string | null;
}

export type DownloadQuality = 'best' | '1080p' | '720p' | '480p';

export interface JobResponse {
  jobId: string;
  status: string;
  projectId?: string;
  type?: string;
  progress: number;
  currentStep?: string;
}

export interface QueueAutoResponse {
  queued: boolean;
  started: boolean;
  jobId?: string;
  projectId: string;
  status: string;
}


export interface JobStatusResponse extends JobResponse {
  logs: LogEntry[];
  errorMessage?: string;
  createdAt?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface StopProcessingResponse {
  message: string;
  stopped: boolean;
}

export interface LogEntry {
  id: string;
  message: string;
  type: 'info' | 'success' | 'error';
  timestamp: string;
  text?: string;
}

export interface VoiceResponse {
  id: string;
  name: string;
  provider: string;
  gender?: string;
  language?: string;
  preview_url?: string;
}

export interface PersonaResponse {
  id: string;
  name: string;
  description?: string;
  promptTemplate?: string;
}

export interface PersonaCreate {
  name: string;
  description?: string;
  promptTemplate?: string;
}

export interface LanguageResponse {
  code: string;
  name: string;
}

export interface PreviewResponse {
  audioUrl: string;
  isCached: boolean;
}

export interface PreviewJobResponse {
  status: 'processing' | 'completed';
  jobId?: string;
  audioUrl?: string;
}

export interface PreviewStatusResponse {
  status: 'none' | 'processing' | 'completed' | 'failed';
  jobId?: string;
  audioUrl?: string;
  error?: string;
}

export interface ProjectStatsResponse {
  processingTimeSec: number;
  totalCost: number;
}

export interface ProjectReportResponse {
  markdown: string;
  json: unknown | null;
}

export interface SettingsResponse {
  apiKeys?: Record<string, string>;
  defaults?: Record<string, unknown>;
}

export interface SettingsUpdate {
  apiKeys?: Record<string, string>;
  defaults?: Record<string, unknown>;
}

export interface ApiKeyStatus {
  configured: boolean;
  masked?: string;
  provider?: string;
}

export interface PresetDefaultsResponse {
  personaId?: string;
  keepBackground?: boolean;
  llmProvider?: string;
  llmModelName?: string;
  llmTemperature?: number;
  enableLlmEditor?: boolean;
  enableLlmTextAdjustment?: boolean;
  editorLlmProvider?: string;
  editorModelName?: string;
  editorTemperature?: number;
  editorReasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'none';
  refinementLlmProvider?: string;
  refinementModelName?: string;
  refinementTemperature?: number;
  ttsSystem?: string;
  ttsModel?: string;
  ttsFallbackModel?: string;
  ttsStyle?: 'podcast' | 'lecture' | 'gothic' | 'news' | 'custom' | 'auto';
  ttsPromptPrefix?: string;
  voiceAutoSelection?: boolean;
  enableEmotionEnrichment?: boolean;
  enableContentValidation?: boolean;
  dubbedVolume?: number;
  backgroundVolume?: number;
  useTwoPassEncoding?: boolean;
  videoQualityPreset?: '720p' | '1080p' | 'original';
  maxWorkers?: number;
  pauseRemoval?: 'cut' | 'disabled';
  videoMinterpolateThreshold?: number;
}

export interface StatusEventHandlers {
  onProgress?: (percent: number, step: string) => void;
  onLog?: (log: LogEntry) => void;
  onComplete?: (status: string, error?: string) => void;
  onTimeout?: () => void;
  onError?: (error: Error) => void;
}

// Singleton instance
export const api = new ApiClient();

export default api;
