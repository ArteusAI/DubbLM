"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel, Field


# --- Project Schemas ---

SpeakerGenderValue = Literal["male", "female", "unknown"]


class SpeakerMetadata(BaseModel):
    """Speaker-level metadata used for translation and voice selection."""
    inferredGender: SpeakerGenderValue = "unknown"
    inferredConfidence: float = 0.0
    rawLabel: Optional[str] = None
    modelId: Optional[str] = None
    overrideGender: Optional[SpeakerGenderValue] = None

class ProjectConfig(BaseModel):
    """Project configuration schema."""
    sourceLang: Optional[str] = None
    targetLang: Optional[str] = None
    personaId: Optional[str] = None
    speakerCount: Optional[int] = None
    keepBackground: bool = False
    pauseRemoval: Literal["cut", "disabled"] = "disabled"
    preset: Optional[Literal["fast", "hq", "ultra"]] = "hq"
    apiKeys: Optional[Dict[str, str]] = None
    autoProcess: Optional[bool] = None  # Auto-continue to dubbing after transcription
    # Extra settings
    startTime: Optional[float] = None
    duration: Optional[float] = None
    transcriptionSystem: Optional[str] = None
    whisperModel: Optional[str] = None
    llmProvider: Optional[str] = None
    llmModelName: Optional[str] = None
    llmTemperature: Optional[float] = None
    enableLlmEditor: Optional[bool] = None
    editorLlmProvider: Optional[str] = None
    editorModelName: Optional[str] = None
    editorTemperature: Optional[float] = None
    editorReasoningEffort: Optional[Literal["minimal", "low", "medium", "high", "xhigh", "none"]] = None
    speakerTtsPrompts: Optional[Dict[str, str]] = None
    speakerVoiceMappings: Optional[Dict[str, str]] = None
    enableSpeakerGenderInference: Optional[bool] = True
    speakerMetadata: Optional[Dict[str, SpeakerMetadata]] = None
    refinementLlmProvider: Optional[str] = None
    refinementModelName: Optional[str] = None
    refinementTemperature: Optional[float] = None
    translationPromptPrefix: Optional[str] = None
    ttsSystem: Optional[str] = None
    ttsModel: Optional[str] = None
    ttsFallbackModel: Optional[str] = None
    ttsPromptPrefix: Optional[str] = None
    voiceAutoSelection: Optional[bool] = None
    enableEmotionEnrichment: Optional[bool] = None
    enableContentValidation: Optional[bool] = None
    dubbedVolume: Optional[float] = None
    backgroundVolume: Optional[float] = None
    keepOriginalAudioRanges: Optional[List[str]] = None
    useTwoPassEncoding: Optional[bool] = None
    videoQualityPreset: Optional[Literal["720p", "1080p", "original"]] = None
    maxWorkers: Optional[int] = None
    postDiarizationMergeGap: Optional[float] = None
    postTranslationMergeGap: Optional[float] = None
    maxSegmentDuration: Optional[float] = None
    minSegmentDuration: Optional[float] = None
    comfortMinAdjustmentRatio: Optional[float] = None
    comfortMaxAdjustmentRatio: Optional[float] = None
    minPauseDuration: Optional[float] = None
    preservePauseDuration: Optional[float] = None
    segmentStretch: Optional[Literal["audio", "audio_and_video", "video"]] = None


class ProjectConfigUpdate(BaseModel):
    """Schema for updating project configuration."""
    sourceLang: Optional[str] = None
    targetLang: Optional[str] = None
    personaId: Optional[str] = None
    speakerCount: Optional[int] = None
    keepBackground: Optional[bool] = None
    pauseRemoval: Optional[Literal["cut", "disabled"]] = None
    preset: Optional[Literal["fast", "hq", "ultra"]] = None
    apiKeys: Optional[Dict[str, str]] = None
    autoProcess: Optional[bool] = None  # Auto-continue to dubbing after transcription
    # Extra settings
    startTime: Optional[float] = None
    duration: Optional[float] = None
    transcriptionSystem: Optional[str] = None
    whisperModel: Optional[str] = None
    llmProvider: Optional[str] = None
    llmModelName: Optional[str] = None
    llmTemperature: Optional[float] = None
    enableLlmEditor: Optional[bool] = None
    editorLlmProvider: Optional[str] = None
    editorModelName: Optional[str] = None
    editorTemperature: Optional[float] = None
    editorReasoningEffort: Optional[Literal["minimal", "low", "medium", "high", "xhigh", "none"]] = None
    speakerTtsPrompts: Optional[Dict[str, str]] = None
    speakerVoiceMappings: Optional[Dict[str, str]] = None
    enableSpeakerGenderInference: Optional[bool] = None
    speakerMetadata: Optional[Dict[str, SpeakerMetadata]] = None
    refinementLlmProvider: Optional[str] = None
    refinementModelName: Optional[str] = None
    refinementTemperature: Optional[float] = None
    translationPromptPrefix: Optional[str] = None
    ttsSystem: Optional[str] = None
    ttsModel: Optional[str] = None
    ttsFallbackModel: Optional[str] = None
    ttsPromptPrefix: Optional[str] = None
    voiceAutoSelection: Optional[bool] = None
    enableEmotionEnrichment: Optional[bool] = None
    enableContentValidation: Optional[bool] = None
    dubbedVolume: Optional[float] = None
    backgroundVolume: Optional[float] = None
    keepOriginalAudioRanges: Optional[List[str]] = None
    useTwoPassEncoding: Optional[bool] = None
    videoQualityPreset: Optional[Literal["720p", "1080p", "original"]] = None
    maxWorkers: Optional[int] = None
    postDiarizationMergeGap: Optional[float] = None
    postTranslationMergeGap: Optional[float] = None
    maxSegmentDuration: Optional[float] = None
    minSegmentDuration: Optional[float] = None
    comfortMinAdjustmentRatio: Optional[float] = None
    comfortMaxAdjustmentRatio: Optional[float] = None
    minPauseDuration: Optional[float] = None
    preservePauseDuration: Optional[float] = None
    segmentStretch: Optional[Literal["audio", "audio_and_video", "video"]] = None


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""
    name: str = Field(..., min_length=1, max_length=255)


class ProjectResponse(BaseModel):
    """Schema for project response."""
    id: str
    name: str
    status: str
    createdAt: datetime
    updatedAt: datetime
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    speakerGenderTranslationStale: bool = False
    sourceFile: Optional[str] = None
    sourceFilename: Optional[str] = None
    sourceSize: Optional[int] = None
    segments: Optional[List["SegmentResponse"]] = None


class ProjectListResponse(BaseModel):
    """Schema for project list response."""
    id: str
    name: str
    status: str
    createdAt: datetime
    updatedAt: datetime
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    speakerGenderTranslationStale: bool = False


# --- Segment Schemas ---

class SegmentResponse(BaseModel):
    """Schema for segment response."""
    id: str
    projectId: str
    speaker: str
    speakerColor: str = "#3B82F6"
    startTime: float
    endTime: float
    originalText: str
    translatedText: Optional[str] = None
    voiceId: Optional[str] = None
    provider: Optional[str] = None
    ttsPrompt: Optional[str] = None
    isMuted: bool = False
    audioUrl: Optional[str] = None


class SegmentUpdate(BaseModel):
    """Schema for updating a segment."""
    startTime: Optional[float] = None
    endTime: Optional[float] = None
    translatedText: Optional[str] = None
    isMuted: Optional[bool] = None
    voiceId: Optional[str] = None
    provider: Optional[str] = None
    ttsPrompt: Optional[str] = None


class SpeakerRename(BaseModel):
    """Schema for renaming a speaker globally."""
    oldName: str
    newName: str


class SpeakerVoiceUpdate(BaseModel):
    """Schema for updating speaker voice globally."""
    speakerName: str
    voiceId: str
    provider: str


class SpeakerGenderUpdate(BaseModel):
    """Schema for setting or clearing a speaker gender override."""
    speakerName: str
    overrideGender: Optional[SpeakerGenderValue] = None


# --- Rephrase Schemas ---

class RephraseRequest(BaseModel):
    """Schema for rephrase request."""
    prompt: Optional[str] = None


class RephraseResponse(BaseModel):
    """Schema for rephrase response."""
    translatedText: str


# --- Preview Schemas ---

class PreviewRequest(BaseModel):
    """Schema for TTS preview request."""
    forceRegenerate: bool = False


class PreviewResponse(BaseModel):
    """Schema for TTS preview response (legacy)."""
    audioUrl: str
    isCached: bool


class PreviewJobResponse(BaseModel):
    """Schema for async preview job start response."""
    status: Literal["processing", "completed"]
    jobId: Optional[str] = None
    audioUrl: Optional[str] = None


class PreviewStatusResponse(BaseModel):
    """Schema for preview status check response."""
    status: Literal["none", "processing", "completed", "failed"]
    jobId: Optional[str] = None
    audioUrl: Optional[str] = None
    error: Optional[str] = None


# --- Upload Schemas ---

class UploadResponse(BaseModel):
    """Schema for upload response."""
    url: str
    filename: str
    size: int


# --- Job Schemas ---

class JobResponse(BaseModel):
    """Schema for job response."""
    jobId: str
    status: str
    projectId: Optional[str] = None
    type: Optional[str] = None
    progress: int = 0
    currentStep: Optional[str] = None


# --- Resource Schemas ---

class VoiceResponse(BaseModel):
    """Schema for voice resource."""
    id: str
    name: str
    provider: str
    gender: Optional[str] = None
    language: Optional[str] = None
    preview_url: Optional[str] = None


class PersonaResponse(BaseModel):
    """Schema for persona resource."""
    id: str
    name: str
    description: Optional[str] = None
    promptTemplate: Optional[str] = None


class PersonaCreate(BaseModel):
    """Schema for creating a persona."""
    name: str
    description: Optional[str] = None
    promptTemplate: Optional[str] = None


# --- SSE Event Schemas ---

class ProgressEvent(BaseModel):
    """Schema for progress event in SSE."""
    percent: int
    step: str


class LogEvent(BaseModel):
    """Schema for log event in SSE."""
    id: str
    message: str
    type: str = "info"
    timestamp: datetime


class CompleteEvent(BaseModel):
    """Schema for complete event in SSE."""
    status: str


# Update forward references
ProjectResponse.model_rebuild()
