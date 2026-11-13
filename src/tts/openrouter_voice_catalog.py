"""Catalog metadata and helpers for OpenRouter TTS voices (Qwen, Grok, …)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
import time

import requests

from src.utils.speaker_gender import DEFAULT_SPEAKER_GENDER, normalize_gender_value

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_SPEECH_MODELS_URL = f"{OPENROUTER_MODELS_URL}?output_modalities=speech"

GROK_VOICE_TTS_MODEL = "x-ai/grok-voice-tts-1.0"
GROK_MULTILINGUAL = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "pl", "tr",
    "ja", "ko", "zh", "ar", "hi", "id", "th", "vi", "sv", "da", "no", "fi",
]

FISH_DRAMA_TTS_MODEL = "fish-audio/drama-3"

# Public Fish Audio voice library. It is served without an API key and returns
# every public voice model with its id, title, description, language tags and
# a ready-made preview clip. Drama 3 selects a voice through this id
# (OpenRouter maps `voice` to the Fish `reference_id`).
FISH_VOICE_LIBRARY_URL = "https://api.fish.audio/model"
FISH_VOICE_LIBRARY_PAGE_SIZE = 100
FISH_VOICE_LIBRARY_TIMEOUT = 8
# How many Fish voices are exposed per language. The library holds 1000+
# entries; a small curated set keeps voice sample generation affordable.
FISH_VOICE_LIMIT = 24
# Below this many native voices the generic listing is pulled in as well.
FISH_VOICE_MIN_NATIVE = 8
# Drama 3 exposes a hand-picked cast for this language instead of the raw
# live library. Other languages keep the live-library fallback.
FISH_CURATED_LANGUAGE = "ru"

# Age buckets derived from the Fish voice tags. Voices outside the adult
# buckets sound childlike and are excluded from automatic selection.
FISH_AGE_TAG_MAP: Dict[str, set] = {
    "child": {"child", "kid", "kids", "toddler", "baby", "girl", "boy", "schoolgirl", "schoolboy"},
    "teen": {"teen", "teenager"},
    "young": {"young"},
    "adult": {"middle-aged", "adult"},
    "senior": {"old", "elderly", "senior"},
}
# Character voices (games, anime, cartoons) carry their own persona and are
# rarely usable for dubbing.
FISH_CHARACTER_TAGS: set = {"character-voice", "anime", "cartoon", "vtuber", "mascot"}
# Languages written in Cyrillic: a voice that claims the language but has no
# Cyrillic in its title/description was cloned from a foreign sample and will
# keep that accent.
FISH_CYRILLIC_LANGUAGES: set = {"ru", "uk", "be", "bg", "sr", "mk", "kk", "mn"}
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Speech models that reject the `voice` parameter: any value (even a voice id
# that is valid for another model) is answered with HTTP 400. They must be
# called without a voice and cannot take part in voice matching.
#
# Drama 3 is NOT in this set: it accepts Fish voice ids (verified - a valid id
# returns audio, an unknown id returns HTTP 400). It is called without a voice
# only when no Fish voice id is known for the segment.
VOICELESS_TTS_MODELS: set[str] = set()
_VOICELESS_TTS_MODELS_LOWER: set[str] = {name.lower() for name in VOICELESS_TTS_MODELS}

# Voices that are never picked automatically: meme, announcer and game
# character clones that sound out of place in a dubbed track, and voices that
# imitate real people. They stay visible in the UI and can be assigned
# manually.
FISH_BLOCKED_VOICES: set = {
    "0a690dbeb3984a9f88cd39353880775f",  # Меллстрой (мем "скуф")
    "90e65eaaf50e4470b8e6d43ee6afd7d5",  # Super Smash Bros. 4/Ultimate Announcer
    "6dc11f3f67a543f6ad4537a4a347e224",  # Mita Miside (game character)
    "5f9e15229e1c4f01a0133ab9f2b69566",  # Зеленский (имитация реального человека)
    "3f4b464b93c041cd80e3862dd31cb055",  # Путин (имитация реального человека)
    "f28d669a0eff409bbf07c4d3f12aabb3",  # голос путина (имитация реального человека)
    "c35aeeb5f9c145199fbffdbc2ef8ed95",  # Рената Литвинова (имитация реального человека)
    "4c774871d6524aa6b91f98612284daa0",  # Рената Литвинова (имитация реального человека)
    "8025b7b7b2ff4b93b85e670a6338f836",  # Карлос Солис (имитация реального человека)
    "b4b3cd5e89cc4fb682301b43e627d5ef",  # мориарти (клон персонажа)
    "cc1b79b1108f4ed3b8aac118ba6ebd07",  # мориарти (клон персонажа)
}

# Verified public Fish voices. Used as an offline fallback for the live
# library and as the guaranteed-available voice set for Drama 3.
FISH_VOICE_META: Dict[str, Dict[str, Any]] = {
    "bf322df2096a46f18c579d0baa36f41d": {
        "name": "Adrian",
        "gender": "male",
        "description": "Fish Audio public voice (EN-US, male).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en"],
        "accent": "en",
        "age": "adult",
        "character": False,
    },
    "c70cfb98867a4bfb9ae20d2401257b92": {
        "name": "Delia",
        "gender": "female",
        "description": "Fish Audio public voice (EN-US, female).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en"],
        "accent": "en",
        "age": "adult",
        "character": False,
    },
    "f3a2b90078d54a65af4b95f63bc7798e": {
        "name": "Sheila",
        "gender": "female",
        "description": "Fish Audio public voice (EN-US, female).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en"],
        "accent": "en",
        "age": "adult",
        "character": False,
    },
    "90e65eaaf50e4470b8e6d43ee6afd7d5": {
        "name": "Announcer",
        "gender": "male",
        "description": "Powerful, authoritative middle-aged male voice.",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en"],
        "accent": "en",
        "age": "adult",
        "character": False,
    },
    "0a690dbeb3984a9f88cd39353880775f": {
        "name": "Меллстрой",
        "gender": "male",
        "description": "Молодой мужской голос, уверенный и энергичный (RU).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
    },
    "2a1036d645634680b3cc69aeeb60375b": {
        "name": "Спокойный женский голос",
        "gender": "female",
        "description": "Спокойный, размеренный женский голос (RU).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "senior",
        "character": False,
    },
    "933563129e564b19a115bedd57b7406a": {
        "name": "Sarah",
        "gender": "female",
        "description": "Engaged, warm female speaker (EN).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en"],
        "accent": "en",
        "age": "young",
        "character": False,
    },
    "6dc11f3f67a543f6ad4537a4a347e224": {
        "name": "Mita Miside (RU voice)",
        "gender": "female",
        "description": "Женский голос персонажа Miside (EN/RU).",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["en", "ru"],
        "accent": "ru",
        "age": "young",
        "character": True,
    },
}

# Curated Russian Drama 3 cast. These voices were checked on three levels:
# the Fish preview, a Gemini audio review (adult / not squeaky / not harsh /
# natural Russian), and two real Drama 3 renders (neutral + dramatic). The
# shortlist deliberately favours normal adult voices and excludes children,
# anime/cartoon characters, memes and real-person impersonations.
#
# Order matters: the first male and the first female entry are the defaults
# for their gender (``default_voice_for`` keeps the input order).
FISH_CURATED_VOICES: Dict[str, Dict[str, Any]] = {
    "67ecc0169ec54f4eb99ff59680cd0458": {
        "name": "Тёплый рассказчик",
        "gender": "male",
        "description": "Спокойный, вдумчивый баритон. Повествование, документальные и драматические сцены. (Fish: «Спокойный Русский Рассказчик»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": True,
        "popularity": 323,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-10-04/46d1a049ce1845608d66b781713254a1.mp3",
    },
    "7312c38557eb4fb384e3874e8e9cea67": {
        "name": "Профессиональный баритон",
        "gender": "male",
        "description": "Чистый, уверенный, деловой. Новости, презентации, авторитетные персонажи. (Fish: «Мужской Профессиональный213»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 342131,
        "preview_url": "https://platform.r2.fish.audio/task/796ede3d8c554d4790fb7f2346415e94.mp3",
    },
    "fa3e3b7d3c2a44d88d9f018bc45db1ce": {
        "name": "Низкий русский",
        "gender": "male",
        "description": "Очень низкий, резонансный, уверенный. Антагонисты, властные роли, трейлеры. (Fish: «Russian»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 53027,
        "preview_url": "https://platform.r2.fish.audio/task/d79a5a79306249928a09df15b8ecb92a.mp3",
    },
    "d315712918f542e5bf866367e4a4975f": {
        "name": "Мягкий наблюдатель",
        "gender": "male",
        "description": "Тихий, внимательный, интимный. Драма, монологи, второй план. (Fish: «Спокойный Наблюдатель»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": True,
        "popularity": 114,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-10-13/e1bd19c827504a29b99be028ce32cefb.mp3",
    },
    "7f34b2a34b6e461aa2dc9632d7c737f3": {
        "name": "Тёплый баритон",
        "gender": "male",
        "description": "Ровный, спокойный, располагающий. Рассказчик, наставник, отец. (Fish: «Aram The1»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": True,
        "popularity": 167,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-07-29/15b68b5e901c4d29919757b8c14ff472.mp3",
    },
    "9f414bb4bdc14378854ad9320f9e6f0e": {
        "name": "Молодой разговорный",
        "gender": "male",
        "description": "Живой, дружелюбный, современный. Студенты, молодые специалисты, комедия. (Fish: «Молодой Парень»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 18810,
        "preview_url": "https://platform.r2.fish.audio/task/9f979b8c8a8c4a0ea8996c672050cbd9.mp3",
    },
    "6ada82e315e648b6a30a2e2210844cbf": {
        "name": "Дерзкий молодой",
        "gender": "male",
        "description": "Уверенный, расслабленный, с характером. Дерзкие и самоуверенные персонажи. (Fish: «Дерзкий Парень»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 179511,
        "preview_url": "https://platform.r2.fish.audio/task/c76338fd69624423a6d09b953cf22f7f.mp3",
    },
    "cfd7057a69204777af3a570d3e3ef164": {
        "name": "Тихий баритон",
        "gender": "male",
        "description": "Мягкий, сдержанный, тихий. Напряжённые сцены, тайны, уязвимость. (Fish: «Спокойный Мужской Голос»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 7397,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-07-28/664b789c7dc34c459e921dab8d8e9c9f.mp3",
    },
    "773ed9d827bb4c0d8b7a1b9267a4255e": {
        "name": "Спокойный резонансный",
        "gender": "male",
        "description": "Глубокий, ровный, уверенный. Универсальный мужской голос, повествование, драма. (Fish: «samoha8989»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": True,
        "popularity": 154,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-10-17/c3669054b4dc4dea8056f578eb166418.mp3",
    },
    "172565be42474bf093d16c9b60885357": {
        "name": "Сбалансированный баритон",
        "gender": "male",
        "description": "Средний по высоте, ровный, естественный. Универсальные взрослые роли. (Fish: «Замена 1»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 14327,
        "preview_url": "https://platform.r2.fish.audio/task/3728d8a073b641dfb0dca103750039c2.mp3",
    },
    "9cc4883d4bbe4276bb67584db3cd117b": {
        "name": "Спокойная рассказчица",
        "gender": "female",
        "description": "Тёплый, ровный, естественный альт. Универсальный женский голос, повествование. (Fish: «Спокойный Женский Голос»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": True,
        "popularity": 596,
        "preview_url": "https://platform.r2.fish.audio/tasks/2026-09-30/2d0f53c5a69048908ad93c20353f71f1.mp3",
    },
    "12f7ab6416aa490880e63c864a434618": {
        "name": "Мягкий интимный голос",
        "gender": "female",
        "description": "Тихий, нежный, доверительный. Драма, близкие сцены, монологи. (Fish: «Нежный Голос»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 5434,
        "preview_url": "https://platform.r2.fish.audio/task/4b85c1fdfbc345bc85112061745b799c.mp3",
    },
    "4bf88e0a955748358c0e06a7b656b2af": {
        "name": "Ясный молодой альт",
        "gender": "female",
        "description": "Светлый, чёткий, энергичный. Молодые героини, активные сцены. (Fish: «Женский Рассказчик»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 12137,
        "preview_url": "https://platform.r2.fish.audio/task/cfd3a73897ef4f1ca9c5678fff4f97a9.mp3",
    },
    "345193fc9ae942888141767d2bc9c4c9": {
        "name": "Чистый молодой голос",
        "gender": "female",
        "description": "Ясный, ровный, молодой. Повседневные сцены, героини 20–30 лет. (Fish: «Женский Голос»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "young",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 28319,
        "preview_url": "https://platform.r2.fish.audio/task/76fc22d00900418c8f3eaf6cb8d06e00.mp3",
    },
    "9a6b5d73a86042dc8191a8c62adfbe47": {
        "name": "Тёплый мелодичный альт",
        "gender": "female",
        "description": "Взрослый, плавный, мелодичный. Драматические и повествовательные роли. (Fish: «женский»)",
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": ["ru"],
        "accent": "ru",
        "age": "adult",
        "character": False,
        "curated": True,
        "licensed": False,
        "popularity": 5697,
        "preview_url": "https://platform.r2.fish.audio/task/77421c82bec7412b9a69289609a360a3.mp3",
    },
}

# Verified working on OpenRouter. DashScope names (Cherry/Alek/…) do NOT work for Qwen here.
OPENROUTER_VOICE_META: Dict[str, Dict[str, Any]] = {
    "loongjohn": {
        "name": "Loongjohn",
        "gender": "male",
        "description": "Male multilingual voice (flash). Good general-purpose RU/EN.",
        "models": ["qwen/qwen-audio-3.0-tts-flash"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanhuan_v3.6": {
        "name": "Longanhuan",
        "gender": "female",
        "description": "Female multilingual voice (flash). Good general-purpose RU/EN.",
        "models": ["qwen/qwen-audio-3.0-tts-flash"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanlingxin": {
        "name": "Longanlingxin",
        "gender": "female",
        "description": "Female multilingual voice (plus). Higher quality lattice.",
        "models": ["qwen/qwen-audio-3.0-tts-plus"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanlufeng": {
        "name": "Longanlufeng",
        "gender": "male",
        "description": "Male multilingual voice (plus). Higher quality lattice.",
        "models": ["qwen/qwen-audio-3.0-tts-plus"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "eve": {
        "name": "Eve",
        "gender": "female",
        "description": "Grok Voice TTS. Warm, clear female voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "ara": {
        "name": "Ara",
        "gender": "female",
        "description": "Grok Voice TTS. Expressive female voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "rex": {
        "name": "Rex",
        "gender": "male",
        "description": "Grok Voice TTS. Confident male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "sal": {
        "name": "Sal",
        "gender": "male",
        "description": "Grok Voice TTS. Calm male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "leo": {
        "name": "Leo",
        "gender": "male",
        "description": "Grok Voice TTS. Energetic male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
}

# Fallback static model -> voices if Models API is unreachable.
STATIC_MODEL_VOICES: Dict[str, List[str]] = {
    "qwen/qwen-audio-3.0-tts-flash": ["loongjohn", "longanhuan_v3.6"],
    "qwen/qwen-audio-3.0-tts-plus": ["longanlingxin", "longanlufeng"],
    GROK_VOICE_TTS_MODEL: ["eve", "ara", "rex", "sal", "leo"],
    # Drama 3 selects a voice by Fish voice id; the Russian curated cast comes
    # first and the verified fallback stays available for other languages.
    FISH_DRAMA_TTS_MODEL: list(FISH_CURATED_VOICES.keys())
    + [voice_id for voice_id in FISH_VOICE_META if voice_id not in FISH_CURATED_VOICES],
}

ALL_OPENROUTER_VOICES: List[str] = list(OPENROUTER_VOICE_META.keys())

DEFAULT_VOICE_BY_GENDER: Dict[str, str] = {
    "male": "loongjohn",
    "female": "longanhuan_v3.6",
    "unknown": "loongjohn",
}

# Preferred defaults when target language is Russian.
DEFAULT_RU_VOICE_BY_GENDER: Dict[str, str] = {
    "male": "loongjohn",
    "female": "longanhuan_v3.6",
    "unknown": "loongjohn",
}

DEFAULT_VOICE_BY_MODEL: Dict[str, str] = {
    "qwen/qwen-audio-3.0-tts-flash": "loongjohn",
    "qwen/qwen-audio-3.0-tts-plus": "longanlufeng",
    GROK_VOICE_TTS_MODEL: "eve",
    FISH_DRAMA_TTS_MODEL: next(iter(FISH_CURATED_VOICES), "c70cfb98867a4bfb9ae20d2401257b92"),
}


def is_qwen_tts_model(model: Optional[str]) -> bool:
    return str(model or "").strip().lower().startswith("qwen/")


def is_grok_tts_model(model: Optional[str]) -> bool:
    key = str(model or "").strip().lower()
    return key.startswith("x-ai/") or "grok-voice-tts" in key


def is_fish_drama_tts_model(model: Optional[str]) -> bool:
    """True for Fish Audio Drama 3 (directions in square brackets, Fish voices)."""
    key = str(model or "").strip().lower()
    return key == FISH_DRAMA_TTS_MODEL or key.endswith("/drama-3")


def resolve_voice_cloning_enabled(
    requested: Optional[Any],
    model: Optional[str],
) -> bool:
    """Resolve the per-model voice-cloning default.

    Drama 3 clones by default when the project has no explicit value.  Other
    models keep the previous behavior, and an explicit ``false`` is respected.
    """
    if not is_fish_drama_tts_model(model):
        return False
    if requested is None:
        return True
    if isinstance(requested, str):
        return requested.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(requested)


def is_voiceless_tts_model(model: Optional[str]) -> bool:
    """True when the model must be called without a `voice` parameter."""
    key = str(model or "").strip().lower()
    return bool(key) and key in _VOICELESS_TTS_MODELS_LOWER


def is_fish_voice_id(voice_name: Optional[str]) -> bool:
    """True for a Fish Audio voice model id (32 hex characters)."""
    key = str(voice_name or "").strip().lower()
    return bool(key) and len(key) == 32 and all(c in "0123456789abcdef" for c in key)


def max_char_limit_for_model(model: Optional[str]) -> int:
    """Per-request character limit for OpenRouter speech models."""
    if is_grok_tts_model(model):
        return 15000
    return 2000

# Reference only: DashScope system voice names that support Russian natively.
# Not accepted by OpenRouter Qwen endpoints (verified 400). Kept for docs / future.
DASHSCOPE_RU_VOICES: List[Dict[str, str]] = [
    {"id": "Alek", "name": "Alek", "gender": "male",
     "description": "Cold like the Russian spirit — dedicated RU-character male voice (DashScope only)"},
    {"id": "Cherry", "name": "Cherry", "gender": "female", "description": "Sunny young woman"},
    {"id": "Serena", "name": "Serena", "gender": "female", "description": "Gentle young woman"},
    {"id": "Ethan", "name": "Ethan", "gender": "male", "description": "Warm energetic man"},
    {"id": "Neil", "name": "Neil", "gender": "male", "description": "News-anchor male"},
    {"id": "Katerina", "name": "Katerina", "gender": "female", "description": "Mature woman, rich rhythm"},
    {"id": "Jennifer", "name": "Jennifer", "gender": "female", "description": "Cinematic American English female"},
    {"id": "Ryan", "name": "Ryan", "gender": "male", "description": "Dramatic male"},
    {"id": "Moon", "name": "Moon", "gender": "male", "description": "Bold handsome man"},
    {"id": "Kai", "name": "Kai", "gender": "male", "description": "Soothing male"},
    {"id": "Mia", "name": "Mia", "gender": "female", "description": "Gentle female"},
    {"id": "Vivian", "name": "Vivian", "gender": "female", "description": "Confident cute female"},
    {"id": "Maia", "name": "Maia", "gender": "female", "description": "Intellectual gentle female"},
    {"id": "Chelsie", "name": "Chelsie", "gender": "female", "description": "Anime-style girlfriend"},
    {"id": "Momo", "name": "Momo", "gender": "female", "description": "Playful mischievous female"},
    {"id": "Bella", "name": "Bella", "gender": "female", "description": "Bubbly playful female"},
    {"id": "Nofish", "name": "Nofish", "gender": "male", "description": "Designer male"},
    {"id": "Aiden", "name": "Aiden", "gender": "male", "description": "American English young man"},
    {"id": "Eldric Sage", "name": "Eldric Sage", "gender": "male", "description": "Calm wise elder"},
    {"id": "Mochi", "name": "Mochi", "gender": "male", "description": "Clever childlike adult male"},
    {"id": "Bellona", "name": "Bellona", "gender": "female", "description": "Powerful clear hero voice"},
    {"id": "Vincent", "name": "Vincent", "gender": "male", "description": "Raspy smoky male"},
    {"id": "Bunny", "name": "Bunny", "gender": "female", "description": "Cute little girl"},
    {"id": "Elias", "name": "Elias", "gender": "female", "description": "Academic storytelling female"},
    {"id": "Arthur", "name": "Arthur", "gender": "male", "description": "Earthy storytelling male"},
    {"id": "Nini", "name": "Nini", "gender": "female", "description": "Soft clingy female"},
    {"id": "Seren", "name": "Seren", "gender": "female", "description": "Soothing sleep female"},
    {"id": "Pip", "name": "Pip", "gender": "male", "description": "Playful boy"},
    {"id": "Stella", "name": "Stella", "gender": "female", "description": "Sweet teenage girl"},
    {"id": "Andre", "name": "Andre", "gender": "male", "description": "Magnetic steady male"},
]

_live_model_voices_cache: Dict[str, Tuple[float, List[str]]] = {}
_fish_voice_cache: Dict[str, Tuple[float, Dict[str, Dict[str, Any]]]] = {}
_LIVE_CACHE_TTL_SECONDS = 3600.0


def get_openrouter_voice_gender(voice_name: Optional[str]) -> str:
    if not voice_name:
        return DEFAULT_SPEAKER_GENDER
    key = str(voice_name).strip()
    meta = OPENROUTER_VOICE_META.get(key)
    if not meta:
        meta = fish_voice_meta(allow_live=False).get(key)
    if not meta:
        return DEFAULT_SPEAKER_GENDER
    return normalize_gender_value(meta.get("gender"))


def default_voice_for(
    model: Optional[str] = None,
    language: Optional[str] = None,
    speaker_gender: Optional[str] = None,
) -> str:
    """Pick a sensible default OpenRouter voice for model/language/gender."""
    model_key = (model or "").strip()
    if is_voiceless_tts_model(model_key):
        # Models without voice support are called without a voice at all.
        return ""

    lang = (language or "").strip().lower()
    gender = normalize_gender_value(speaker_gender)
    allowed = voices_for_model(model_key, language=lang) if model_key else list(ALL_OPENROUTER_VOICES)

    model_default = DEFAULT_VOICE_BY_MODEL.get(model_key)

    if is_fish_drama_tts_model(model_key):
        # Fish voices are opaque ids; match the speaker gender and prefer a
        # voice that natively speaks the dubbing language. The static model
        # default is only a last resort - it may have a foreign accent.
        if gender in ("male", "female"):
            same_gender = filter_fish_voices(
                allowed,
                language=lang,
                gender=gender,
                adult_only=True,
                native_only=True,
            )
            if same_gender:
                return same_gender[0]
        native = filter_fish_voices(
            allowed,
            language=lang,
            adult_only=True,
            native_only=True,
        )
        if native:
            return native[0]
        if model_default and model_default in allowed:
            return model_default
        return allowed[0] if allowed else ""

    # Model default first when it matches gender (or gender unknown).
    if model_default and model_default in allowed:
        if gender == DEFAULT_SPEAKER_GENDER or get_openrouter_voice_gender(model_default) == gender:
            return model_default

    # Global language/gender prefs only if they belong to the active model.
    if lang.startswith("ru"):
        preferred = DEFAULT_RU_VOICE_BY_GENDER.get(gender, DEFAULT_RU_VOICE_BY_GENDER["unknown"])
    else:
        preferred = DEFAULT_VOICE_BY_GENDER.get(gender, DEFAULT_VOICE_BY_GENDER["unknown"])
    if preferred in allowed:
        return preferred

    # Prefer same gender among allowed, else model default, else first allowed.
    for voice_id in allowed:
        if get_openrouter_voice_gender(voice_id) == gender:
            return voice_id
    if model_default and model_default in allowed:
        return model_default
    return allowed[0] if allowed else ALL_OPENROUTER_VOICES[0]


def curated_fish_voice_ids(language: Optional[str] = None) -> List[str]:
    """The hand-picked Drama 3 cast for a language (currently Russian only)."""
    lang = (language or "").strip().lower()
    if lang and not lang.startswith(FISH_CURATED_LANGUAGE):
        return []
    return list(FISH_CURATED_VOICES.keys())


def voices_for_model(
    model: str,
    *,
    allow_live: bool = True,
    language: Optional[str] = None,
) -> List[str]:
    """Return voice IDs supported for a model (live API + static fallback)."""
    model_key = (model or "").strip()
    if is_voiceless_tts_model(model_key):
        # No selectable voices: the API rejects every `voice` value.
        return []
    if is_fish_drama_tts_model(model_key):
        curated = curated_fish_voice_ids(language)
        if curated:
            return curated
        voices = list(
            fish_voice_meta(language=language, allow_live=allow_live).keys()
        )
        if language and not language.strip().lower().startswith(FISH_CURATED_LANGUAGE):
            # The Russian curated cast must not leak into other languages.
            voices = [voice for voice in voices if voice not in FISH_CURATED_VOICES]
        return voices
    if allow_live:
        live = _fetch_live_model_voices(model_key)
        if live:
            return live
    if model_key in STATIC_MODEL_VOICES:
        return list(STATIC_MODEL_VOICES[model_key])
    return list(ALL_OPENROUTER_VOICES)


def _fetch_live_model_voices(model: str) -> Optional[List[str]]:
    model_key = (model or "").strip()
    if not model_key:
        return None

    now = time.time()
    cached = _live_model_voices_cache.get(model_key)
    if cached and (now - cached[0]) < _LIVE_CACHE_TTL_SECONDS:
        return list(cached[1])

    try:
        response = requests.get(OPENROUTER_SPEECH_MODELS_URL, timeout=8)
        response.raise_for_status()
        payload = response.json() or {}
        rows = payload.get("data") or []
        found: Optional[List[str]] = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or "").strip()
            if row_id != model_key and not row_id.startswith(model_key):
                continue
            voices = row.get("supported_voices") or []
            cleaned = [str(v).strip() for v in voices if str(v).strip()]
            if cleaned:
                found = cleaned
                break
        if found:
            _live_model_voices_cache[model_key] = (now, found)
            return list(found)
    except Exception:
        return None
    return None


def _fish_profile_from_item(
    item: Dict[str, Any],
    target_language: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Build a voice profile from a Fish library entry."""
    voice_id = str(item.get("_id") or "").strip()
    if not voice_id:
        return None

    tags = {str(tag).strip().lower() for tag in (item.get("tags") or [])}
    if "male" in tags:
        gender = "male"
    elif "female" in tags:
        gender = "female"
    else:
        gender = DEFAULT_SPEAKER_GENDER

    age = "unknown"
    for bucket, keys in FISH_AGE_TAG_MAP.items():
        if tags & keys:
            age = bucket
            break

    languages = [
        str(code).strip().lower()
        for code in (item.get("languages") or [])
        if str(code).strip()
    ]
    title = str(item.get("title") or voice_id).strip()
    description = str(item.get("description") or "").strip()
    accent = languages[0] if languages else ""

    try:
        popularity = int(item.get("task_count") or 0)
    except (TypeError, ValueError):
        popularity = 0

    licensed = bool(item.get("licensed"))

    preview_url = ""
    for sample in item.get("samples") or []:
        if isinstance(sample, dict) and sample.get("audio"):
            preview_url = str(sample["audio"])
            break

    return voice_id, {
        "name": title,
        "gender": gender,
        "age": age,
        "character": bool(tags & FISH_CHARACTER_TAGS),
        "description": description,
        "models": [FISH_DRAMA_TTS_MODEL],
        "languages": languages,
        "accent": accent,
        "popularity": popularity,
        "licensed": licensed,
        "preview_url": preview_url,
    }


def _is_native_accent(
    accent: str,
    title: str,
    description: str,
    target_language: str,
) -> bool:
    """True when the voice natively speaks the target language.

    A Cyrillic-language voice without any Cyrillic in its title/description
    was cloned from a foreign sample and keeps that accent.
    """
    if not accent or not target_language:
        return False
    if accent != target_language:
        return False
    if target_language in FISH_CYRILLIC_LANGUAGES:
        if not _CYRILLIC_RE.search(f"{title} {description}"):
            return False
    return True


def _fetch_fish_page(language: Optional[str]) -> List[Dict[str, Any]]:
    """Fetch one page of the Fish voice library (optionally language filtered)."""
    params: Dict[str, Any] = {"page_size": FISH_VOICE_LIBRARY_PAGE_SIZE, "page_number": 1}
    if language:
        params["language"] = language
    try:
        response = requests.get(
            FISH_VOICE_LIBRARY_URL,
            params=params,
            timeout=FISH_VOICE_LIBRARY_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json() or {}
    except Exception:
        return []
    items = payload.get("items") or []
    return [item for item in items if isinstance(item, dict)]


def _fetch_fish_voice_library(language: Optional[str] = None) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch the public Fish Audio voice library (no API key required).

    The library is queried with a language filter first, so the returned set
    is dominated by voices that natively speak the dubbing language. The
    generic listing is added when the language filter is too narrow.
    """
    lang = (language or "").strip().lower() or "en"
    now = time.time()
    cached = _fish_voice_cache.get(lang)
    if cached and (now - cached[0]) < _LIVE_CACHE_TTL_SECONDS:
        return {key: dict(value) for key, value in cached[1].items()}

    parsed: List[Tuple[str, Dict[str, Any]]] = []
    seen: set = set()
    for item in _fetch_fish_page(lang):
        built = _fish_profile_from_item(item, lang)
        if not built:
            continue
        voice_id, profile = built
        if voice_id in seen:
            continue
        seen.add(voice_id)
        parsed.append((voice_id, profile))

    if not parsed:
        # The language filtered page came back empty (library unreachable or
        # no voices for that language) - do not hammer the endpoint again.
        return None

    native_count = sum(
        1
        for _, profile in parsed
        if _is_native_accent(
            profile.get("accent", ""),
            profile.get("name", ""),
            profile.get("description", ""),
            lang,
        )
    )
    if native_count < FISH_VOICE_MIN_NATIVE:
        # Not enough native voices: pull the generic listing as well.
        for item in _fetch_fish_page(None):
            built = _fish_profile_from_item(item, lang)
            if not built:
                continue
            voice_id, profile = built
            if voice_id in seen:
                continue
            seen.add(voice_id)
            parsed.append((voice_id, profile))

    if not parsed:
        return None

    def _rank(pair: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, int]:
        profile = pair[1]
        native = _is_native_accent(
            profile.get("accent", ""),
            profile.get("name", ""),
            profile.get("description", ""),
            lang,
        )
        if native:
            tier = 0
        elif profile.get("accent") == lang:
            tier = 1
        else:
            tier = 2
        # Voices whose rights were secured by Fish Audio come first.
        licensed = 0 if profile.get("licensed") else 1
        return (tier, licensed, -int(profile.get("popularity") or 0))

    parsed.sort(key=_rank)

    selected: Dict[str, Dict[str, Any]] = {}
    for voice_id, profile in parsed:
        if len(selected) >= FISH_VOICE_LIMIT:
            break
        selected[voice_id] = profile
    # The verified Drama 3 voices stay available even when the library is
    # re-ranked or a target language has no native voices.
    for voice_id, meta in FISH_VOICE_META.items():
        selected.setdefault(voice_id, dict(meta))

    _fish_voice_cache[lang] = (now, selected)
    return {key: dict(value) for key, value in selected.items()}


def fish_voice_meta(
    language: Optional[str] = None,
    *,
    allow_live: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fish Audio voices (curated + live library + verified fallback)."""
    requested_lang = (language or "").strip().lower()
    lang = requested_lang or "en"
    # Curated entries first: they carry the hand-checked metadata and must win
    # over the same voice coming back from the live library.
    meta: Dict[str, Dict[str, Any]] = {
        key: dict(value) for key, value in FISH_CURATED_VOICES.items()
    }
    for key, value in FISH_VOICE_META.items():
        meta.setdefault(key, dict(value))
    if allow_live:
        live = _fetch_fish_voice_library(lang)
        if live:
            for key, value in live.items():
                if key not in FISH_CURATED_VOICES:
                    meta[key] = value
    # The "native" flag depends on the dubbing language, so it is computed
    # per request instead of being cached with the library entry.
    for profile in meta.values():
        profile.setdefault("curated", False)
        if profile.get("curated"):
            profile["native"] = (
                not requested_lang
                or requested_lang.startswith(FISH_CURATED_LANGUAGE)
            )
            continue
        profile["native"] = _is_native_accent(
            profile.get("accent", ""),
            profile.get("name", ""),
            profile.get("description", ""),
            lang,
        )
    return meta


def fish_voice_profile(
    voice_id: Optional[str],
    language: Optional[str] = None,
    *,
    allow_live: bool = True,
) -> Dict[str, Any]:
    """Profile of a single Fish voice (gender, age, accent, native flag)."""
    key = str(voice_id or "").strip()
    if not key:
        return {}
    return fish_voice_meta(language=language, allow_live=allow_live).get(key, {})


def filter_fish_voices(
    voice_ids: Iterable[str],
    *,
    language: Optional[str] = None,
    gender: Optional[str] = None,
    adult_only: bool = True,
    native_only: bool = True,
    allow_character: bool = False,
    allow_blocked: bool = False,
    allow_live: bool = True,
    extra_blocked: Optional[Iterable[str]] = None,
) -> List[str]:
    """Filter Fish voices by gender, age and accent.

    ``gender`` accepts male/female; unknown or missing gender keeps voices
    whose gender could not be determined. Globally blocked voices (meme and
    character clones) are excluded unless ``allow_blocked`` is set; project
    level blocks can be passed via ``extra_blocked``.
    """
    wanted_gender = normalize_gender_value(gender, default="")
    blocked = set(FISH_BLOCKED_VOICES)
    if extra_blocked:
        blocked.update(str(item).strip().lower() for item in extra_blocked if str(item).strip())
    meta = fish_voice_meta(language=language, allow_live=allow_live)
    result: List[str] = []
    for voice_id in voice_ids:
        if not allow_blocked and voice_id.strip().lower() in blocked:
            continue
        profile = meta.get(voice_id)
        if not profile:
            continue
        voice_gender = normalize_gender_value(profile.get("gender"), default="")
        if wanted_gender and voice_gender and voice_gender != wanted_gender:
            continue
        if adult_only and profile.get("age") in ("child", "teen"):
            continue
        if not allow_character and profile.get("character"):
            continue
        if native_only and not profile.get("native"):
            continue
        result.append(voice_id)
    return result


FISH_GENDER_LABELS: Dict[str, str] = {
    "male": "муж.",
    "female": "жен.",
}
FISH_AGE_LABELS: Dict[str, str] = {
    "child": "детский",
    "teen": "подросток",
    "young": "молодой",
    "senior": "старший",
}


def _fish_voice_label(profile: Dict[str, Any]) -> str:
    """Human readable label: name (gender, accent[, age])."""
    label = str(profile.get("name") or "").strip()
    gender = str(profile.get("gender") or "").strip().lower()
    accent = str(profile.get("accent") or "").strip().lower()
    age = str(profile.get("age") or "").strip().lower()

    parts: List[str] = []
    if gender in FISH_GENDER_LABELS:
        parts.append(FISH_GENDER_LABELS[gender])
    if accent:
        parts.append(accent)
    if age in FISH_AGE_LABELS:
        parts.append(FISH_AGE_LABELS[age])
    if not parts:
        return label
    return f"{label} ({', '.join(parts)})"


def build_openrouter_voice_entries(
    *,
    model: Optional[str] = None,
    include_all_known: bool = True,
    language: Optional[str] = None,
    allow_live: bool = False,
) -> List[Dict[str, Any]]:
    """Build UI voice entries for OpenRouter provider.

    Each entry carries the list of models that accept the voice so the
    frontend can show only the voices that work with the selected model.
    Live lookups (the Fish Audio voice library) are opt-in: they perform
    network I/O and must not run at import time.
    """
    entries: List[Dict[str, Any]] = []
    seen = set()

    fish_meta = fish_voice_meta(language=language, allow_live=allow_live)
    # Curated voices keep their hand-picked order (the first male and the
    # first female are the defaults). Other voices: native first, then
    # popular ones, so the dubbing language does not sound foreign.
    curated_ids = [
        voice_id for voice_id in FISH_CURATED_VOICES if voice_id in fish_meta
    ]
    other_ids = [
        voice_id for voice_id in fish_meta if voice_id not in FISH_CURATED_VOICES
    ]
    other_ids.sort(
        key=lambda voice_id: (
            not fish_meta[voice_id].get("native"),
            -int(fish_meta[voice_id].get("popularity") or 0),
        )
    )
    fish_ids = curated_ids + other_ids

    preferred_ids: Sequence[str]
    if model:
        preferred_ids = voices_for_model(model, allow_live=allow_live, language=language)
    elif include_all_known:
        preferred_ids = list(ALL_OPENROUTER_VOICES) + fish_ids
    else:
        preferred_ids = list(ALL_OPENROUTER_VOICES)

    for voice_id in preferred_ids:
        if voice_id in seen:
            continue
        seen.add(voice_id)
        meta = OPENROUTER_VOICE_META.get(voice_id)
        if not meta:
            meta = fish_meta.get(voice_id, {})
        models = meta.get("models") or []
        model_hint = ""
        if models:
            short = []
            for m in models:
                slug = m.split("/")[-1]
                slug = slug.replace("qwen-audio-3.0-tts-", "").replace("grok-voice-tts-", "grok-")
                short.append(slug)
            model_hint = f" [{'/'.join(short)}]"
        label = meta.get("name") or voice_id
        description = meta.get("description") or ""
        is_fish = voice_id in fish_meta
        if is_fish:
            label = _fish_voice_label(meta)
            if not meta.get("native"):
                # The voice speaks the dubbing language with a foreign accent.
                label = f"{label} · чужой акцент"
        name = f"{label}{model_hint}"
        entry: Dict[str, Any] = {
            "id": voice_id,
            "name": name,
            "provider": "openrouter",
            "gender": meta.get("gender") or "unknown",
            "description": description,
        }
        if models:
            entry["models"] = list(models)
        languages = meta.get("languages") or []
        if languages:
            entry["language"] = ", ".join(str(code) for code in languages[:4])
        if is_fish:
            entry["accent"] = meta.get("accent") or ""
            entry["age"] = meta.get("age") or "unknown"
            entry["native"] = bool(meta.get("native"))
            entry["character"] = bool(meta.get("character"))
            entry["curated"] = bool(meta.get("curated"))
        preview_url = meta.get("preview_url")
        if preview_url:
            entry["preview_url"] = preview_url
        entries.append(entry)
    return entries


def list_openrouter_voices_for_gender(
    speaker_gender: Optional[str],
    *,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> List[str]:
    allowed = voices_for_model(model, language=language) if model else list(ALL_OPENROUTER_VOICES)
    gender = normalize_gender_value(speaker_gender)
    if gender == DEFAULT_SPEAKER_GENDER:
        return list(allowed)
    return [v for v in allowed if get_openrouter_voice_gender(v) == gender]
