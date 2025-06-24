# Directory structure:
# livekit-plugins/livekit-plugins-kokoro/

# File: livekit-plugins/livekit-plugins-kokoro/pyproject.toml
"""
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "livekit-plugins-kokoro"
version = "0.1.0"
description = "Kokoro TTS plugin for LiveKit Agents"
readme = "README.md"
license = { text = "Apache-2.0" }
authors = [
    { name = "LiveKit Community", email = "community@livekit.io" },
]
maintainers = [
    { name = "LiveKit Community", email = "community@livekit.io" },
]
keywords = ["livekit", "agents", "tts", "kokoro", "voice synthesis", "realtime"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10", 
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Communications",
    "Framework :: AsyncIO",
]
requires-python = ">=3.9"
dependencies = [
    "livekit-agents>=1.0.0",
    "aiohttp>=3.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0",
    "pytest-timeout>=2.2.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "types-aiofiles",
    "aioresponses>=0.7.6",
]

[project.urls]
Homepage = "https://livekit.io"
Documentation = "https://docs.livekit.io/agents/plugins/kokoro"
Repository = "https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-kokoro"
Issues = "https://github.com/livekit/agents/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["livekit*"]

[tool.setuptools.package-data]
"*" = ["py.typed"]

[tool.ruff]
line-length = 88
target-version = "py39"
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501", # line too long (handled by black)
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.mypy]
python_version = "3.9"
strict = true
ignore_missing_imports = true
exclude = ["build", "dist"]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --asyncio-mode=auto"
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
timeout = 30
"""

# File: livekit-plugins/livekit-plugins-kokoro/livekit/plugins/kokoro/__init__.py
"""
# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .models import AudioFormat, TTSModels, TTSVoices
from .tts import TTS

__all__ = [
    "TTS",
    "TTSModels",
    "TTSVoices", 
    "AudioFormat",
]

__version__ = "0.1.0"
"""

# File: livekit-plugins/livekit-plugins-kokoro/livekit/plugins/kokoro/models.py
"""
# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Literal


class TTSModels(str, Enum):
    \"\"\"Available Kokoro TTS models.\"\"\"
    KOKORO = "kokoro"


class TTSVoices(str, Enum):
    \"\"\"Available Kokoro TTS voices.
    
    Attributes:
        AF: Adult Female (default voice)
        AM: Adult Male
        BF: British Female
        BM: British Male
        AF_BELLA: Adult Female - Bella variant
        AF_SARAH: Adult Female - Sarah variant
        AM_ADAM: Adult Male - Adam variant
        AM_MICHAEL: Adult Male - Michael variant
    \"\"\"
    AF = "af"
    AM = "am"
    BF = "bf"
    BM = "bm"
    AF_BELLA = "af_bella"
    AF_SARAH = "af_sarah"
    AM_ADAM = "am_adam"
    AM_MICHAEL = "am_michael"


# Type literal for voice options
TTSVoiceOptions = Literal[
    "af", "am", "bf", "bm", 
    "af_bella", "af_sarah", "am_adam", "am_michael"
]


class AudioFormat(str, Enum):
    \"\"\"Supported audio output formats.
    
    Attributes:
        MP3: MPEG Layer 3 audio (default)
        WAV: Waveform Audio File Format
        OGG: Ogg Vorbis audio
        AAC: Advanced Audio Coding
        FLAC: Free Lossless Audio Codec
    \"\"\"
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"
    FLAC = "flac"


# Type literal for format options
AudioFormatOptions = Literal["mp3", "wav", "ogg", "aac", "flac"]


# Default values
DEFAULT_MODEL = TTSModels.KOKORO
DEFAULT_VOICE = TTSVoices.AF
DEFAULT_FORMAT = AudioFormat.MP3
DEFAULT_LANGUAGE = "en"
DEFAULT_CACHE_DURATION = 24
"""

# File: livekit-plugins/livekit-plugins-kokoro/livekit/plugins/kokoro/tts.py
"""
# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Union

import aiohttp
from livekit import agents
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, tts

from . import _utils
from .models import (
    DEFAULT_CACHE_DURATION,
    DEFAULT_FORMAT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    AudioFormat,
    AudioFormatOptions,
    TTSModels,
    TTSVoiceOptions,
    TTSVoices,
)

logger = logging.getLogger(__name__)


@dataclass
class _TTSOptions:
    \"\"\"Internal TTS options configuration.\"\"\"
    model: Union[TTSModels, str]
    voice: Union[TTSVoices, str]
    format: Union[AudioFormat, str]
    speed: float
    language: str
    cache_duration_hours: int


class TTS(tts.TTS):
    \"\"\"
    Kokoro TTS plugin for LiveKit Agents.
    
    Provides high-quality text-to-speech synthesis using the Kokoro TTS API
    with support for 8 different voices and multiple audio formats.
    
    Example:
        ```python
        from livekit.plugins import kokoro
        
        tts = kokoro.TTS(
            voice="af_bella",
            format="mp3",
        )
        
        async def example():
            stream = tts.synthesize("Hello, world!")
            async for chunk in stream:
                # Process audio chunks
                pass
        ```
    
    Args:
        model: TTS model to use (default: "kokoro")
        voice: Voice to use for synthesis (default: "af")
        format: Audio output format (default: "mp3")
        speed: Speech speed multiplier (0.5-2.0, default: 1.0)
        language: Language code (default: "en")
        cache_duration_hours: How long to cache results (default: 24)
        base_url: Custom API base URL (optional)
        http_session: Custom aiohttp session (optional)
    \"\"\"

    def __init__(
        self,
        *,
        model: Union[TTSModels, str] = DEFAULT_MODEL,
        voice: Union[TTSVoices, TTSVoiceOptions, str] = DEFAULT_VOICE,
        format: Union[AudioFormat, AudioFormatOptions, str] = DEFAULT_FORMAT,
        speed: float = 1.0,
        language: str = DEFAULT_LANGUAGE,
        cache_duration_hours: int = DEFAULT_CACHE_DURATION,
        base_url: Optional[str] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            )
        )
        
        # Validate inputs
        voice = _utils.validate_voice(voice)
        format = _utils.validate_format(format)
        speed = _utils.validate_speed(speed)
        
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            format=format,
            speed=speed,
            language=language,
            cache_duration_hours=cache_duration_hours,
        )
        
        self._base_url = base_url or _utils.DEFAULT_BASE_URL
        self._session = http_session
        self._owned_session = http_session is None
        
        logger.info(
            f"Initialized Kokoro TTS with voice={voice}, "
            f"format={format}, speed={speed}"
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        \"\"\"Ensure HTTP session exists.\"\"\"
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
            )
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._session

    def update_options(
        self,
        *,
        model: Optional[Union[TTSModels, str]] = None,
        voice: Optional[Union[TTSVoices, TTSVoiceOptions, str]] = None,
        format: Optional[Union[AudioFormat, AudioFormatOptions, str]] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
        cache_duration_hours: Optional[int] = None,
    ) -> None:
        \"\"\"
        Update TTS options dynamically.
        
        Args:
            model: New model to use
            voice: New voice to use
            format: New audio format
            speed: New speech speed (0.5-2.0)
            language: New language code
            cache_duration_hours: New cache duration
        \"\"\"
        if model is not None:
            self._opts.model = model
        if voice is not None:
            self._opts.voice = _utils.validate_voice(voice)
        if format is not None:
            self._opts.format = _utils.validate_format(format)
        if speed is not None:
            self._opts.speed = _utils.validate_speed(speed)
        if language is not None:
            self._opts.language = language
        if cache_duration_hours is not None:
            self._opts.cache_duration_hours = cache_duration_hours
            
        logger.debug(f"Updated TTS options: {self._opts}")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = APIConnectOptions(
            max_retry=3,
            retry_interval=0.5,
            timeout=30.0,
        ),
    ) -> "ChunkedStream":
        \"\"\"
        Synthesize text to speech.
        
        Args:
            text: Text to convert to speech (1-4000 characters)
            conn_options: API connection options
            
        Returns:
            ChunkedStream: Audio stream that can be iterated
            
        Raises:
            ValueError: If text is empty or too long
            APIConnectionError: If connection fails
            APIStatusError: If API returns an error
        \"\"\"
        # Validate input
        if not text:
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        if len(text) > 4000:
            raise ValueError(
                f"Text too long: {len(text)} characters (max 4000)"
            )
        
        # Log synthesis request
        logger.debug(
            f"Synthesizing {len(text)} characters with voice={self._opts.voice}"
        )
        
        # Create and return stream
        return ChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        \"\"\"Close the TTS instance and cleanup resources.\"\"\"
        if self._owned_session and self._session:
            await self._session.close()
            self._session = None


class ChunkedStream(tts.ChunkedStream):
    \"\"\"
    Chunked audio stream implementation for Kokoro TTS.
    
    Simulates streaming by breaking the complete audio into chunks,
    providing low-latency playback for real-time applications.
    \"\"\"

    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=text,
            conn_options=conn_options,
        )
        self._text = text
        self._opts = tts._opts
        self._base_url = tts._base_url
        self._conn_options = conn_options
        
    async def _run(self) -> None:
        \"\"\"Main task to fetch and stream audio.\"\"\"
        try:
            request_id = self._event_ch.send_nowait(
                tts.SynthesizeEvent(request_id="", text=self._text)
            )
            
            session = await self._tts._ensure_session()
            
            # Fetch audio from API
            audio_data = await self._fetch_audio(session, request_id)
            
            # Determine audio format parameters
            sample_rate, num_channels = _utils.get_audio_format_params(
                self._opts.format
            )
            
            # Stream audio in chunks
            chunk_size = _utils.calculate_chunk_size(sample_rate)
            total_size = len(audio_data)
            offset = 0
            
            while offset < total_size:
                chunk_end = min(offset + chunk_size, total_size)
                chunk = audio_data[offset:chunk_end]
                
                # Create audio frame
                frame = agents.audio.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                )
                
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        frame=frame,
                    )
                )
                
                offset = chunk_end
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in TTS streaming: {str(e)}")
            raise

    async def _fetch_audio(
        self,
        session: aiohttp.ClientSession,
        request_id: str,
    ) -> bytes:
        \"\"\"Fetch audio from Kokoro TTS API.\"\"\"
        url = f"{self._base_url}/api/v1/tts/synthesize"
        
        payload = {
            "text": self._text,
            "voice": str(self._opts.voice),
            "provider": "kokoro",
            "format": str(self._opts.format),
            "speed": str(self._opts.speed),
            "language": self._opts.language,
            "cache_duration_hours": self._opts.cache_duration_hours,
        }

        headers = _utils.get_default_headers()

        retry_count = 0
        last_error = None

        while retry_count <= self._conn_options.max_retry:
            try:
                timeout = aiohttp.ClientTimeout(total=self._conn_options.timeout)
                
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        return await self._handle_success_response(
                            response, request_id
                        )
                    else:
                        await self._handle_error_response(response)
                        
            except asyncio.TimeoutError:
                last_error = APIConnectionError("Request timed out")
                logger.warning(f"Attempt {retry_count + 1} timed out")
                
            except aiohttp.ClientError as e:
                last_error = APIConnectionError(f"Connection error: {str(e)}")
                logger.warning(f"Attempt {retry_count + 1} failed: {str(e)}")
                
            except APIStatusError:
                raise
                
            except Exception as e:
                last_error = APIConnectionError(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error during TTS synthesis: {str(e)}")

            # Retry logic
            if retry_count < self._conn_options.max_retry:
                retry_count += 1
                await asyncio.sleep(
                    self._conn_options.retry_interval * retry_count
                )
            else:
                break

        # All retries exhausted
        raise last_error or APIConnectionError("Failed to synthesize speech")

    async def _handle_success_response(
        self,
        response: aiohttp.ClientResponse,
        request_id: str,
    ) -> bytes:
        \"\"\"Handle successful API response.\"\"\"
        data = await response.json()
        
        # Validate response structure
        if "audio_data" not in data:
            raise APIStatusError(
                "Invalid response: missing audio_data field"
            )
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(data["audio_data"])
            
            # Log metrics if available
            if "processing_time_ms" in data:
                logger.debug(
                    f"TTS synthesis completed in {data['processing_time_ms']}ms"
                )
            
            # Send metrics event
            self._event_ch.send_nowait(
                tts.SynthesizeMetrics(
                    request_id=request_id,
                    ttfb=data.get("processing_time_ms", 0) / 1000.0,
                    duration=data.get("duration_seconds", 0),
                    characters=data.get("text_length", len(self._text)),
                    cached=data.get("cached", False),
                )
            )
            
            return audio_bytes
            
        except Exception as e:
            raise APIStatusError(
                f"Failed to decode audio data: {str(e)}"
            )

    async def _handle_error_response(
        self,
        response: aiohttp.ClientResponse,
    ) -> None:
        \"\"\"Handle error API response.\"\"\"
        if response.status == 422:
            error_data = await response.json()
            error_detail = error_data.get("detail", "Validation error")
            raise APIStatusError(
                f"Validation error: {error_detail}",
                status_code=response.status,
            )
        
        elif response.status == 429:
            # Rate limiting
            retry_after = response.headers.get("Retry-After", "60")
            raise APIStatusError(
                f"Rate limited. Retry after {retry_after} seconds",
                status_code=response.status,
            )
        
        else:
            text = await response.text()
            raise APIStatusError(
                f"API request failed with status {response.status}: {text}",
                status_code=response.status,
            )
"""

# File: livekit-plugins/livekit-plugins-kokoro/livekit/plugins/kokoro/_utils.py
"""
# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Internal utilities for Kokoro TTS plugin.\"\"\"

import logging
from typing import Dict, Tuple, Union

from .models import AudioFormat, TTSVoices

logger = logging.getLogger(__name__)

# Default API configuration
DEFAULT_BASE_URL = "https://e02aa7eb-6259-40e0-b631-a57539e1cae0-00-t4kog4anjjdq.kirk.replit.dev"
DEFAULT_API_VERSION = "v1"

# Audio format configurations
AUDIO_FORMAT_PARAMS: Dict[str, Tuple[int, int]] = {
    AudioFormat.MP3: (24000, 1),     # 24kHz, mono
    AudioFormat.WAV: (48000, 1),     # 48kHz, mono
    AudioFormat.OGG: (24000, 1),     # 24kHz, mono
    AudioFormat.AAC: (44100, 1),     # 44.1kHz, mono
    AudioFormat.FLAC: (48000, 1),    # 48kHz, mono
}


def validate_voice(voice: Union[TTSVoices, str]) -> TTSVoices:
    \"\"\"Validate and convert voice input to TTSVoices enum.\"\"\"
    if isinstance(voice, TTSVoices):
        return voice
        
    if isinstance(voice, str):
        # Try to convert string to enum
        try:
            return TTSVoices(voice)
        except ValueError:
            # Check if it's a valid voice value
            valid_voices = [v.value for v in TTSVoices]
            if voice in valid_voices:
                return TTSVoices(voice)
            
            logger.warning(
                f"Unknown voice '{voice}', using default. "
                f"Valid voices: {', '.join(valid_voices)}"
            )
            return TTSVoices.AF
            
    logger.warning(f"Invalid voice type: {type(voice)}, using default")
    return TTSVoices.AF


def validate_format(format_input: Union[AudioFormat, str]) -> AudioFormat:
    \"\"\"Validate and convert format input to AudioFormat enum.\"\"\"
    if isinstance(format_input, AudioFormat):
        return format_input
        
    if isinstance(format_input, str):
        # Try to convert string to enum
        try:
            return AudioFormat(format_input)
        except ValueError:
            # Check if it's a valid format value
            valid_formats = [f.value for f in AudioFormat]
            if format_input in valid_formats:
                return AudioFormat(format_input)
            
            logger.warning(
                f"Unknown format '{format_input}', using default. "
                f"Valid formats: {', '.join(valid_formats)}"
            )
            return AudioFormat.MP3
            
    logger.warning(f"Invalid format type: {type(format_input)}, using default")
    return AudioFormat.MP3


def validate_speed(speed: float) -> float:
    \"\"\"Validate speech speed is within acceptable range.\"\"\"
    if not isinstance(speed, (int, float)):
        logger.warning(f"Invalid speed type: {type(speed)}, using default")
        return 1.0
        
    # Clamp speed to valid range (0.5 - 2.0)
    if speed < 0.5:
        logger.warning(f"Speed {speed} too slow, using minimum 0.5")
        return 0.5
    elif speed > 2.0:
        logger.warning(f"Speed {speed} too fast, using maximum 2.0")
        return 2.0
        
    return float(speed)


def get_audio_format_params(format_enum: Union[AudioFormat, str]) -> Tuple[int, int]:
    \"\"\"Get sample rate and number of channels for audio format.\"\"\"
    format_str = str(format_enum)
    return AUDIO_FORMAT_PARAMS.get(format_str, (24000, 1))


def calculate_chunk_size(sample_rate: int, chunk_duration_ms: int = 20) -> int:
    \"\"\"Calculate chunk size in bytes for streaming.
    
    Args:
        sample_rate: Audio sample rate in Hz
        chunk_duration_ms: Desired chunk duration in milliseconds
        
    Returns:
        Chunk size in bytes
    \"\"\"
    # Calculate samples per chunk
    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
    
    # Assuming 16-bit audio (2 bytes per sample)
    bytes_per_sample = 2
    
    return samples_per_chunk * bytes_per_sample


def get_default_headers() -> Dict[str, str]:
    \"\"\"Get default HTTP headers for API requests.\"\"\"
    return {
        "Content-Type": "application/json",
        "User-Agent": "livekit-plugins-kokoro/0.1.0",
        "Accept": "application/json",
    }
"""

# File: livekit-plugins/livekit-plugins-kokoro/livekit/plugins/kokoro/py.typed
"""
# Marker file for PEP 561
# This package provides type information
"""

# File: livekit-plugins/livekit-plugins-kokoro/README.md
"""
# LiveKit Kokoro TTS Plugin

Plugin for [LiveKit Agents](https://github.com/livekit/agents) that integrates Kokoro TTS, providing high-quality text-to-speech synthesis with 8 unique voices and multiple audio formats.

## Features

- 🎙️ **8 High-Quality Voices**: Diverse selection including male/female and regional variants
- 🎵 **Multiple Audio Formats**: MP3, WAV, OGG, AAC, and FLAC support
- ⚡ **Streaming Support**: Low-latency chunked audio streaming
- 💾 **Intelligent Caching**: Built-in caching for improved performance
- 🛡️ **Production Ready**: Comprehensive error handling and retry logic
- 🆓 **Zero Cost**: No external API charges

## Installation

```bash
pip install livekit-plugins-kokoro
```

## Quick Start

```python
from livekit import agents
from livekit.plugins import kokoro

async def main():
    # Initialize TTS
    tts = kokoro.TTS(
        voice="af_bella",
        format="mp3",
    )
    
    # Use in AgentSession
    session = agents.AgentSession(
        stt=your_stt,
        llm=your_llm,
        tts=tts,
    )
```

## Configuration

### Available Voices

| Voice ID | Description |
|----------|-------------|
| `af` | Adult Female (default) |
| `am` | Adult Male |
| `bf` | British Female |
| `bm` | British Male |
| `af_bella` | Adult Female - Bella |
| `af_sarah` | Adult Female - Sarah |
| `am_adam` | Adult Male - Adam |
| `am_michael` | Adult Male - Michael |

### Audio Formats

- `mp3` - MPEG Layer 3 (default)
- `wav` - Waveform Audio
- `ogg` - Ogg Vorbis
- `aac` - Advanced Audio Coding
- `flac` - Free Lossless Audio Codec

### Speech Speed

Speed can be adjusted from 0.5 (half speed) to 2.0 (double speed):

```python
tts = kokoro.TTS(speed=1.25)  # 25% faster
```

## Advanced Usage

### Custom Configuration

```python
tts = kokoro.TTS(
    voice="am_michael",
    format="wav",
    speed=0.9,
    language="en",
    cache_duration_hours=48,
    base_url="https://your-kokoro-instance.com"
)
```

### Dynamic Options Update

```python
# Update voice mid-session
tts.update_options(voice="af_sarah")

# Update multiple options
tts.update_options(
    voice="am_adam",
    speed=1.5,
    format="flac"
)
```

### Error Handling

```python
from livekit.agents import APIConnectionError, APIStatus