#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tavus video service implementation for avatar-based video generation.

This module implements Tavus as a sink transport layer, providing video
avatar functionality through Tavus's streaming API.
"""

import asyncio
from dataclasses import dataclass

import aiohttp
from daily.daily import AudioData, VideoFrame
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.transports.tavus.transport import (
    AVATAR_VAD_STOP_SECS,
    TavusCallbacks,
    TavusParams,
    TavusTransportClient,
)


@dataclass
class TavusVideoSettings(ServiceSettings):
    """Settings for the Tavus video service."""

    pass


class TavusVideoService(AIService):
    """Service that proxies audio to Tavus and receives audio and video in return.

    Uses the TavusTransportClient to manage sessions and handle communication.
    When audio is sent, Tavus responds with both audio and video streams, which
    are routed through Pipecat's media pipeline.

    In use cases with DailyTransport, this creates two distinct virtual rooms:

    - Tavus room: Contains the Tavus Avatar and the Pipecat Bot
    - User room: Contains the Pipecat Bot and the user
    """

    Settings = TavusVideoSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",
        session: aiohttp.ClientSession,
        settings: Settings | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Tavus video service.

        Args:
            api_key: Tavus API key used for authentication.
            replica_id: ID of the Tavus voice replica to use for speech synthesis.
            persona_id: ID of the Tavus persona. Defaults to "pipecat-stream" for Pipecat TTS voice.
            session: Async HTTP session used for communication with Tavus.
            settings: Runtime-updatable settings. Tavus has no model concept, so this
                is primarily used for the ``extra`` dict.
            **kwargs: Additional arguments passed to the parent AIService class.
        """
        default_settings = ServiceSettings(model=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._api_key = api_key
        self._session = session
        self._replica_id = replica_id
        self._persona_id = persona_id

        self._other_participant_has_joined = False
        self._client: TavusTransportClient | None = None

        self._resampler = create_stream_resampler()
        self._send_task: asyncio.Task | None = None
        self._transport_ready = False

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Tavus video service.

        Args:
            setup: Frame processor setup configuration.
        """
        await super().setup(setup)
        callbacks = TavusCallbacks(
            on_joined=self._on_joined,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=self._api_key,
            replica_id=self._replica_id,
            persona_id=self._persona_id,
            session=self._session,
            params=TavusParams(
                audio_in_enabled=True,
                video_in_enabled=True,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up the service and release resources."""
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

    async def _on_joined(self, data):
        """Handle bot joined the Daily room."""
        logger.info("Tavus bot joined Daily room")

    async def _on_participant_left(self, participant, reason):
        """Handle participant leaving the session."""
        participant_id = participant["id"]
        logger.info(f"Participant left {participant_id}, reason: {reason}")

    async def _on_participant_joined(self, participant):
        """Handle participant joining the session."""
        participant_id = participant["id"]
        logger.info(f"Participant joined {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame, 30
            )
            await self._client.capture_participant_audio(
                participant_id=participant_id,
                callback=self._on_participant_audio_data,
                sample_rate=self._client.out_sample_rate,
            )

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        """Handle incoming video frames from participants."""
        frame = OutputImageRawFrame(
            image=video_frame.buffer,
            size=(video_frame.width, video_frame.height),
            format=video_frame.color_format,
        )
        frame.transport_source = video_source
        if self._transport_ready:
            await self.push_frame(frame)

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle incoming audio data from participants."""
        frame = SpeechOutputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        if self._transport_ready:
            await self.push_frame(frame)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Tavus service supports metrics generation.
        """
        return True

    async def get_persona_name(self) -> str:
        """Get the name of the current persona.

        Returns:
            The persona name from the Tavus client.
        """
        return await self._client.get_persona_name()

    async def start(self, frame: StartFrame):
        """Start the Tavus video service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._client.start(frame)
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the Tavus video service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Tavus video service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through the service.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
        elif isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, BotStartedSpeakingFrame):
            # We constantly receive audio through WebRTC, but most of the time it is silence.
            # As soon as we receive actual audio, the base output transport will create a
            # BotStartedSpeakingFrame, which we can use as a signal for the TTFB metrics.
            await self.stop_ttfb_metrics()
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self):
        """Handle interruption events by resetting send tasks and notifying client."""
        await self._cancel_send_task()
        await self._create_send_task()
        await self._client.send_interrupt_message()

    async def _end_conversation(self):
        """End the current conversation and reset state."""
        await self._client.stop()
        self._other_participant_has_joined = False

    async def _create_send_task(self):
        """Create the audio sending task if it doesn't exist."""
        if not self._send_task:
            self._queue = asyncio.Queue()
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        """Cancel the audio sending task if it exists."""
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _handle_audio_frame(self, frame: OutputAudioRawFrame):
        """Queue an audio frame for sending."""
        await self._queue.put(frame)

    async def _send_task_handler(self):
        """Accumulate audio into chunks and send via conversation.echo.

        Tracks inference_id from the first TTSAudioRawFrame of each utterance.
        Accumulates resampled audio until 400ms is reached, then sends with done=False.
        On AVATAR_VAD_STOP_SECS timeout, flushes any remainder and sends a done=True
        silence to signal end of utterance.
        """
        sample_rate = self._client.out_sample_rate
        audio_chunk_bytes = int(sample_rate * 2 * 0.4)  # 400ms, 16-bit mono
        done_silence = bytes(int(sample_rate * 2 / 20))  # 50ms silence for done signal
        audio_buffer = bytearray()
        inference_id: str | None = None
        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=AVATAR_VAD_STOP_SECS)
                if inference_id is None:
                    inference_id = str(frame.id)
                audio = await self._resampler.resample(
                    frame.audio, frame.sample_rate, sample_rate
                )
                audio_buffer.extend(audio)
                while len(audio_buffer) >= audio_chunk_bytes:
                    send_chunk = bytes(audio_buffer[:audio_chunk_bytes])
                    del audio_buffer[:audio_chunk_bytes]
                    await self._client.encode_audio_and_send(send_chunk, False, inference_id)
                self._queue.task_done()
            except TimeoutError:
                if not inference_id:
                    # nothing to do here, we're waiting for the first audio frame
                    continue
                if audio_buffer:
                    await self._client.encode_audio_and_send(
                        bytes(audio_buffer), False, inference_id
                    )
                    audio_buffer.clear()
                await self._client.encode_audio_and_send(done_silence, True, inference_id)
                inference_id = None
