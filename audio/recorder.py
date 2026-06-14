import sounddevice as sd
import soundfile as sf
import tempfile
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LiveVoiceRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recording = []
        self.temp_file_path = None
        self.stream = None

    def start_recording(self):
        """Finds the best microphone device and starts a non-blocking stream."""
        logger.info("Initializing mic stream...")
        self.recording = []
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Stream status warning: {status}")
            self.recording.append(indata.copy())

        try:
            # Look for default input device hardware configuration
            device_info = sd.query_devices(kind='input')
            logger.info(f"Using recording device: {device_info['name']}")
            
            # Start the input stream safely
            self.stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                callback=callback
            )
            self.stream.start()
        except Exception as e:
            logger.exception("Failed to connect to microphone device hardware")
            raise RuntimeError(f"Microphone configuration failure. Please ensure a mic is plugged in! Details: {str(e)}")

    def stop_recording(self) -> str:
        """Stops recording, verifies data integrity, and saves to disk."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        
        if not self.recording or len(self.recording) == 0:
            logger.error("No raw audio data buffer captured during stream runtime.")
            return ""
            
        # Concat all chunk frames into a continuous sound wave matrix
        audio_data = np.concatenate(self.recording, axis=0)
        
        # Check if the file is completely silent / empty
        if np.max(np.abs(audio_data)) < 0.0001:
            logger.error("Audio buffer caught, but contains absolute zero volume (silence).")
            return ""
            
        # Generate a temporary file path safely
        temp_dir = tempfile.gettempdir()
        self.temp_file_path = os.path.join(temp_dir, "live_speech_input.wav")
        
        # Save buffer to WAV disk format
        sf.write(self.temp_file_path, audio_data, self.sample_rate)
        logger.info(f"Saved sound wave data to: {self.temp_file_path}")
        return self.temp_file_path