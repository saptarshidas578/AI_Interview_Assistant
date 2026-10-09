# AI Interview Assistant - System Architecture

## Multimodal Processing Pipeline

```
  +------------------+         +-------------------------+
  |  Webcam Feed     | ------> | OpenCV + MediaPipe      | ---> Eye Contact %
  |  (30 FPS)        |         | Face Mesh 468 Landmarks |      Attention Score
  +------------------+         +-------------------------+

  +------------------+         +-------------------------+
  |  Microphone      | ------> | sounddevice Stream      | ---> RMS Amplitude
  |  Input           |         | (16 kHz, 16-bit WAV)    |      WAV File Export
  +------------------+         +-------------------------+

  +------------------+         +-------------------------+
  |  Whisper STT     | ------> | Groq LLaMA 3.1 Model    | ---> STAR Rubric
  |  Transcription   |         | Fast Inference Engine   |      Next Question
  +------------------+         +-------------------------+
```

## Rubric Dimensions
1. **Relevance (0-10):** Alignment with question context.
2. **Clarity (0-10):** Articulation and structure.
3. **STAR Method (0-10):** Situation, Task, Action, Result coverage.
4. **Technical Depth (0-10):** Domain accuracy and nuance.
5. **Pacing:** Estimated words per minute (WPM).
