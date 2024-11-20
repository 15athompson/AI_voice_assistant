# AI Voice Assistant

An advanced voice assistant that uses speech recognition and OpenAI's GPT model to provide intelligent responses to your voice commands.

## Features

- Wake word detection ("assistant")
- Natural language processing using GPT-3.5
- Voice command system
- Multiple voice options
- Volume control
- Conversation history management
- Ambient noise adjustment
- Error handling and recovery
- System status monitoring

## Voice Commands

- "assistant" - Wake word to activate the assistant
- "sleep" - Put the assistant to sleep (waiting for wake word)
- "volume up" - Increase speech volume
- "volume down" - Decrease speech volume
- "clear history" - Clear conversation history
- "change voice" - Toggle between available voices
- "status" - Get system status information
- "goodbye" or "bye" - Exit the assistant

## Requirements

- Python 3.8 or higher
- A microphone for voice input
- Speakers for voice output
- OpenAI API key

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the voice assistant:
   ```
   python voice_assistant.py
   ```
2. Say the wake word "assistant" to activate
3. Use voice commands or ask questions
4. Say "sleep" to put the assistant in standby mode
5. Say "goodbye" or "bye" to exit

## Features in Detail

### Wake Word Detection
The assistant listens for the wake word "assistant" before processing any commands. This helps conserve resources and provides better privacy control.

### Voice Commands
Built-in commands for system control:
- Volume adjustment
- Voice changing
- History management
- System status checking

### Conversation Management
- Maintains conversation history for context
- Limits context window to last 5 exchanges for better performance
- Option to clear history

### Error Handling
- Robust error recovery for speech recognition
- Network error handling
- Timeout management
- Clear error messages

## Troubleshooting

- If you encounter issues with PyAudio installation on Windows:
  1. Download the appropriate PyAudio wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
  2. Install it using pip: `pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl`

- If the assistant can't hear you:
  1. Check your microphone settings
  2. Make sure your microphone is set as the default input device
  3. Try speaking louder or moving closer to the microphone
  4. Check if the assistant is in sleep mode (waiting for wake word)

## Note

This voice assistant requires an active internet connection for:
- Speech recognition (Google Speech Recognition API)
- AI responses (OpenAI API)
