# Generate videos with Google Veo

The video studio builds a detailed storyboard from the user's description and
any supplied Jira evidence. Veo generates full-screen animated footage for
each scene; when a diagram helps, its boxes and arrows are described to Veo as
part of that scene rather than drawn as a separate panel. The app adds
Text-to-Speech narration and synchronized subtitles, then assembles the MP4.
The total duration follows the storyboard rather than a fixed limit.

1. Install the project dependencies with `pip install -r requirements.txt`.
2. Set `GOOGLE_API_KEY` for Gemini storyboard generation and `SA_FILE` to a
   Google service-account JSON file. Set `PROJECT_ID` to your Google Cloud
   project and `LOCATION` to a supported Vertex AI region. `MODEL` defaults
   to `veo-3.1-lite-generate-001`; `TTS_VOICE_NAME` optionally selects the
   narration voice. Relative `SA_FILE` paths are resolved from the project
   root. On Windows, use a relative path or forward slashes in absolute paths
   to avoid `.env` interpreting backslashes as escape sequences.
3. Ensure the service account has permission to use Vertex AI and Google Cloud
   Text-to-Speech, and that both APIs are enabled for the project.

Veo generates an initial 4-, 6-, or 8-second clip per scene at 1080p. Longer
narration may require additional 7-second Veo extension requests, so generation
time and cloud usage increase with the number and duration of scenes.

Keep `.env` and service-account credentials out of source control.
