from flask import Flask, request, jsonify
import requests
import io
import os
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        audio_url = data.get('audio_url')
        if not audio_url:
            return jsonify({"success": False, "error": "No audio URL provided"}), 400

        zoho_response = requests.get(audio_url, timeout=30)
        if zoho_response.status_code != 200:
            return jsonify({"success": False, "error": f"Failed to download audio: {zoho_response.status_code}"}), 400

        audio_data = zoho_response.content
        if len(audio_data) < 50:
            return jsonify({"success": False, "error": "Audio file too small"}), 400

        if audio_data.startswith((b'\xff\xfb', b'\xff\xf3', b'ID3')):
            mime_type = 'audio/mpeg'
            extension = 'mp3'
        elif b'ftyp' in audio_data[:20]:
            mime_type = 'audio/mp4'
            extension = 'm4a'
        elif audio_data.startswith(b'RIFF'):
            mime_type = 'audio/wav'
            extension = 'wav'
        else:
            mime_type = 'audio/mpeg'
            extension = 'mp3'

        file_tuple = (f'audio.{extension}', io.BytesIO(audio_data), mime_type)

        transcription_response = groq_client.audio.translations.create(
            file=file_tuple,
            model="whisper-large-v3",
            response_format="verbose_json"
        )

        full_text = ""
        if hasattr(transcription_response, "segments") and transcription_response.segments:
            segments = []
            for seg in transcription_response.segments:
                seg_dict = seg if isinstance(seg, dict) else seg.__dict__
                text = seg_dict.get("text", "").strip()
                if text:
                    segments.append(text)
            full_text = " ".join(segments)
        elif hasattr(transcription_response, "text"):
            full_text = transcription_response.text.strip()
        else:
            full_text = str(transcription_response)

        if not full_text:
            full_text = "(No speech detected)"

        return jsonify({"success": True, "transcription": full_text})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "transcription"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
