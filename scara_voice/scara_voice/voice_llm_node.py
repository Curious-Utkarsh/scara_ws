#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import whisper
import sounddevice as sd
import os
from scipy.io.wavfile import write
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play


class VoiceLLMNode(Node):
    def __init__(self):
        super().__init__("voice_llm_node")
        self.publisher = self.create_publisher(String, "/scara/pick_command", 10)

        keys = self.read_api_keys()
        gemini_key = keys.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        eleven_key = keys.get("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY"))
        self.voice_id = keys.get(
            "ELEVENLABS_VOICE_ID",
            os.getenv("ELEVENLABS_VOICE_ID", "flq6f7yk4E4fJM5XTYuZ"),
        )

        if not gemini_key or not eleven_key:
            raise RuntimeError(
                "Add GEMINI_API_KEY and ELEVENLABS_API_KEY to scara_voice/api.txt"
            )

        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")

        genai.configure(api_key=gemini_key)
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")
        self.eleven_client = ElevenLabs(api_key=eleven_key)
        self.get_logger().info("Ready. Say 'hey scara'.")

    def read_api_keys(self):
        """Read KEY=value lines from scara_voice/api.txt if it exists."""
        api_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api.txt")
        keys = {}
        if not os.path.exists(api_path):
            return keys

        with open(api_path, "r", encoding="utf-8") as api_file:
            for line in api_file:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    keys[key.strip()] = value.strip()
        return keys

    def get_voice_input(self, seconds=4):
        sample_rate = 16000
        wav_path = "/tmp/scara_voice_input.wav"
        self.get_logger().info("Listening...")
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        write(wav_path, sample_rate, audio)

        text = self.model.transcribe(wav_path)["text"].strip()
        self.get_logger().info("Heard: %s" % text)
        return text.lower()

    def get_color_command(self, text):
        prompt = (
            "Reply with only G, B, R, or UNKNOWN. "
            "G means pick green box, B means pick blue box, and R means pick red box. "
            "Reply UNKNOWN unless the user asks to pick one of these boxes. "
            "User said: " + text
        )
        answer = self.gemini.generate_content(prompt).text.strip().upper()
        if answer in ("G", "B", "R"):
            return answer
        return None

    def speak(self, text):
        self.get_logger().info("SCARA: %s" % text)
        audio = self.eleven_client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        play(audio)

    def run(self):
        while rclpy.ok():
            # First say only: "hey scara".
            wake_text = self.get_voice_input()
            if "hey scara" not in wake_text and "hey_scara" not in wake_text:
                continue

            self.speak("Yes, what would you like me to pick up?")
            command_text = self.get_voice_input(seconds=6)
            command = self.get_color_command(command_text)

            if command is None:
                self.speak("Please say pick green box, blue box, or red box.")
                continue

            msg = String()
            msg.data = command
            self.publisher.publish(msg)
            self.speak("Okay, picking up the box.")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceLLMNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
