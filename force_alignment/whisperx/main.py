import whisperx
import json
from pathlib import Path
import logging

logging.basicConfig(
        filename="./errors.log",
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

ERROR_TOKEN = ['。', '？', '！', '，', '、', '；', '：']

class WhisperXTranscriber:
    def __init__(self, device="cuda", compute_type="float16", batch_size=16):
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.model = None
        self.audio = None

    def load_model(self, model_name="large-v3", model_dir=None):
        kwargs = {"compute_type": self.compute_type}
        if model_dir:
            kwargs["download_root"] = model_dir
        self.model = whisperx.load_model(model_name, self.device, vad_method="silero", **kwargs)

    def load_audio(self, audio_file):
        self.audio = whisperx.load_audio(str(audio_file))

    def transcribe(self):
        if self.model is None or self.audio is None:
            raise RuntimeError("Model and audio must be loaded before transcription.")
        return self.model.transcribe(self.audio, batch_size=self.batch_size)

    def align(self, segments, language_code):
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device
        )
        return whisperx.align(
            segments, model_a, metadata, self.audio,
            self.device, return_char_alignments=False
        )

    @staticmethod
    def save_to_jsonl(segments:list, audio_path:str, output_path:str):
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                words = segment["words"]
                if words[-1]['word'] in ERROR_TOKEN:
                    words = words[:-1]
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_path},
                            {"type": "text", "text": "transcribe it"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text_stream", "text_stream": [[word["start"], word["end"], word["word"]] for word in words]}
                        ]
                    }
                ]
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        print(f"  Saved {len(segments)} segments to {output_path}")

    def process_file(self, audio_file, output_dir):
        audio_file = Path(audio_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = audio_file.stem

        after_path = output_dir / f"{stem}_after.jsonl"

        print(f"  Loading audio: {audio_file.name}")
        self.load_audio(audio_file)

        print(f"  Transcribing...")
        result = self.transcribe()

        print(f"  Aligning...")
        aligned_result = self.align(result["segments"], result["language"])
        self.save_to_jsonl(aligned_result["segments"], str(audio_file), after_path)

    def run_directory(self, input_dir, output_dir, audio_extensions=None, model_name="large-v3"):
        if audio_extensions is None:
            audio_extensions = {".mp3", ".mp4", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".mkv"}

        input_dir = Path(input_dir)
        audio_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return

        print(f"Found {len(audio_files)} audio file(s). Loading model...")
        self.load_model(model_name)

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            try:
                self.process_file(audio_file, output_dir)
            except Exception as e:
                print(f"  ERROR processing {audio_file.name}: {e}")
                logging.error(
                                " forced alignment error | index=[%d] | audio path=[%s] | error=[%s] |",
                                i, audio_file, 
                                str(e)
                            )

        print("\nDone.")
        
    def run(self, input_path, output_dir, audio_extensions=None, model_name="large-v3"):
        if audio_extensions is None:
            audio_extensions = {".mp3", ".mp4", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".mkv"}
        audio_file = Path(input_path)
        self.load_model(model_name)
        try:
            self.process_file(audio_file, output_dir)
        except Exception as e:
            print(f"  ERROR processing {audio_file.name}: {e}")
            logging.error(
                            " forced alignment error | audio path=[%s] | error=[%s] |",
                            audio_file, 
                            str(e)
                        )
        


if __name__ == "__main__":
    input_dir = "/n/work1/muyun/Dataset/zoom2025/audios/B_all"
    output_dir = "./transcripts"

    transcriber = WhisperXTranscriber(device="cuda", compute_type="float16", batch_size=16)
    # transcriber.run_directory(
    #     input_dir=input_dir,
    #     output_dir=output_dir,
    #     model_name="large-v3"
    # )
    transcriber.run("/n/work1/muyun/Dataset/zoom2025/audios/B_all/606_b.wav", output_dir=output_dir, model_name="large-v3")