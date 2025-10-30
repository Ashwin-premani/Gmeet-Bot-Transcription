import os
import whisper
from pyannote.audio import Pipeline
import torch
from datetime import timedelta
import warnings
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

AUDIO_FILE_PATH = r"C:\Users\ashwi\Desktop\transcript\audio\meeting3.wav"
REPO_PATH = r"C:\Users\ashwi\Desktop\transcript\repo\intern-match-ai-main"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
VERBOSE_LOGGING = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

try:
    print("Loading models...")
    whisper_model = whisper.load_model("base", device=DEVICE)
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    ).to(torch.device(DEVICE))
    print("Models loaded successfully.\n")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

if OPENROUTER_API_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
else:
    client = None

def transcribe_and_diarize(audio_path):
    print("Starting transcription and diarization...")
    diarization = diarization_pipeline(audio_path)
    result = whisper_model.transcribe(audio_path, language="en")
    transcript_segments = []
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        speaker = "SPEAKER_UNKNOWN"
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if turn.start <= start_time <= turn.end or turn.start <= end_time <= turn.end:
                speaker = speaker_label
                break
        transcript_segments.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": text
        })
    return transcript_segments

def format_transcript(segments):
    formatted = []
    current_speaker = None
    current_text = []
    for seg in segments:
        if seg["speaker"] != current_speaker:
            if current_text:
                formatted.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = seg["speaker"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])
    if current_text:
        formatted.append(f"{current_speaker}: {' '.join(current_text)}")
    return "\n\n".join(formatted)

def get_repo_structure(repo_path, max_depth=3):
    structure = []
    repo_path = Path(repo_path)
    ignore_dirs = {'.git', 'node_modules', '__pycache__', '.next', 'dist', 'build', 'venv'}
    def scan_dir(path, depth=0, prefix=""):
        if depth > max_depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items:
                if item.name in ignore_dirs or item.name.startswith('.'):
                    continue
                if item.is_dir():
                    structure.append(f"{prefix}📁 {item.name}/")
                    scan_dir(item, depth + 1, prefix + "  ")
                else:
                    structure.append(f"{prefix}📄 {item.name}")
        except PermissionError:
            pass
    scan_dir(repo_path)
    return "\n".join(structure)

def get_key_files_content(repo_path):
    repo_path = Path(repo_path)
    key_files = {}
    patterns = ['**/routes.ts', '**/routes.js', '**/package.json', '**/tsconfig.json']
    for pattern in patterns:
        for file in repo_path.rglob(pattern):
            try:
                if 'node_modules' not in str(file):
                    relative_path = file.relative_to(repo_path)
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) < 10000:
                            key_files[str(relative_path)] = content
            except Exception:
                pass
    return key_files

def analyze_transcription_with_llm(transcript_text, repo_structure, key_files):
    if not client:
        print("Warning: OPENROUTER_API_KEY not set, using mock analysis")
        return mock_analyze_transcription()
    key_files_text = "\n\n".join([
        f"File: {path}\n```\n{content}\n```" 
        for path, content in key_files.items()
    ])
    analysis_prompt = f"""You are analyzing a meeting transcript to extract development tasks. 

    MEETING TRANSCRIPT:
    {transcript_text}

    REPOSITORY STRUCTURE:
    {repo_structure}

    KEY EXISTING FILES:
    {key_files_text}

    Extract all development tasks discussed. For each task, provide:
    1. task_type: FEATURE, BUGFIX, or REFACTOR
    2. description: Clear summary of what needs to be done
    3. details: Specific implementation details mentioned
    4. files_mentioned: Files that need to be created or modified
    5. acceptance_criteria: How to verify the task is complete

    Return your analysis as a JSON object with a "tasks" array."""
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": "You are an expert software development assistant."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return mock_analyze_transcription()

def mock_analyze_transcription():
    return { "tasks": [{"task_type": "FEATURE", "description": "Mock Task: Create a new endpoint."}] }

def generate_code_with_llm(structured_tasks, repo_structure, key_files):
    if not client:
        print("Warning: OPENROUTER_API_KEY not set, using mock code generation")
        return mock_generate_code()
    key_files_text = "\n\n".join([
        f"File: {path}\n```\n{content}\n```" 
        for path, content in key_files.items()
    ])
    tasks_text = json.dumps(structured_tasks, indent=2)
    coder_prompt = f"""You are a senior software engineer. Implement the following tasks.

    TASKS TO IMPLEMENT:
    {tasks_text}

    REPOSITORY STRUCTURE:
    {repo_structure}

    EXISTING KEY FILES:
    {key_files_text}

    Generate the complete code for all new and modified files.
    Format your response EXACTLY like this for each file:
    ```language:path/to/file.ext
    // Complete file content here
    ```"""
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free", 
            # nvidia/nemotron-nano-9b-v2:free
            # google/gemma-3n-e2b-it:free
            # deepseek/deepseek-r1-0528-qwen3-8b:free
            messages=[{"role": "user", "content": coder_prompt}],
            max_tokens=8192
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating code: {e}")
        return mock_generate_code()

def mock_generate_code():
    return "```typescript:server/mock.ts\n// Mock code - set OPENROUTER_API_KEY for real output.\n```"

def process_audio_to_code():
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Error: File not found at '{AUDIO_FILE_PATH}'.")
        return
    if not os.path.exists(REPO_PATH):
        print(f"Error: Repository not found at '{REPO_PATH}'.")
        return
    print(f"{'='*60}\nProcessing: {os.path.basename(AUDIO_FILE_PATH)}\n{'='*60}\n")
    try:
        print("STEP 1: TRANSCRIPTION & DIARIZATION\n" + "-"*60)
        segments = transcribe_and_diarize(AUDIO_FILE_PATH)
        transcript = format_transcript(segments)
        print(f"\n📝 TRANSCRIPT:\n{'='*60}\n{transcript}\n{'='*60}")
        print("\n\nSTEP 2: ANALYZING REPOSITORY\n" + "-"*60)
        repo_structure = get_repo_structure(REPO_PATH)
        key_files = get_key_files_content(REPO_PATH)
        print(f"Found {len(key_files)} key files in repository")
        print("\n\nSTEP 3: ANALYZING TASKS FROM TRANSCRIPT\n" + "-"*60)
        structured_tasks = analyze_transcription_with_llm(transcript, repo_structure, key_files)
        print(f"\n📋 EXTRACTED TASKS:\n{'='*60}\n{json.dumps(structured_tasks, indent=2)}\n{'='*60}")
        print("\n\nSTEP 4: GENERATING CODE\n" + "-"*60)
        generated_code = generate_code_with_llm(structured_tasks, repo_structure, key_files)
        print(f"\n\n✅ GENERATED CODE FILES\n{'='*60}\n{generated_code}\n{'='*60}")

        
        print("\n\n✅ Processing Complete!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_audio_to_code()