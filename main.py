import torch
import os
import sys
import soundfile as sf
import gc
import re
import platform
import time
from datetime import datetime

# Configuration
MAX_CHAR_LIMIT = 600
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
REPO_PATH = os.path.join(os.getcwd(), "Qwen3-TTS")
THERMAL_COOLDOWN_SECONDS = 0.3
ENABLE_TORCH_COMPILE = False

if os.path.exists(REPO_PATH):
    sys.path.append(REPO_PATH)

# Model Definitions
MODELS = {
    # 1.7B Models (High Quality)
    "1": {
        "name": "CustomVoice (Pro)",
        "path": os.path.abspath("./models/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "folder": "CustomVoice_Pro",
        "mini": False
    },
    "2": {
        "name": "VoiceDesign (Pro)",
        "path": os.path.abspath("./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        "folder": "VoiceDesign_Pro",
        "mini": False
    },
    "3": {
        "name": "VoiceClone (Pro)",
        "path": os.path.abspath("./models/Qwen3-TTS-12Hz-1.7B-Base"),
        "folder": "VoiceClone_Pro",
        "mini": False
    },
    # 0.6B Models (Fast / Low Resource)
    "4": {
        "name": "CustomVoice (Mini)",
        "path": os.path.abspath("./models/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        "folder": "CustomVoice_Mini",
        "mini": True
    },
    "5": {
        "name": "VoiceClone (Mini)",
        "path": os.path.abspath("./models/Qwen3-TTS-12Hz-0.6B-Base"),
        "folder": "VoiceClone_Mini",
        "mini": True
    }
}

SPEAKER_MAP = {
    "English": ["Ryan", "Aiden"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"]
}

# Try to import the library
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("[Error] Could not load Qwen library. Please ensure dependencies are installed.")
    sys.exit(1)


def clean_memory(model_ref):
    print("\n[System] Cleaning memory...")
    if model_ref is not None:
        try:
            model_ref.to("cpu")
        except:
            pass
        del model_ref

    for _ in range(3):
        gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    print("[System] Memory freed.")


def save_audio_file(wav, sr, mode_folder, text_snippet):
    save_path = os.path.join(BASE_OUTPUT_DIR, mode_folder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = re.sub(r'[^\w\s-]', '', text_snippet)[:20].strip().replace(' ', '_')
    filename = f"{timestamp}_{clean_text}.wav"
    full_path = os.path.join(save_path, filename)

    sf.write(full_path, wav[0], sr)
    print(f"[IO] Saved to: outputs/{mode_folder}/{filename}")


def estimate_max_tokens(text):
    word_count = len(text.split())
    return max(128, min(word_count * 8, 2048))


def warmup_model(model, speaker="Vivian"):
    print("[System] Warming up model...")
    try:
        with torch.inference_mode():
            _ = model.generate_custom_voice(
                text="Hello.",
                language="English",
                speaker=speaker,
                max_new_tokens=64
            )
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        print("[System] Warmup complete.")
    except Exception:
        print("[System] Warmup skipped.")


def load_model_safe(model_info):
    full_path = model_info["path"]
    print(f"\n[Loader] Loading {model_info['name']}...")

    if not os.path.exists(full_path):
        print(f"[Error] Model folder not found at: {full_path}")
        return None

    use_dtype = torch.float16
    try:
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16).to("mps")
        use_dtype = torch.bfloat16
        print("[System] M3/M4 detected. Using bfloat16.")
    except:
        print("[System] Using float16 compatibility mode.")

    try:
        model = Qwen3TTSModel.from_pretrained(
            full_path,
            dtype=use_dtype,
            attn_implementation="sdpa",
            device_map="mps",
            low_cpu_mem_usage=True,
        )
        print("[Loader] Model loaded.")
        if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                model.model = torch.compile(model.model, mode="reduce-overhead")
                print("[System] torch.compile enabled.")
            except Exception as e:
                print(f"[System] torch.compile not available: {e}")
        
        return model
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None


def get_safe_input():
    while True:
        text = input("\nEnter text (or 'exit'): ")

        if text.lower() in ['exit', 'quit']:
            return None

        if len(text) > MAX_CHAR_LIMIT:
            print(f"[Warning] Input too long ({len(text)} chars). Limit is {MAX_CHAR_LIMIT}.")
            continue

        return text


def run_custom_session(key):
    info = MODELS[key]
    model = load_model_safe(info)
    if not model: return

    print(f"\n--- {info['name']} Mode ---")

    # Speaker Selection
    print("\nAvailable Speakers:")
    for lang, names in SPEAKER_MAP.items():
        print(f"  [{lang}]: {', '.join(names)}")

    speaker = "Vivian"
    while True:
        user_input = input("\nSelect Speaker: ").strip()
        if not user_input:
            break
        
        found = False
        for lang, names in SPEAKER_MAP.items():
            for name in names:
                if name.lower() == user_input.lower():
                    speaker = name
                    found = True
        
        if found:
            print(f"[Selection] Speaker set to: {speaker}")
            break
        print(f"[Error] Speaker '{user_input}' not found.")

    # Speed Selection
    speed_instruct = ""
    print("\nSelect Speed:")
    print("  1. Normal")
    print("  2. Fast")
    print("  3. Slow")
    speed_choice = input("Choice (1-3): ").strip()

    if speed_choice == "2":
        speed_instruct = "Speak at a fast pace."
        print("[Config] Speed set to FAST")
    elif speed_choice == "3":
        speed_instruct = "Speak slowly and clearly."
        print("[Config] Speed set to SLOW")

    try:
        while True:
            text = get_safe_input()
            if text is None: break

            print("[Status] Generating...")
            max_tokens = estimate_max_tokens(text)
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="English",
                speaker=speaker,
                instruct=speed_instruct,
                max_new_tokens=max_tokens
            )
            save_audio_file(wavs, sr, info["folder"], text)
            
            # Brief cooldown to reduce thermal load
            if THERMAL_COOLDOWN_SECONDS > 0:
                time.sleep(THERMAL_COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        clean_memory(model)


def run_design_session(key):
    info = MODELS[key]
    model = load_model_safe(info)
    if not model: return

    print(f"\n--- {info['name']} Mode ---")
    instruct = input("Describe the voice prompt: ")

    try:
        while True:
            text = get_safe_input()
            if text is None: break

            print("[Status] Designing and Generating...")
            max_tokens = estimate_max_tokens(text)
            wavs, sr = model.generate_voice_design(
                text=text,
                language="English",
                instruct=instruct,
                max_new_tokens=max_tokens
            )
            save_audio_file(wavs, sr, info["folder"], text)
            
            # Brief cooldown to reduce thermal load
            if THERMAL_COOLDOWN_SECONDS > 0:
                time.sleep(THERMAL_COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        clean_memory(model)


def run_clone_session(key):
    info = MODELS[key]
    model = load_model_safe(info)
    if not model:
        input("Press Enter to return...")
        return

    print(f"\n--- {info['name']} Mode ---")
    ref_audio = input("Reference audio path: ").strip().strip("'").strip('"')

    if not os.path.exists(ref_audio):
        print("[Error] File not found.")
        clean_memory(model)
        input("Press Enter to return...")
        return

    voice_prompt = None
    if info["mini"]:
        print("[Config] Using Fast Mode (Mini Model)")
        try:
            voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, x_vector_only_mode=True)
        except Exception as e:
            print(f"[Error] Analysis failed: {e}")
            clean_memory(model)
            return
    else:
        print("1. Fast Mode (No transcript needed)")
        print("2. High Quality (Requires transcript)")
        c = input("Choice (1/2): ")
        
        try:
            if c == "1":
                voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, x_vector_only_mode=True)
            else:
                ref_text = input("Transcript: ")
                voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
        except Exception as e:
            print(f"[Error] Analysis failed: {e}")
            clean_memory(model)
            return

    try:
        while True:
            text = get_safe_input()
            if text is None: break

            print("[Status] Cloning...")
            max_tokens = estimate_max_tokens(text)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=voice_prompt,
                max_new_tokens=max_tokens
            )
            save_audio_file(wavs, sr, info["folder"], text)
            
            # Brief cooldown to reduce thermal load
            if THERMAL_COOLDOWN_SECONDS > 0:
                time.sleep(THERMAL_COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        input("Press Enter to continue...")
    finally:
        clean_memory(model)


def main_menu():
    print("\n" + "="*40)
    print(" Qwen3-TTS Studio (Mac Optimized)")
    print("="*40)
    print(" High Quality (1.7B)")
    print("  1. Custom Voice")
    print("  2. Voice Design")
    print("  3. Voice Clone")
    print("\n Fast / Low Res (0.6B)")
    print("  4. Custom Voice")
    print("  5. Voice Clone")
    print("\n  6. Exit")

    choice = input("\nSelect Option: ")

    if choice == "1": run_custom_session("1")
    elif choice == "2": run_design_session("2")
    elif choice == "3": run_clone_session("3")
    elif choice == "4": run_custom_session("4")
    elif choice == "5": run_clone_session("5")
    elif choice == "6": sys.exit()
    else: print("Invalid selection.")


if __name__ == "__main__":
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        while True:
            main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
