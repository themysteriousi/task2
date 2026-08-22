"""
Download ai4bharat/MSMARCO-XI dataset for selected Indic languages.
Files are saved to: ./data/
"""

import os
from huggingface_hub import hf_hub_download

LANG_FILES = {
    "hi":  "train/hintrain.parquet",   # Hindi
    "pa":  "train/pantrain.parquet",   # Punjabi
    "gu":  "train/gutrain.parquet",    # Gujarati
    "mr":  "train/martrain.parquet",   # Marathi
}

LANG_NAMES = {"hi": "Hindi", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi"}
REPO_ID = "ai4bharat/MSMARCO-XI"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def download_all():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n📁 Saving dataset files to: {SAVE_DIR}\n")

    for lang, filename in LANG_FILES.items():
        print(f"⬇️  Downloading [{lang}] {LANG_NAMES[lang]} → {filename} ...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=SAVE_DIR,
            )
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"   ✅ Saved: {local_path}  ({size_mb:.1f} MB)\n")
        except Exception as e:
            print(f"   ❌ Failed [{lang}]: {e}\n")

    print("🎉 Download complete!")
    print(f"📂 All files in: {SAVE_DIR}")

if __name__ == "__main__":
    download_all()
