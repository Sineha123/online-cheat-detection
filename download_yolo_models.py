"""
YOLOv4-tiny Model Downloader
Run this script ONCE before starting the exam system.

Usage:
    python download_yolo_models.py

Files downloaded to:  models/
    - yolov4-tiny.weights   (~23 MB)
    - yolov4-tiny.cfg       (~2 KB)
    - coco.names            (~1 KB)
"""

import os
import sys
import ssl
import urllib.request

# ── Fix SSL certificate verification failure on Windows ──────────────────────
# This is a common issue on Windows where Python can't verify GitHub's cert.
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# Monkey-patch urllib to use our SSL context globally
_orig_urlopen = urllib.request.urlopen
def _patched_urlopen(url, *args, **kwargs):
    if isinstance(url, str):
        return _orig_urlopen(url, *args, context=ssl_ctx, **kwargs)
    return _orig_urlopen(url, *args, **kwargs)
urllib.request.urlopen = _patched_urlopen

MODELS_DIR = 'models'
FILES = {
    'yolov4-tiny.weights': (
        'https://github.com/AlexeyAB/darknet/releases/download/'
        'darknet_yolo_v4_pre/yolov4-tiny.weights',
        22801840  # ~23 MB expected size
    ),
    'yolov4-tiny.cfg': (
        'https://raw.githubusercontent.com/AlexeyAB/darknet/master/'
        'cfg/yolov4-tiny.cfg',
        None
    ),
    'coco.names': (
        'https://raw.githubusercontent.com/pjreddie/darknet/master/'
        'data/coco.names',
        None
    ),
}


def download_with_progress(url, dest_path, expected_size=None):
    """Download file with a simple progress bar."""
    print(f"\n  Downloading: {os.path.basename(dest_path)}")
    print(f"  URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded / total_size * 100)
            bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
            sys.stdout.write(f"\r  [{bar}] {pct:.0f}%  ")
            sys.stdout.flush()

    try:
        # Use ssl context to bypass Windows certificate verification issues
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_ctx) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192
            downloaded = 0
            block_num = 0
            with open(dest_path, 'wb') as out_file:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    out_file.write(block)
                    downloaded += len(block)
                    block_num += 1
                    _progress(block_num, block_size, total_size)
        print(f"\n  ✅ Saved → {dest_path}")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        # Clean up partial file
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == 0:
            os.remove(dest_path)
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  YOLOv4-tiny Model Downloader")
    print("  These models enable REAL phone/book detection")
    print("=" * 60)

    all_ok = True
    for filename, (url, expected_size) in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            fsize = os.path.getsize(dest)
            # If weights file is too small it's probably corrupt
            if expected_size and fsize < expected_size * 0.5:
                print(f"\n  ⚠️  {filename} exists but looks corrupt ({fsize} bytes). Re-downloading...")
                ok = download_with_progress(url, dest, expected_size)
            else:
                print(f"\n  ✅ {filename} already exists ({fsize:,} bytes) — skip")
                ok = True
        else:
            ok = download_with_progress(url, dest, expected_size)
        
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ All model files ready!")
        print("  You can now start the exam system.")
        print()
        print("  Verify by running:")
        print("    python -c \"import cv2; net=cv2.dnn.readNetFromDarknet('models/yolov4-tiny.cfg','models/yolov4-tiny.weights'); print('YOLOv4-tiny loaded OK')\"")
    else:
        print("  ❌ Some downloads failed. Try manual download:")
        print()
        print("  mkdir -p models")
        print("  # Weights (~23 MB):")
        print("  wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -P models/")
        print("  # Config:")
        print("  wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P models/")
        print("  # Class names:")
        print("  wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P models/")
    print("=" * 60)


if __name__ == '__main__':
    main()