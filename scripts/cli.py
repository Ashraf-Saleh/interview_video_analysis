"""
CLI to analyze a video -> JSON.
"""
from __future__ import annotations
import argparse, json
from core.config import Settings
from core.pipeline import analyze_video_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out", default="output/analysis.json", help="Path to output JSON")
    args = p.parse_args()

    settings = Settings()
    result = analyze_video_pipeline(args.video, settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Also write to file
    import os
    os.makedirs("output", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Analysis written to {args.out}")

if __name__ == "__main__":
    main()
