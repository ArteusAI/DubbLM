#!/usr/bin/env python3
"""Example of using the DubbLM API."""

import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"


def main():
    """Demonstrate DubbLM API usage."""
    
    # 1. Create a new project
    print("1. Creating project...")
    response = requests.post(f"{BASE_URL}/projects", json={"name": "My Dubbing Project"})
    response.raise_for_status()
    project = response.json()
    project_id = project["id"]
    print(f"   Created project: {project_id}")
    
    # 2. Configure the project
    print("\n2. Configuring project...")
    config = {
        "sourceLang": "en",
        "targetLang": "ru",
        "personaId": "normal",
        "keepBackground": True,
    }
    response = requests.patch(f"{BASE_URL}/projects/{project_id}/config", json=config)
    response.raise_for_status()
    print("   Configuration updated")
    
    # 3. Upload video (example path - replace with actual video)
    video_path = Path("example_video.mp4")
    if video_path.exists():
        print("\n3. Uploading video...")
        with open(video_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/upload",
                files={"file": (video_path.name, f, "video/mp4")}
            )
        response.raise_for_status()
        upload_info = response.json()
        print(f"   Uploaded: {upload_info['filename']} ({upload_info['size']} bytes)")
    else:
        print("\n3. Skipping video upload (no example_video.mp4 found)")
        print("   To test fully, place a video file named 'example_video.mp4' in this directory")
        return
    
    # 4. Start transcription
    print("\n4. Starting transcription...")
    response = requests.post(f"{BASE_URL}/projects/{project_id}/process/transcribe")
    response.raise_for_status()
    job = response.json()
    print(f"   Job started: {job['jobId']}")
    
    # 5. Monitor progress via SSE (simplified polling for example)
    print("\n5. Monitoring progress...")
    while True:
        response = requests.get(f"{BASE_URL}/projects/{project_id}/jobs/{job['jobId']}/status")
        response.raise_for_status()
        status = response.json()
        print(f"   Progress: {status['progress']}% - {status['currentStep']}")
        
        if status["status"] in ["completed", "failed"]:
            print(f"   Final status: {status['status']}")
            break
        
        time.sleep(5)
    
    # 6. Get segments
    print("\n6. Getting segments...")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/segments")
    response.raise_for_status()
    segments = response.json()
    print(f"   Found {len(segments)} segments")
    for seg in segments[:3]:  # Show first 3
        print(f"   - {seg['speaker']}: {seg['translatedText'][:50]}...")
    
    # 7. Edit a segment
    if segments:
        print("\n7. Editing segment...")
        seg_id = segments[0]["id"]
        response = requests.patch(
            f"{BASE_URL}/projects/{project_id}/segments/{seg_id}",
            json={"translatedText": "Custom translation text"}
        )
        response.raise_for_status()
        print("   Segment updated")
    
    # 8. Start dubbing
    print("\n8. Starting dubbing...")
    response = requests.post(f"{BASE_URL}/projects/{project_id}/process/dub")
    response.raise_for_status()
    dub_job = response.json()
    print(f"   Job started: {dub_job['jobId']}")
    
    # Wait for completion (simplified)
    while True:
        response = requests.get(f"{BASE_URL}/projects/{project_id}/jobs/{dub_job['jobId']}/status")
        response.raise_for_status()
        status = response.json()
        print(f"   Progress: {status['progress']}% - {status['currentStep']}")
        
        if status["status"] in ["completed", "failed"]:
            print(f"   Final status: {status['status']}")
            break
        
        time.sleep(10)
    
    # 9. Download result
    if status["status"] == "completed":
        print("\n9. Downloading result...")
        response = requests.get(f"{BASE_URL}/projects/{project_id}/download/video")
        if response.status_code == 200:
            output_path = Path(f"dubbed_{project_id}.mp4")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"   Saved to: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

