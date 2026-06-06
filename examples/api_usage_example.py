#!/usr/bin/env python3
"""Example of using the DubbLM API."""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"


def external_translate(video_path: Path, target_lang: str = "ru", preset: str = "hq") -> None:
    """One-shot translate: upload video, start full pipeline, poll, download."""
    print("External translate (one-shot)...")
    with open(video_path, "rb") as video_file:
        response = requests.post(
            f"{BASE_URL}/translate",
            files={"file": (video_path.name, video_file, "video/mp4")},
            data={
                "targetLang": target_lang,
                "preset": preset,
                "sourceLang": "auto",
                "minimalDiarizationMerge": "true",
            },
        )
    response.raise_for_status()
    result = response.json()
    print(f"   Project: {result['projectId']}")
    print(f"   Job: {result['jobId']}")
    print(f"   Poll: {BASE_URL}{result['pollUrl']}")

    poll_url = f"{BASE_URL}{result['pollUrl']}"
    while True:
        status_response = requests.get(poll_url)
        status_response.raise_for_status()
        status = status_response.json()
        print(f"   Progress: {status['progress']}% - {status['currentStep']}")

        if status["status"] in ["completed", "failed"]:
            print(f"   Final status: {status['status']}")
            break

        time.sleep(10)

    if status["status"] == "completed":
        download_url = f"{BASE_URL}{result['downloadUrl']}"
        download_response = requests.get(download_url)
        if download_response.status_code == 200:
            output_path = Path(f"dubbed_{result['projectId']}.mp4")
            with open(output_path, "wb") as output_file:
                output_file.write(download_response.content)
            print(f"   Saved to: {output_path}")


def main():
    """Demonstrate DubbLM API usage."""
    video_path = Path("example_video.mp4")
    if not video_path.exists():
        print("Place a video file named 'example_video.mp4' in this directory to run examples.")
        return

    print("=== One-shot external translate ===")
    external_translate(video_path, target_lang="ru", preset="hq")

    print("\n=== Step-by-step project workflow ===")

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

    # 3. Upload video
    print("\n3. Uploading video...")
    with open(video_path, "rb") as video_file:
        response = requests.post(
            f"{BASE_URL}/projects/{project_id}/upload",
            files={"file": (video_path.name, video_file, "video/mp4")},
        )
    response.raise_for_status()
    upload_info = response.json()
    print(f"   Uploaded: {upload_info['filename']} ({upload_info['size']} bytes)")

    # 4. Start transcription
    print("\n4. Starting transcription...")
    response = requests.post(f"{BASE_URL}/projects/{project_id}/process/transcribe")
    response.raise_for_status()
    job = response.json()
    print(f"   Job started: {job['jobId']}")

    # 5. Monitor progress
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
    for seg in segments[:3]:
        print(f"   - {seg['speaker']}: {seg['translatedText'][:50]}...")

    # 7. Edit a segment
    if segments:
        print("\n7. Editing segment...")
        seg_id = segments[0]["id"]
        response = requests.patch(
            f"{BASE_URL}/projects/{project_id}/segments/{seg_id}",
            json={"translatedText": "Custom translation text"},
        )
        response.raise_for_status()
        print("   Segment updated")

    # 8. Start dubbing
    print("\n8. Starting dubbing...")
    response = requests.post(f"{BASE_URL}/projects/{project_id}/process/dub")
    response.raise_for_status()
    dub_job = response.json()
    print(f"   Job started: {dub_job['jobId']}")

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
            with open(output_path, "wb") as output_file:
                output_file.write(response.content)
            print(f"   Saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
