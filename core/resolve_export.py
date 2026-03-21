"""Export video results for DaVinci Resolve. Starting with CSV."""

from __future__ import annotations

import csv
import os

from core.schemas import VideoResult


HEADERS = [
    "File Name", "Keywords", "Description", "Comments",
    "Shot Type", "Camera Angle", "Camera Movement",
    "Lighting", "Location", "Subject", "Roll Type",
    "Subject Name", "Interview", "Content Tags",
    "Multicam Group",
]


def _result_to_row(result: VideoResult) -> list[str]:
    """Convert a VideoResult into a CSV row matching HEADERS."""
    filename = os.path.basename(result.video_path)
    keywords_str = ", ".join(result.keywords)

    description = ""
    if result.transcript_summary:
        description = result.transcript_summary.summary

    comments = ""
    if result.transcript_summary:
        comments = (f"Title: {result.transcript_summary.title}. "
                    f"Topics: {', '.join(result.transcript_summary.topics)}")

    shot_type = ""
    camera_angle = ""
    camera_movement = ""
    lighting = ""
    location = ""
    subject = ""
    roll_type = ""
    if result.clip_classification:
        c = result.clip_classification
        shot_type = c.shot_type.title()
        camera_angle = c.camera_angle.title()
        camera_movement = c.camera_movement.title()
        lighting = c.lighting.title()
        location = c.location.title()
        subject = c.subject
        roll_type = c.roll_type.upper()

    subject_name = ""
    is_interview = ""
    refined_subject = ""
    content_tags = ""
    if result.clip_refinement:
        r = result.clip_refinement
        if r.subject_name.lower() != "unknown":
            subject_name = r.subject_name
            subject = r.refined_subject
        is_interview = "Yes" if r.is_interview else "No"
        refined_subject = r.refined_subject
        content_tags = ", ".join(r.content_tags)

    multicam_group = result.multicam_group_id or ""

    return [
        filename, keywords_str, description, comments,
        shot_type, camera_angle, camera_movement,
        lighting, location, subject, roll_type,
        subject_name, is_interview, content_tags,
        multicam_group,
    ]


def export_csv(result: VideoResult, output_path: str) -> str:
    """
    Generate a per-clip CSV file for DaVinci Resolve metadata import.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerow(_result_to_row(result))

    return output_path


def export_combined_csv(results: list[VideoResult], output_path: str) -> str:
    """
    Generate a single combined CSV with all clips in one file.
    One import in Resolve covers everything.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for result in results:
            writer.writerow(_result_to_row(result))

    return output_path
