import logging
import os

import anthropic

from app.pipeline.reference_catalog import get_adavu_reference

logger = logging.getLogger(__name__)


def generate_coaching_feedback(
    frame_count: int,
    duration_ms: int,
    item_name: str | None = None,
    item_type: str | None = None,
    talam: str | None = None,
    pose_summary: dict | None = None,
    dancer_label: str | None = None,
) -> str:
    """Generate Bharatanatyam coaching feedback using Claude API.

    Args:
        frame_count: Number of frames analyzed.
        duration_ms: Video duration in milliseconds.
        item_name: Name of the dance item (e.g., "3rd Tattadavu").
        item_type: Type of dance item (e.g., "tattadavu", "nattadavu").
        talam: Rhythm pattern (e.g., "Aadhi Taalam").
        pose_summary: Aggregated pose statistics from frame analysis.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, returning placeholder feedback")
        return (
            "Coaching feedback unavailable (API key not configured). "
            "The pose skeleton overlay is available for visual review."
        )

    client = anthropic.Anthropic(api_key=api_key)

    duration_sec = duration_ms / 1000
    context_parts = [f"Duration: {duration_sec:.1f} seconds", f"Frames analyzed: {frame_count}"]
    if item_name:
        context_parts.append(f"Item: {item_name}")
    if item_type:
        context_parts.append(f"Type: {item_type}")
    if talam:
        context_parts.append(f"Talam: {talam}")

    if dancer_label:
        context_parts.append(f"Dancer: {dancer_label}")
    context_str = "\n".join(context_parts)

    # Build reference section from adavu catalog
    reference_text = get_adavu_reference(item_type, item_name)
    reference_section = ""
    if reference_text:
        reference_section = f"""

--- ADAVU REFERENCE DATA ---
The following is extracted from authoritative Bharatanatyam instructional texts.
Use this to provide technique-specific coaching rather than generic advice.

{reference_text}
--- END REFERENCE DATA ---
"""

    # Build pose statistics section if available
    pose_section = ""
    if pose_summary:
        pose_section = f"""

--- POSE SKELETON STATISTICS ---
The following measurements are aggregated from the dancer's pose skeleton across all frames:

{_format_pose_summary(pose_summary)}
--- END POSE STATISTICS ---
"""

    prompt = f"""You are an expert Bharatanatyam dance coach and guru. A student has uploaded a practice video
for AI-assisted form analysis. Here is what we know about the performance:

{context_str}
{reference_section}{pose_section}
Our system extracts a 17-point body pose skeleton per frame (COCO keypoints: nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles). We do NOT yet have hand gesture (mudra) detection or
facial expression (abhinaya) analysis -- those are planned for future stages.

Based on the pose data and reference material, provide coaching feedback. If reference data for a
specific adavu is provided above, tailor your feedback to that adavu's technique checkpoints.

Focus on:

1. **Aramandi**: Evaluate knee bend depth and consistency. If pose statistics are provided, reference
   the actual measured angles. For tattadavu, aramandi should be maintained throughout. For nattadavu,
   it should be maintained even during lateral leg extensions.

2. **Upper Body Posture**: Torso uprightness and shoulder alignment. For nattadavu, note shoulder
   coordination with arm movements and wrist turns.

3. **Arm Positioning**: For nattadavu, arms should be fully extended at shoulder height. For tattadavu,
   hands should be firmly on waist (natyarambhe). Comment on measured arm extension angles if available.

4. **Balance & Weight Distribution**: Centered balance, stability during transitions. Reference the
   specific adavu's weight transfer pattern if known.

5. **Symmetry**: Left-right symmetry is critical. If the adavu has both right-side and left-side
   sequences, note that both sides should mirror each other.

6. **Rhythm & Speed**: If beat pattern data is available from the reference, comment on how the dancer's
   timing aligns with the expected syllable pattern and speed progressions.

Keep the feedback encouraging, specific, and actionable. Write no more than 400 words.
Address the dancer respectfully as a guru would address a student.
When referencing technique points, cite the specific adavu and movement (e.g., "In the 3rd Tattadavu,
your aramandi depth on the Tam beats...")."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def _format_pose_summary(pose_summary: dict) -> str:
    """Format pose summary statistics for the LLM prompt."""
    lines = []

    if "avg_knee_angle" in pose_summary:
        lines.append(f"Average knee angle (aramandi depth): {pose_summary['avg_knee_angle']:.1f} degrees")
    if "min_knee_angle" in pose_summary:
        lines.append(f"Minimum knee angle: {pose_summary['min_knee_angle']:.1f} degrees")
    if "max_knee_angle" in pose_summary:
        lines.append(f"Maximum knee angle: {pose_summary['max_knee_angle']:.1f} degrees")
    if "knee_angle_std" in pose_summary:
        lines.append(f"Knee angle standard deviation: {pose_summary['knee_angle_std']:.1f} degrees (lower = more consistent)")
    if "avg_torso_angle" in pose_summary:
        lines.append(f"Average torso uprightness deviation: {pose_summary['avg_torso_angle']:.1f} degrees from vertical")
    if "avg_arm_extension_left" in pose_summary:
        lines.append(f"Average left arm extension: {pose_summary['avg_arm_extension_left']:.1f} degrees")
    if "avg_arm_extension_right" in pose_summary:
        lines.append(f"Average right arm extension: {pose_summary['avg_arm_extension_right']:.1f} degrees")
    if "hip_symmetry_avg" in pose_summary:
        lines.append(f"Average hip symmetry deviation: {pose_summary['hip_symmetry_avg']:.1f} degrees (0 = perfect symmetry)")
    if "balance_score" in pose_summary:
        lines.append(f"Overall balance score: {pose_summary['balance_score']:.2f} (0-1, higher is better)")
    if "avg_foot_turnout" in pose_summary:
        lines.append(f"Average foot turnout: {pose_summary['avg_foot_turnout']:.1f} degrees from vertical (ideal for aramandi: 45-60)")
    if "avg_foot_turnout_left" in pose_summary:
        lines.append(f"  Left foot turnout: {pose_summary['avg_foot_turnout_left']:.1f} degrees")
    if "avg_foot_turnout_right" in pose_summary:
        lines.append(f"  Right foot turnout: {pose_summary['avg_foot_turnout_right']:.1f} degrees")
    if "avg_foot_flatness" in pose_summary:
        lines.append(f"Average foot flatness: {pose_summary['avg_foot_flatness']:.4f} (lower = flatter strike, important for tattadavu)")

    return "\n".join(lines) if lines else "No aggregated pose statistics available yet."
