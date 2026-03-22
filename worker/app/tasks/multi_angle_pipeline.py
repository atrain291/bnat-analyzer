"""Multi-angle pipeline: audio sync, wait for individual pipelines, fuse scores.

This task is dispatched after dancer linking. It:
1. Computes audio sync offsets between videos
2. Polls until all individual performance pipelines complete
3. Fuses scores across views for each linked dancer
4. Generates multi-angle coaching feedback via LLM
"""

import logging
import time
import traceback

from app.celery_app import app
from app.db import get_session
from app.models.performance import (
    Performance, MultiAngleGroup, MultiAngleAnalysis,
    PerformanceDancer, Analysis,
)
from app.pipeline.audio_sync import compute_group_sync_offsets
from app.pipeline.score_fusion import fuse_scores
from app.pipeline.llm import generate_coaching_feedback

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 5
MAX_WAIT_SEC = 3600  # 1 hour max wait


@app.task(bind=True, name="worker.app.tasks.multi_angle_pipeline.run_multi_angle_pipeline")
def run_multi_angle_pipeline(self, group_id: int):
    logger.info(f"Starting multi-angle pipeline for group {group_id}")

    try:
        # Step 1: Audio sync
        with get_session() as session:
            group = session.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
            if not group:
                raise ValueError(f"MultiAngleGroup {group_id} not found")

            performances = session.query(Performance).filter(
                Performance.multi_angle_group_id == group_id
            ).all()

            video_paths = {}
            perf_info = {}
            for p in performances:
                video_paths[p.id] = f"/app/uploads/{p.video_key}"
                perf_info[p.id] = {
                    "camera_label": p.camera_label,
                    "item_name": p.item_name,
                    "item_type": p.item_type,
                    "talam": p.talam,
                }

        logger.info(f"Computing audio sync for {len(video_paths)} videos")
        sync_result = compute_group_sync_offsets(video_paths)

        with get_session() as session:
            group = session.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
            if group:
                group.sync_offsets = {str(k): v for k, v in sync_result["offsets"].items()}
                group.sync_confidence = sync_result["confidence"]

        # Step 2: Wait for all individual pipelines to complete
        logger.info("Waiting for individual pipelines to complete...")
        waited = 0
        while waited < MAX_WAIT_SEC:
            with get_session() as session:
                performances = session.query(Performance).filter(
                    Performance.multi_angle_group_id == group_id
                ).all()

                statuses = {p.id: p.status for p in performances}
                all_complete = all(s == "complete" for s in statuses.values())
                any_failed = any(s == "failed" for s in statuses.values())

            if any_failed:
                logger.warning(f"Some pipelines failed: {statuses}")
                # Continue with whatever completed
                break

            if all_complete:
                logger.info("All individual pipelines complete")
                break

            time.sleep(POLL_INTERVAL_SEC)
            waited += POLL_INTERVAL_SEC

        if waited >= MAX_WAIT_SEC:
            logger.error(f"Timed out waiting for pipelines in group {group_id}")

        # Step 3: Gather per-view analyses and fuse scores per dancer
        with get_session() as session:
            # Get all dancer labels linked in this group
            perf_ids = [p.id for p in session.query(Performance).filter(
                Performance.multi_angle_group_id == group_id
            ).all()]

            # Get all PerformanceDancers across views
            pds = session.query(PerformanceDancer).filter(
                PerformanceDancer.performance_id.in_(perf_ids)
            ).all()

            # Group by dancer label
            dancers_by_label = {}
            for pd in pds:
                label = pd.label or f"dancer_{pd.track_id}"
                if label not in dancers_by_label:
                    dancers_by_label[label] = []
                dancers_by_label[label].append(pd)

        # For each dancer, collect analyses from all views and fuse
        for label, pd_list in dancers_by_label.items():
            per_view = []

            with get_session() as session:
                for pd in pd_list:
                    analysis = session.query(Analysis).filter(
                        Analysis.performance_dancer_id == pd.id
                    ).first()

                    if not analysis:
                        continue

                    perf = session.query(Performance).filter(
                        Performance.id == pd.performance_id
                    ).first()

                    per_view.append({
                        "performance_id": pd.performance_id,
                        "camera_label": perf.camera_label if perf else None,
                        "aramandi_score": analysis.aramandi_score,
                        "upper_body_score": analysis.upper_body_score,
                        "symmetry_score": analysis.symmetry_score,
                        "rhythm_consistency_score": analysis.rhythm_consistency_score,
                        "overall_score": analysis.overall_score,
                        "technique_scores": analysis.technique_scores,
                        "llm_summary": analysis.llm_summary,
                    })

            if not per_view:
                continue

            fused = fuse_scores(per_view)

            # Generate multi-angle coaching summary
            coaching_text = _generate_multi_angle_coaching(
                label, per_view, fused, perf_info
            )

            with get_session() as session:
                ma_analysis = MultiAngleAnalysis(
                    multi_angle_group_id=group_id,
                    dancer_label=label,
                    aramandi_score=fused.get("aramandi_score"),
                    upper_body_score=fused.get("upper_body_score"),
                    symmetry_score=fused.get("symmetry_score"),
                    rhythm_consistency_score=fused.get("rhythm_consistency_score"),
                    overall_score=fused.get("overall_score"),
                    per_view_scores=fused.get("per_view_scores"),
                    score_sources=fused.get("score_sources"),
                    llm_summary=coaching_text,
                )
                session.add(ma_analysis)

        # Mark group complete
        with get_session() as session:
            group = session.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
            if group:
                group.status = "complete"

        logger.info(f"Multi-angle pipeline complete for group {group_id}")
        return {"status": "complete", "group_id": group_id}

    except Exception as e:
        logger.error(f"Multi-angle pipeline failed for group {group_id}: {e}\n{traceback.format_exc()}")

        with get_session() as session:
            group = session.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
            if group:
                group.status = "failed"

        raise


def _generate_multi_angle_coaching(
    dancer_label: str,
    per_view: list[dict],
    fused: dict,
    perf_info: dict,
) -> str:
    """Generate coaching feedback that synthesizes insights from multiple camera angles."""
    import os
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Multi-angle coaching unavailable (API key not configured)."

    # Build per-view summary
    view_summaries = []
    for pv in per_view:
        cam = pv.get("camera_label", "Unknown angle")
        summary_text = pv.get("llm_summary", "No analysis available")
        scores = {
            "aramandi": pv.get("aramandi_score"),
            "upper_body": pv.get("upper_body_score"),
            "symmetry": pv.get("symmetry_score"),
            "rhythm": pv.get("rhythm_consistency_score"),
            "overall": pv.get("overall_score"),
        }
        score_str = ", ".join(f"{k}: {v}" for k, v in scores.items() if v is not None)
        view_summaries.append(
            f"--- {cam} VIEW ---\n"
            f"Scores: {score_str}\n"
            f"Coaching notes:\n{summary_text}\n"
        )

    fused_scores = {
        "aramandi": fused.get("aramandi_score"),
        "upper_body": fused.get("upper_body_score"),
        "symmetry": fused.get("symmetry_score"),
        "rhythm": fused.get("rhythm_consistency_score"),
        "overall": fused.get("overall_score"),
    }
    fused_str = ", ".join(f"{k}: {v}" for k, v in fused_scores.items() if v is not None)

    # Get dance metadata from first perf
    first_perf = perf_info.get(per_view[0]["performance_id"], {})
    item_name = first_perf.get("item_name", "")
    item_type = first_perf.get("item_type", "")

    prompt = f"""You are an expert Bharatanatyam dance guru. A student ({dancer_label}) has been
recorded from multiple camera angles performing {item_name or item_type or 'a dance piece'}.

Each camera view has been analyzed independently. Your task is to synthesize the insights
from all views into unified, multi-angle coaching feedback.

{chr(10).join(view_summaries)}

--- FUSED CONSENSUS SCORES ---
{fused_str}

Key advantages of multi-angle analysis:
- Front view is best for assessing symmetry and arm positioning
- Side view is best for assessing aramandi depth, torso uprightness, and forward lean
- When scores differ between views, explain what each view reveals

Provide synthesized coaching feedback (max 500 words) that:
1. Highlights insights that are only visible from specific angles
2. Notes any discrepancies between views (e.g., "aramandi looks good from front but side view reveals forward lean")
3. Gives specific, actionable corrections referencing which angle revealed the issue
4. Prioritizes the most impactful improvements

Address the dancer respectfully as a guru would."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
