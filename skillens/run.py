"""SkilLens command-line entry point for clinical skill assessment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .assessment import run_assessment
from .audio import build_audio_evidence, should_use_audio
from .evaluate import evaluate_assessment
from . import model_config
from .knowledge_base import DEFAULT_KNOWLEDGE_BASE, PROJECT_ROOT, RubricKnowledgeBase
from .media import probe_video
from .perception import run_full_api, run_sam31, run_yolo, run_yolo_modal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a shared nursing-skill assessment pipeline with selectable stages."
    )
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--exam", help="Exam ID or alias discovered from the knowledge base.")
    parser.add_argument("--knowledge-base", type=Path, action="append")
    parser.add_argument("--list-exams", action="store_true")
    parser.add_argument("--backend", choices=("yolo", "sam31", "full_api"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force-perception", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--perception-only", action="store_true")
    parser.add_argument("--assessment-only", action="store_true")

    perception = parser.add_argument_group("perception")
    perception.add_argument("--yolo-model", type=Path)
    perception.add_argument(
        "--yolo-execution", choices=("local", "modal"), default="local"
    )
    perception.add_argument(
        "--modal-yolo-app",
        default=os.getenv("MODAL_YOLO_APP", "skillens-yolo-perception"),
    )
    perception.add_argument(
        "--modal-yolo-volume",
        default=os.getenv("MODAL_YOLO_VOLUME", "skillens-yolo-data"),
    )
    perception.add_argument("--device")
    perception.add_argument("--confidence", type=float, default=0.25)
    perception.add_argument("--iou", type=float, default=0.70)
    perception.add_argument("--tracker", default="bytetrack.yaml")
    perception.add_argument("--sam-manifest", type=Path)
    perception.add_argument("--modal-app")
    perception.add_argument("--modal-volume")
    perception.add_argument("--sam-analysis-fps", type=float, default=2.0)
    perception.add_argument("--sam-threshold", type=float, default=0.45)
    perception.add_argument("--sam-max-anchors", type=int, default=1)
    perception.add_argument("--hand-chunk-frames", type=int, default=300)
    perception.add_argument("--event-seconds", type=float, default=30.0)

    observation = parser.add_argument_group("observation model")
    observation.add_argument(
        "--observation-provider",
        choices=("openai", "openai-compatible", "modal"),
        default=os.getenv("OBSERVATION_PROVIDER", "openai-compatible"),
    )
    observation.add_argument("--observation-model")
    observation.add_argument("--observation-base-url")
    observation.add_argument("--observation-workers", type=int, default=4)
    observation.add_argument("--frames-per-event", type=int, default=8)
    observation.add_argument("--max-image-side", type=int, default=1280)
    observation.add_argument("--image-detail", choices=("low", "high", "auto", "original"), default="high")

    assessment = parser.add_argument_group("assessment model")
    assessment.add_argument(
        "--assessment-provider",
        choices=("openai", "openai-compatible", "modal"),
        default=os.getenv("ASSESSMENT_PROVIDER", "openai-compatible"),
    )
    assessment.add_argument("--assessment-model")
    assessment.add_argument("--assessment-base-url")
    assessment.add_argument("--assessment-policy", choices=("conservative", "lenient"), default="conservative")
    assessment.add_argument(
        "--assessment-images",
        choices=("never", "rescue", "uncertain", "evidence", "always"),
        default=os.getenv("ASSESSMENT_IMAGES", "never"),
        help=(
            "Raw-image verification policy for the assessment model. "
            "Default is never because gpt-oss-20b is text-only."
        ),
    )
    assessment.add_argument(
        "--assessment-image-confidence-threshold",
        type=float,
        default=float(os.getenv("ASSESSMENT_IMAGE_CONFIDENCE_THRESHOLD", "0.75")),
        help="Review criteria whose text-first confidence is below this value.",
    )
    assessment.add_argument(
        "--assessment-visual-override-confidence",
        type=float,
        default=float(os.getenv("ASSESSMENT_VISUAL_OVERRIDE_CONFIDENCE", "0.85")),
        help="Minimum confidence for images to override a conflicting text score.",
    )
    assessment.add_argument(
        "--assessment-visual-override-margin",
        type=float,
        default=float(os.getenv("ASSESSMENT_VISUAL_OVERRIDE_MARGIN", "0.15")),
        help="Required visual confidence advantage over a conflicting text score.",
    )
    assessment.add_argument(
        "--assessment-images-per-criterion",
        type=int,
        default=int(os.getenv("ASSESSMENT_IMAGES_PER_CRITERION", "3")),
    )
    assessment.add_argument(
        "--assessment-context-frames",
        type=int,
        default=int(os.getenv("ASSESSMENT_CONTEXT_FRAMES", "1")),
    )
    assessment.add_argument(
        "--critical-criterion-images",
        type=int,
        default=int(os.getenv("CRITICAL_CRITERION_IMAGES", "6")),
    )
    assessment.add_argument(
        "--assessment-image-workers",
        type=int,
        default=int(os.getenv("ASSESSMENT_IMAGE_WORKERS", "4")),
    )
    assessment.add_argument(
        "--criterion-localization",
        choices=("off", "on"),
        default=os.getenv("CRITERION_LOCALIZATION", "off"),
        help=(
            "Locate each selected rubric action in the full timeline before "
            "sending evidence frames to the assessment model."
        ),
    )
    assessment.add_argument(
        "--localization-candidates",
        type=int,
        default=int(os.getenv("LOCALIZATION_CANDIDATES", "3")),
    )
    assessment.add_argument(
        "--localization-coarse-seconds",
        type=float,
        default=float(os.getenv("LOCALIZATION_COARSE_SECONDS", "5")),
    )
    assessment.add_argument(
        "--localization-search-radius",
        type=float,
        default=float(os.getenv("LOCALIZATION_SEARCH_RADIUS", "5")),
    )
    assessment.add_argument(
        "--localization-refinement-fps",
        type=float,
        default=float(os.getenv("LOCALIZATION_REFINEMENT_FPS", "8")),
    )
    assessment.add_argument(
        "--localization-evidence-images",
        type=int,
        default=int(os.getenv("LOCALIZATION_EVIDENCE_IMAGES", "12")),
    )
    assessment.add_argument(
        "--localization-confidence-threshold",
        type=float,
        default=float(os.getenv("LOCALIZATION_CONFIDENCE_THRESHOLD", "0.70")),
    )
    assessment.add_argument(
        "--localization-workers",
        type=int,
        default=int(os.getenv("LOCALIZATION_WORKERS", "4")),
    )
    assessment.add_argument("--confidence-threshold", type=float, default=0.60)
    assessment.add_argument(
        "--assessment-rescue-images",
        type=int,
        default=30,
        help="Maximum number of unresolved criteria reviewed in rescue mode.",
    )
    assessment.add_argument("--retries", type=int, default=2)

    audio = parser.add_argument_group("audio")
    audio.add_argument("--audio", choices=("auto", "always", "never"))
    audio.add_argument("--audio-provider", choices=("openai", "openai-compatible"), default=os.getenv("AUDIO_PROVIDER", "openai"))
    audio.add_argument("--audio-model", default=os.getenv("AUDIO_MODEL", "gpt-4o-mini-transcribe"))
    audio.add_argument("--audio-base-url", default=os.getenv("AUDIO_BASE_URL"))
    audio.add_argument("--audio-api-key", default=os.getenv("AUDIO_API_KEY"))
    audio.add_argument("--audio-language", default=os.getenv("AUDIO_LANGUAGE", "zh"))
    audio.add_argument("--audio-chunk-seconds", type=float, default=30.0)
    audio.add_argument("--audio-workers", type=int, default=4)
    audio.add_argument("--force-audio", action="store_true")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--evaluate", action="store_true")
    evaluation.add_argument("--gold-workbook", type=Path)
    evaluation.add_argument("--student-name")
    return parser


def _default_output(exam: str, video: Path, backend: str) -> Path:
    return PROJECT_ROOT / "output" / exam / video.stem / backend


def _load_cached_perception(
    path: Path,
    video: Path,
    backend: str,
    expected_stage_config: dict | None = None,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Assessment-only requires cached perception: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("backend") != backend:
        raise ValueError(f"Cached backend is {result.get('backend')!r}, requested {backend!r}")
    source = result.get("metadata", {}).get("source_video")
    if source and Path(source).resolve() != video.resolve():
        raise ValueError("Cached perception belongs to a different source video")
    if expected_stage_config is not None and result.get("stage_config") != expected_stage_config:
        raise ValueError("Cached SAM manifest or perception settings changed")
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parser().parse_args(argv)
    kb = RubricKnowledgeBase(args.knowledge_base or (DEFAULT_KNOWLEDGE_BASE,))
    if args.list_exams:
        for spec in kb.list():
            warnings = kb.validate_assets(spec)
            status = "OK" if not warnings else f"{len(warnings)} asset warning(s)"
            print(
                f"{spec.exam_id}\t{len(spec.rubric['steps'])} criteria\t"
                f"{spec.rubric['maximum_score']} points\t{spec.default_backend}\t{status}"
            )
        return 0
    if not args.exam or args.video is None:
        raise ValueError("video and --exam are required unless --list-exams is used")
    spec = kb.get(args.exam)
    backend = args.backend or spec.default_backend
    audio_mode = args.audio or str(spec.config("audio", "auto"))
    observation_endpoint = None
    assessment_endpoint = None
    if not args.perception_only:
        observation_endpoint = model_config.resolve_model_endpoint(
            stage="observation",
            provider=args.observation_provider,
            model=args.observation_model,
            base_url=args.observation_base_url,
            default_openai_model="gpt-4.1",
            default_compatible_model=model_config.DEFAULT_OBSERVATION_MODEL,
        )
        assessment_endpoint = model_config.resolve_model_endpoint(
            stage="assessment",
            provider=args.assessment_provider,
            model=args.assessment_model,
            base_url=args.assessment_base_url,
            default_openai_model="gpt-4.1",
            default_compatible_model=model_config.DEFAULT_ASSESSMENT_MODEL,
        )
        if model_config.is_text_only_assessment_model(assessment_endpoint.model) and args.assessment_images != "never":
            raise ValueError(
                f"{assessment_endpoint.model} cannot review images. Set "
                "--assessment-images never, or choose a multimodal assessment model."
            )
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output_dir = (args.output_dir or _default_output(spec.exam_id, video, backend)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rubric_path = output_dir / "rubric.normalized.json"
    rubric_path.write_text(
        json.dumps(spec.rubric, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rubric = spec.rubric
    print(f"Exam: {rubric['title']} ({spec.exam_id})")
    print(f"Rubric: {spec.rubric_path}")
    print(f"Backend: {backend}")
    print(f"Output: {output_dir}")
    perception_path = output_dir / "perception.json"
    if args.assessment_only:
        perception = _load_cached_perception(perception_path, video, backend)
        print("Perception: reused by --assessment-only")
    elif backend == "yolo":
        model = args.yolo_model or spec.asset_path("yolo_model")
        if model is None:
            raise ValueError("This exam has no verified YOLO weights; supply --yolo-model")
        if args.yolo_execution == "modal":
            perception = run_yolo_modal(
                video,
                model,
                output_dir,
                confidence=args.confidence,
                iou=args.iou,
                tracker=args.tracker,
                force=args.force_perception,
                modal_app=args.modal_yolo_app,
                modal_volume=args.modal_yolo_volume,
            )
        else:
            perception = run_yolo(
                video,
                model,
                output_dir,
                confidence=args.confidence,
                iou=args.iou,
                device=args.device,
                tracker=args.tracker,
                force=args.force_perception,
            )
    elif backend == "sam31":
        manifest = args.sam_manifest or spec.asset_path("sam_manifest")
        if manifest is None:
            raise ValueError("This exam has no verified SAM manifest; supply --sam-manifest")
        manifest = manifest.expanduser().resolve()
        if (
            args.sam_manifest is None
            and spec.asset_path("sam_manifest_video") is not None
            and video != spec.asset_path("sam_manifest_video")
        ):
            raise ValueError(
                "The bundled SAM anchors are specific to the reference candidate video; "
                "supply a manifest annotated for this video with --sam-manifest"
            )
        modal_app = args.modal_app or spec.asset_value("modal_app")
        modal_volume = args.modal_volume or spec.asset_value("modal_volume")
        if not modal_app or not modal_volume:
            raise ValueError(
                "SAM backend requires modal_app and modal_volume in the rubric "
                "assets or through --modal-app and --modal-volume"
            )
        sam_stage_config = {
            "manifest": str(manifest),
            "manifest_size": manifest.stat().st_size,
            "manifest_mtime_ns": manifest.stat().st_mtime_ns,
            "analysis_fps": args.sam_analysis_fps,
            "event_seconds": args.event_seconds,
            "threshold": args.sam_threshold,
            "max_anchors": args.sam_max_anchors,
            "hand_chunk_frames": args.hand_chunk_frames,
            "modal_app": str(modal_app),
            "modal_volume": str(modal_volume),
            "max_instances": spec.asset_value("sam_max_instances", {}),
        }
        perception = None
        if perception_path.is_file() and not args.force_perception:
            try:
                perception = _load_cached_perception(
                    perception_path,
                    video,
                    backend,
                    expected_stage_config=sam_stage_config,
                )
                print("SAM perception cache hit (validated local artifacts)")
            except ValueError as error:
                print(f"SAM perception cache miss: {error}")
        if perception is None:
            perception = run_sam31(
                video,
                rubric_path,
                manifest,
                output_dir,
                analysis_fps=args.sam_analysis_fps,
                event_seconds=args.event_seconds,
                threshold=args.sam_threshold,
                max_anchors=args.sam_max_anchors,
                hand_chunk_frames=args.hand_chunk_frames,
                force=args.force_perception,
                modal_app=sam_stage_config["modal_app"],
                modal_volume=sam_stage_config["modal_volume"],
                class_prompts=spec.asset_value("sam_prompts", {}),
                class_max_instances=spec.asset_value("sam_max_instances", {}),
            )
    else:
        perception = run_full_api(video, output_dir, event_seconds=args.event_seconds)
    if args.perception_only:
        print(f"Perception complete: {perception_path}")
        return 0

    video_info = probe_video(video)
    audio_enabled, audio_reason = should_use_audio(
        audio_mode,
        rubric=rubric,
        profile_requires_audio=spec.requires_audio,
        video_has_audio=video_info["audio_available"],
    )
    audio_evidence = build_audio_evidence(
        video,
        output_dir,
        enabled=audio_enabled,
        activation_reason=audio_reason,
        provider=args.audio_provider,
        model=args.audio_model,
        base_url=args.audio_base_url,
        api_key=args.audio_api_key,
        language=args.audio_language,
        chunk_seconds=args.audio_chunk_seconds,
        workers=args.audio_workers,
        force=args.force_audio,
    )
    print(f"Audio: {'enabled' if audio_enabled else 'disabled'} ({audio_reason})")

    assert observation_endpoint is not None and assessment_endpoint is not None
    result = run_assessment(
        video=video,
        perception=perception,
        rubric_path=rubric_path,
        output_dir=output_dir,
        observation_endpoint=observation_endpoint,
        assessment_endpoint=assessment_endpoint,
        audio_evidence=audio_evidence,
        observation_workers=args.observation_workers,
        frames_per_event=args.frames_per_event,
        max_image_side=args.max_image_side,
        image_detail=args.image_detail,
        confidence_threshold=args.confidence_threshold,
        assessment_policy=args.assessment_policy,
        assessment_images=args.assessment_images,
        assessment_image_confidence_threshold=args.assessment_image_confidence_threshold,
        assessment_visual_override_confidence=args.assessment_visual_override_confidence,
        assessment_visual_override_margin=args.assessment_visual_override_margin,
        assessment_images_per_criterion=args.assessment_images_per_criterion,
        assessment_context_frames=args.assessment_context_frames,
        critical_criterion_images=args.critical_criterion_images,
        assessment_image_workers=args.assessment_image_workers,
        assessment_rescue_images=args.assessment_rescue_images,
        criterion_localization=args.criterion_localization == "on",
        localization_candidates=args.localization_candidates,
        localization_coarse_seconds=args.localization_coarse_seconds,
        localization_search_radius=args.localization_search_radius,
        localization_refinement_fps=args.localization_refinement_fps,
        localization_evidence_images=args.localization_evidence_images,
        localization_confidence_threshold=args.localization_confidence_threshold,
        localization_workers=args.localization_workers,
        retries=args.retries,
        resume=not args.no_resume,
    )
    print(f"Assessment: {result['summary']['confirmed_score']}/{result['summary']['maximum_score']}")
    if args.evaluate:
        evaluation = evaluate_assessment(
            spec=spec,
            assessment_path=output_dir / "assessment.json",
            workbook=args.gold_workbook,
            student_name=args.student_name,
            output_dir=output_dir,
        )
        print(json.dumps(evaluation["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
