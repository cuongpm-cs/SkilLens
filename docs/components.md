# SkilLens Component Input/Output Reference

This document defines the data contract of every main SkilLens component. JSON
examples are shortened for readability; omitted fields remain available in the
real artifacts.

## Component overview

| Component | Main input | Main output |
|---|---|---|
| Rubric Knowledge Base | Rubric JSON and exam ID | Validated `ExamSpec` and normalized rubric |
| Pipeline Orchestrator | Video, exam ID, backend and model options | One run directory containing all artifacts |
| YOLO Perception | Video, YOLO checkpoint and thresholds | Detections, tracks, contacts and annotated video |
| SAM 3.1 Perception | Video, anchor manifest and Modal application | Segments, hand-tool graph and timeline events |
| Full API Timeline | Video and window duration | Uniform full-video events without detections |
| Audio Evidence | Video audio, rubric modalities and ASR model | Timestamped transcript chunks |
| Observation VLM | Frames, event metadata and compact rubric | Direct visual observations with timestamps |
| Assessment LLM | Full rubric, observations and audio evidence | Criterion scores, evidence links and review flags |
| Unified Evaluation | Assessment output and gold labels | Metrics and criterion-level comparison |

## 1. Rubric Knowledge Base

Implementation: `skillens/knowledge_base.py`

### Input

- A knowledge-base directory containing `*.json` rubric files.
- An exam ID or alias, for example `aseptic`.
- Each rubric contains:
  - exam identity and aliases;
  - criteria and descriptions;
  - legal score values;
  - video/audio modalities;
  - optional YOLO, SAM and gold-label assets.

Example input:

```json
{
  "framework": {
    "exam_id": "new_exam",
    "default_backend": "full_api",
    "audio": "auto"
  },
  "title": "New Clinical Skills Exam",
  "maximum_score": 2,
  "steps": [
    {
      "id": 1,
      "description": "Explain the procedure to the patient.",
      "allowed_scores": [0, 1, 2],
      "max_score": 2,
      "modalities": ["video", "audio"]
    }
  ]
}
```

### Processing

The knowledge base scans all rubric-shaped JSON files and validates:

- unique exam IDs, aliases and criterion IDs;
- non-empty criterion descriptions;
- legal `0/1/2` score values;
- agreement between criterion maxima and `maximum_score`;
- supported default backend and audio settings.

Missing optional fields are normalized. For example, a missing criterion `key`
becomes `criterion_1`, and missing modalities default to `video`.

### Output

An in-memory `ExamSpec` plus the run-specific artifact:

```text
rubric.normalized.json
```

Example normalized criterion:

```json
{
  "id": 1,
  "key": "criterion_1",
  "description": "Explain the procedure to the patient.",
  "allowed_scores": [0, 1, 2],
  "max_score": 2,
  "modalities": ["video", "audio"],
  "required_objects": [],
  "observable_evidence": []
}
```

Useful commands:

```bash
python -m skillens.kb add /path/to/new_exam.json
python -m skillens.kb validate
python -m skillens.run --list-exams
```

## 2. Pipeline Orchestrator

Implementation: `skillens/run.py`

### Input

- Candidate video path.
- `--exam`: ID or alias retrieved from the knowledge base.
- Optional `--backend`: `yolo`, `sam31`, or `full_api`.
- Independent observation, assessment and audio model settings.
- Cache and evaluation options.

Example:

```bash
python -m skillens.run /data/student_1.mp4 \
  --exam aseptic \
  --backend full_api \
  --observation-provider openai-compatible \
  --observation-model Qwen/Qwen3-VL-8B-Instruct \
  --assessment-provider openai-compatible \
  --assessment-model gpt-oss-20b \
  --assessment-images never
```

### Output

The orchestrator creates one run directory:

```text
output/<exam_id>/<video_name>/<backend>/
├── rubric.normalized.json
├── perception.json
├── video_with_boxes.mp4       # YOLO only
├── detections.csv             # YOLO only
├── graph.json                  # SAM only
├── audio_evidence.json
├── observations.json
├── assessment_images.json
├── assessment.json
├── scores.csv
├── evaluation.json            # with --evaluate
└── evaluation_steps.csv       # with --evaluate
```

## 3. Perception Layer

All three backends produce the same minimum contract: `perception.json` with
video metadata and a complete list of timeline events.

### Common event output

```json
{
  "event_id": "event_000",
  "start_seconds": 0.0,
  "end_seconds": 4.974,
  "basis": "uniform_timeline_coverage",
  "objects": [
    {
      "class_name": "vial",
      "detection_count": 1,
      "contact_count": 0,
      "max_confidence": 0.273,
      "track_ids": []
    }
  ]
}
```

`basis` explains why the event exists. Object/contact evidence is a fallible
hint and is not a criterion score.

### 3.1 YOLO + ByteTrack + MediaPipe

Implementations: `skillens/local_yolo.py`, `skillens/modal_yolo.py`, and
`skillens/modal_yolo_app.py`

Input:

- source video;
- YOLO checkpoint;
- confidence and IoU thresholds;
- tracker configuration;
- optional device such as `0` or `cpu`.
- execution mode: local or a deployed Modal Function.

Example:

```bash
python -m skillens.run video.mp4 \
  --exam aseptic --backend yolo \
  --confidence 0.25 --iou 0.70 --device 0 \
  --perception-only
```

Remote Modal example:

```bash
modal deploy -m skillens.modal_yolo_app
python -m skillens.run video.mp4 \
  --exam aseptic --backend yolo \
  --yolo-execution modal --perception-only
```

The Modal service uses one L40S for YOLO and eight CPU cores for MediaPipe. A
private Volume stores input and output caches. The local client calls the Modal
Function API and downloads the same three artifacts produced by local execution.

Primary outputs:

- `video_with_boxes.mp4`: objects, track IDs and hand boxes.
- `detections.csv`: one row for each detection.
- `perception.json`: aggregated tracks and hybrid full-timeline events.

Example CSV row:

```csv
frame_index,time_seconds,class_id,class_name,track_id,confidence,x1,y1,x2,y2,hand_contact,hand_index
16,0.534,1,saline,,0.2526,1447.75,623.10,1478.73,696.26,False,
```

### 3.2 SAM 3.1 + MediaPipe

Implementation: `skillens/perception.py::run_sam31`

Input:

- source video;
- video-specific `manifest.tsv` box anchors;
- SAM threshold and maximum anchors per class;
- analysis FPS and MediaPipe chunk size;
- deployed Modal app and volume names.

Example manifest rows:

```tsv
class	source_frame	x	y	width	height	quality	note
urinary_catheter	130	650	80	600	850	strong	Visible catheter
10_ml_syringe	199	700	180	650	500	strong	Connected syringe
```

Primary outputs:

- `perception.json`: SAM detections and MediaPipe chunks.
- `graph.json`: timestamped hand-tool contact candidates.
- timeline events containing object/contact summaries.

Example graph edge:

```json
{
  "frame_index": 173,
  "timestamp": 86.5,
  "hand": "right",
  "tool_class": "urinary_drainage_bag",
  "relation": "contact_candidate",
  "overlap_over_smaller": 0.8928,
  "landmark_inside_tool_box": true
}
```

The anchor coordinates belong to one specific video. They must not be reused on
a different candidate video.

### 3.3 Full API Timeline

Implementation: `skillens/perception.py::run_full_api`

Input:

- source video;
- event duration, default `30` seconds.

It requires no detector, checkpoint, annotation, or object manifest.

Output example:

```json
{
  "schema_version": "1.0",
  "backend": "full_api",
  "metadata": {
    "duration": 775.6564,
    "width": 1920,
    "height": 1080,
    "audio_available": true,
    "timeline_coverage": "uniform_full_video"
  },
  "events": [
    {
      "event_id": "event_000",
      "start_seconds": 0.0,
      "end_seconds": 30.0,
      "basis": "full_timeline_coverage",
      "objects": []
    }
  ]
}
```

## 4. Audio Evidence

Implementation: `skillens/audio.py`

### Input

- video audio stream;
- audio mode: `auto`, `always`, or `never`;
- rubric modalities;
- transcription provider, model, language and chunk duration.

With `auto`, audio runs only if at least one criterion includes `audio` and the
video contains an audio stream.

### Output

Artifact: `audio_evidence.json`

```json
{
  "schema_version": "1.0",
  "enabled": true,
  "activation_reason": "required by exam rubric",
  "provider": "openai",
  "model": "gpt-4o-mini-transcribe",
  "speaker_policy": "speaker identity is unverified; use transcript as supporting evidence only",
  "chunks": [
    {
      "audio_id": "audio_0000",
      "start_seconds": 0.0,
      "end_seconds": 30.0,
      "text": "I will now explain the procedure...",
      "usable": true,
      "speaker_verified": false
    }
  ],
  "summary": {
    "total_chunks": 26,
    "usable_chunks": 24
  }
}
```

When audio is disabled, the file is still written with `enabled: false` and an
empty `chunks` list.

## 5. Observation VLM

Implementation: `skillens/vlm_assessment.py::_analyze_event`

### Input

- sampled raw-video frames from one event;
- optional annotated frames;
- event timestamps and perception hints;
- compact observation rubric containing:
  - criterion ID, key and description;
  - required objects;
  - video/audio modalities;
  - observable evidence descriptions.

The Observation VLM does not receive criterion scoring rules. Its task is to
describe only direct visual evidence, not assign official scores.

Conceptual prompt input:

```json
{
  "event": {
    "event_id": "event_006",
    "start_seconds": 43.0,
    "end_seconds": 49.0
  },
  "rubric_steps": [
    {
      "id": 1,
      "description": "Remove the wristwatch and wash the hands.",
      "modalities": ["video"],
      "observable_evidence": ["The candidate removes the watch."]
    }
  ],
  "frames": ["43.375s", "44.125s", "...encoded images..."]
}
```

### Output

Artifact: `observations.json`

```json
[
  {
    "event_id": "event_006",
    "start_seconds": 43.0,
    "end_seconds": 49.0,
    "sampled_timestamps": [43.375, 44.125, 44.875],
    "objects": [],
    "observations": [
      {
        "observation_id": "event_006_obs_000",
        "event_id": "event_006",
        "step_id": 1,
        "status": "complete",
        "confidence": 0.93,
        "evidence_seconds": [44.125, 44.875],
        "description": "The candidate removes an object from the wrist and rubs both hands."
      }
    ],
    "visual_limitations": [],
    "safety_concerns": []
  }
]
```

Only supplied timestamps and valid rubric IDs survive output normalization.

## 6. Assessment LLM

Implementations: `skillens/vlm_assessment.py::_judge` and
`skillens/vlm_assessment.py::_review_step_with_images`

### Input

- complete scoring rubric;
- all normalized visual observations across the video;
- usable audio chunks;
- optional raw frames selected for each criterion;
- configured confidence threshold and assessment policy.

The complete rubric view includes:

- legal scores and per-criterion `scoring_rules`;
- `do_not_infer` instructions;
- assessment policy;
- sequence constraints;
- critical rules and score calculation;
- maximum score and score scale.

Conceptual prompt input:

```json
{
  "criterion": {
    "id": 1,
    "allowed_scores": [0, 2],
    "scoring_rules": {
      "2": "Direct evidence shows the action was completed.",
      "0": "Direct evidence shows incorrect performance."
    }
  },
  "observations": ["event_006_obs_000"],
  "audio_evidence": ["audio_0000"]
}
```

Assessment runs in two passes:

1. **Text-first scoring:** the complete observation timeline is mapped to the
   rubric.
2. **Optional image verification:** only a vision-capable Assessment LLM can
   recheck selected criteria from raw frames. The GPT-OSS-20B default is
   text-only, so this pass is disabled by default. When enabled, the request
   contains the full criterion rules but not the Observation VLM's prose.
3. **Balanced fusion (when image review is enabled):** text remains primary
   unless direct visual evidence is substantially more confident. Each
   decision is preserved in audit fields.

Image modes:

- `never`: disable the second pass (default with GPT-OSS-20B);
- `rescue`: review only text-first `unverifiable` criteria;
- `uncertain`: review unresolved, flagged, or low-confidence criteria;
- `evidence`: review every criterion with direct observation timestamps;
- `always`: review all criteria and use a temporal search prior when no direct
  timestamp exists.

The timestamp selector uses raw, unannotated frames. For example, one anchor at
`44.125` seconds with one context frame and a three-image limit produces frames
near `43.125`, `44.125`, and `45.125` seconds. The timestamp locates evidence;
the associated observation description is not sent to the visual verifier.

### Output

Artifacts: `assessment_images.json`, `assessment.json`, and `scores.csv`

Example `assessment_images.json` when using a multimodal assessment model:

```json
{
  "mode": "uncertain",
  "targets": [
    {
      "step_id": 1,
      "critical": false,
      "selection_basis": "observation_timestamp",
      "timestamps": [43.125, 44.125, 45.125],
      "status": "completed",
      "cache_hit": false
    }
  ],
  "summary": {
    "selected_steps": 1,
    "selected_images": 3,
    "completed_steps": 1,
    "failed_steps": 0
  }
}
```

```json
{
  "summary": {
    "official_score": null,
    "maximum_score": 60,
    "confirmed_score": 33,
    "score_range": [33, 57],
    "scored_steps": 18,
    "unverifiable_steps": 12,
    "requires_human_review": true
  },
  "steps": [
    {
      "step_id": 1,
      "key": "wash_hands",
      "score": 2,
      "max_score": 2,
      "verdict": "complete",
      "confidence": 0.93,
      "score_basis": "observation_evidence",
      "evidence_ids": ["event_006_obs_000"],
      "audio_evidence_ids": [],
      "reason": "Direct evidence shows watch removal and hand hygiene.",
      "requires_review": false,
      "review_reasons": []
    }
  ],
  "sequence_violations": [],
  "overall_feedback": "...",
  "safety_concerns": []
}
```

Important behavior:

- insufficient evidence becomes `score: null`, not zero;
- an image-grounded result confirms matching text, rescues unresolved text, or
  overrides conflicting text only when its direct evidence is substantially more
  confident;
- illegal scores are rejected;
- low-confidence evidence is flagged or abstained according to policy;
- critical deductions requiring continuous evidence or human confirmation are
  reported for review rather than invented automatically.

## 7. Cache Layer

Implementations: `skillens/local_yolo_cache.py`, observation caches, assessment
image caches, and audio cache keys.

### Input identity

Cache keys include the relevant stage inputs:

- YOLO: source video, checkpoint, tracker and perception parameters;
- Observation: event, perception cache, observation model, image settings and rubric ID;
- Audio: video identity, provider, model, language and chunk duration.
- Assessment images: video identity, rubric criterion, timestamps, assessment
  provider/model, image settings and nearby permitted audio.

### Output

- YOLO cache metadata is embedded in `perception.json`.
- Observation events are stored under `.cache/observations/`.
- Audio cache metadata is embedded in `audio_evidence.json`.
- Per-criterion visual assessment responses are stored under
  `.cache/assessment_images/`; their selection manifest is
  `assessment_images.json`.

Changing only the Assessment LLM reuses perception and observations. Use:

```bash
python -m skillens.run video.mp4 \
  --exam aseptic --backend full_api --assessment-only \
  --assessment-model gpt-oss-20b
```

## 8. Unified Evaluation

Implementation: `skillens/evaluate.py`

### Input

- `assessment.json`;
- exam rubric from the knowledge base;
- gold labels:
  - existing exam XLSX formats; or
  - generic long CSV for a new exam;
- optional candidate ID.

Generic gold example:

```csv
candidate_id,criterion_id,score
student_1,1,2
student_1,2,1
```

Example command:

```bash
python -m skillens.evaluate \
  --exam new_exam \
  --student-name student_1 \
  output/new_exam/student_1/full_api/assessment.json
```

### Output

Artifacts: `evaluation.json` and `evaluation_steps.csv`

```json
{
  "exam_id": "new_exam",
  "candidate_id": "student_1",
  "gold_total": 3,
  "predicted_confirmed_total": 2,
  "maximum_score": 4,
  "metrics": {
    "coverage": 1.0,
    "strict_exact_accuracy": 0.5,
    "selective_exact_accuracy": 0.5,
    "strict_soft_accuracy": 0.75,
    "mae": 0.5,
    "quadratic_weighted_kappa": 0.0,
    "confusion_matrix": {
      "labels": [0, 1, 2],
      "rows_are_gold": [[0, 0, 0], [0, 1, 0], [0, 1, 0]]
    }
  }
}
```

Soft accuracy awards `1.0` for an exact score, `0.5` for a one-point error, and
`0.0` for a two-point error. Selective metrics exclude abstentions; strict
metrics count abstentions as zero credit.

## End-to-end data flow example

```text
aseptic.json + student_1.mp4
        │
        ├── Knowledge Base → rubric.normalized.json
        │
        ├── YOLO Perception → perception.json + detections.csv + boxed video
        │
        ├── Audio Policy → audio_evidence.json
        │
        ├── Observation VLM → observations.json
        │
        ├── Assessment LLM → assessment.json + scores.csv
        │
        └── Gold Evaluation → evaluation.json + evaluation_steps.csv
```
