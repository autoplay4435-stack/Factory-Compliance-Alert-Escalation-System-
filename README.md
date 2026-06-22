# Factory Compliance and Alert Escalation System

A 5-module automated system for factory floor compliance monitoring. It combines
computer vision detection, policy-derived severity routing, immutable audit
reports, and a Streamlit operations dashboard.

## Architecture

```text
factory-compliance-system/
├── README.md
├── compliance_policy.pdf
├── requirements.txt
├── data/
├── outputs/
├── src/
│   ├── detection/
│   │   ├── detectors.py
│   │   ├── engine.py
│   │   └── video_source.py
│   ├── severity/
│   │   ├── parser.py
│   │   └── matrix.py
│   ├── escalation/
│   │   └── pipeline.py
│   ├── reports/
│   │   └── database.py
│   └── dashboard/
│       └── app.py
├── scripts/
│   ├── run_parser.py
│   ├── run_detection.py
│   └── run_all.py
└── tests/
    ├── test_detectors.py
    ├── test_parser.py
    ├── test_matrix.py
    ├── test_pipeline.py
    └── test_database.py
```

## Data Flow

```text
compliance_policy.pdf
        |
        v
src/severity/parser.py  --->  outputs/policy_rules.json
        |                              |
        |                              v
        |                      src/detection/engine.py
        |                              |
        |                              v
        |                      src/escalation/pipeline.py
        |                              |
        |                              v
        |                      src/reports/database.py
        |                              |
        |                              v
        +--------------------> src/dashboard/app.py
```

## Module Coverage

| Module | Directory | Purpose |
|--------|-----------|---------|
| 1 | `src/detection/` | OpenCV + MediaPipe detection for the four behavior classes |
| 2 | `src/severity/` | Policy parsing and severity routing matrix |
| 3 | `src/escalation/` | Direct routing workflow for database logging and dashboard strobe alerts |
| 4 | `src/reports/` | SQLite audit records and CSV export |
| 5 | `src/dashboard/` | Streamlit views for live status, timeline, historical log, and export |


## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional policy parsing requires an OpenAI API key:

```bash
set OPENAI_API_KEY=sk-your-key-here
```

## Running the System

Full system:

```bash
python scripts/run_all.py --fallback --webcam 0
python scripts/run_all.py --video data/factory_clip.mp4
```

Dashboard only:

```bash
streamlit run src/dashboard/app.py
```

Detection only:

```bash
python scripts/run_detection.py --webcam 0 --display
```

## Model Selection Rationale

The detection engine uses deterministic OpenCV and MediaPipe techniques so each
finding can be traced back to policy-derived thresholds and coordinates. The
policy parser can use an LLM once at setup time, and hardcoded fallback rules are
available when an API key is not configured.

Classical computer vision was selected because the policy defines concrete
visual indicators: green walkway boundaries, vest color, panel state, and block
count. These can be represented as color thresholds, fixed regions, contour
counts, and point-in-polygon tests without training a custom object detector.
This keeps the pipeline explainable for an audit-style take-home assignment.

## Known Limitations

1. Detection accuracy depends on camera angle, lighting, and ROI calibration.
   The fallback `zone_polygon` and `roi_coords` values may need tuning for new
   videos.
2. MediaPipe pose detection is required for vest and walkway checks. If a
   person is occluded or pose landmarks are not detected, those detectors skip
   the frame.
3. The system detects violations frame-by-frame and does not track the same
   person or forklift across time, so repeated detections may represent the same
   real-world incident.
4. The current parser expects extracted policy text. Direct binary PDF text
   extraction is not built into the parser.
5. The dashboard is a functional Streamlit monitor, not a production security
   product. It has no authentication, user roles, or deployment hardening.
6. The fallback rules are policy-aligned but static. For a different facility,
   the policy parser output or fallback rule parameters should be regenerated.
