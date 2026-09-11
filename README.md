# NeuroScan AI — Brain MRI Screening Dashboard

A full-stack web app wrapping your two-stage EfficientNet-B0 brain MRI tumor
classifier (originally packaged as a Windows `.exe`) in a real FastAPI backend
and a browser dashboard — no Windows/desktop app required, deployable to any
Linux server, VPS, or platform like Render/Railway/Fly.io.

## What this is

- **Stage 1** (binary EfficientNet-B0): tumor vs. no-tumor, tuned for high
  sensitivity (~99.9%) at ≥90% specificity, exactly as calibrated in your
  original `stage1/deployment_bundle.json`.
- **Stage 2** (3-class EfficientNet-B0): glioma / meningioma / pituitary,
  only run if Stage 1 flags a tumor.
- **OOD detection**: energy-based out-of-distribution check on both stages,
  using the exact cutoffs from your original bundles, so non-MRI or garbage
  input is flagged as "uncertain" instead of forced into a confident answer.
- All thresholds, temperatures, and class names are read directly from your
  original `deployment_bundle.json` files — this is a faithful re-implementation
  of the original app's decision logic, running your original `.pt` weights.

## Project structure

```
app/
├── backend/
│   ├── main.py              # FastAPI app: /api/predict, /api/history, /api/stats, /api/history/{id}/report
│   ├── inference.py         # Preprocessing + 2-stage model pipeline
│   ├── database.py          # SQLite scan history storage
│   ├── requirements.txt
│   ├── configs/             # Copied from your original bundle (common/stage1/stage2 yaml)
│   ├── model_weights/
│   │   ├── stage1/          # stage1_efficientnet_b0_best.pt + deployment_bundle.json
│   │   └── stage2/          # stage2_efficientnet_b0_best.pt + deployment_bundle.json
│   └── history.db           # created automatically on first run
└── frontend/
    └── index.html           # Dashboard UI (vanilla JS + Chart.js, no build step)
```

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` — the dashboard is served directly by the
same FastAPI app (no separate frontend server needed).

## How a user runs the GitHub version

After cloning the repository, users can run the complete app with Docker:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
docker build -t neuroscan-ai .
docker run --rm -p 8000:8000 neuroscan-ai
```

While the container is running, open a web browser and enter
`http://localhost:8000` in the address bar. The browser connects to the
FastAPI server, which serves both the dashboard and the `/api` endpoints.

Users without Docker can install Python 3.11 or newer and run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

They then open `http://127.0.0.1:8000` in a web browser. Do not open
`frontend/index.html` directly with `file://`, because the dashboard needs the
running FastAPI server to process uploads and provide the API responses.

## Deploying

This is a single deployable unit: one Python process serves both the API and
the static dashboard. It will run as-is on:
- A VPS (DigitalOcean, Linode, Hetzner, AWS EC2) behind nginx/Caddy as a
  reverse proxy, with `uvicorn` run via `systemd` or inside a `screen`/`tmux`.
- Render / Railway / Fly.io: point the build at `backend/`, install
  `requirements.txt`, run `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Docker: a minimal Dockerfile based on `python:3.11-slim`, installing
  `requirements.txt` and running the same uvicorn command, works unchanged.

**GPU note:** inference runs on CPU by default and works fine for single-image
requests. If you deploy on a GPU instance, `inference.py` will automatically
use CUDA if `torch.cuda.is_available()`.

## Important — read before showing this to anyone else

This reproduces your model's outputs faithfully, but a few things are
worth being clear-eyed about before calling it "done":

1. **It is not a validated medical device.** The dashboard displays a
   disclaimer for this reason — keep it. Regulatory clearance (e.g. FDA
   510(k) or equivalent) is a completely different, much longer process than
   building the software.
2. **No independent test-set evaluation was done here.** The sensitivity/
   specificity numbers shown come from your original bundle's validation
   run, not from a fresh evaluation against held-out data. If you haven't
   already, run this pipeline against a labeled test set you trust before
   relying on the numbers.
3. **Preprocessing had to be reconstructed from `common.yaml`.** I matched
   resize, percentile clipping, grayscale handling, and normalization exactly
   as configured, with one approximation: PIL doesn't have a literal "area"
   interpolation mode, so bilinear resize is used instead. For 224×224
   downsampling of MRI slices the visual difference is generally negligible,
   but if you have the original training/eval code, it's worth a quick
   side-by-side check against a few known images to confirm outputs match
   bit-for-bit.
4. **History is local SQLite**, fine for a single-instance deployment or
   internal tool; if you expect concurrent multi-user traffic at scale,
   swap it for Postgres before going further.

## Where this fits in the current ML landscape

Two-stage cascades like this (cheap screen → expensive specialist model) are
increasingly the standard pattern for medical imaging triage tools, because
they let you tune sensitivity and specificity independently at each stage —
exactly what your Stage 1 threshold does. Energy-based OOD detection (used
here) is a lightweight, increasingly common way to make classifiers safer in
deployment without retraining — worth knowing about if you build more of
these, since it generalizes to any classifier with logits, not just this one.
