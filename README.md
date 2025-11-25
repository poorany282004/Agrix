# Agrix — Rice Disease Detection (Minimal Prototype)

## Goal
A simple software-only Agrix prototype:
- Upload or capture leaf images via the browser
- Lightweight image heuristic predictor (no heavy ML required)
- Sensor fusion (simulated) to compute disease stage:
  HEALTHY / PARTIALLY_INFECTED / FULLY_INFECTED
- Stage-specific advisory from JSON
- Dashboard to view last scan

## Setup (local)
1. Clone project folder `Agrix/` or create it and add files shown above.
2. Copy your logo image to `static/images/logo.png`.
3. (Optional) create virtualenv:
