# Threat Model & Privacy

## Data Collected
- Interaction events: {sid, uid, ts, ev, x, y, el_hash, dom_path_short, view:{w,h,y}, perf, aff_label?}
- No raw page text, no keystroke content, no images/video frames.

## Local vs Server
- On-device (extension): optional affect inference; consent UI; buffer in IndexedDB; sendBeacon flush.
- Server: pseudonymous events; aggregates may include DP noise; k-anonymity enforced for cohorts.

## Protections
- Hashed IDs; HTTPS; token on /ingest.
- DP on aggregates (Laplace/Gaussian; (ε,δ) documented).
- k-anonymity threshold k≥5 on any group; groups below k are suppressed.

```mermaid
flowchart LR
  A[Browser Ext\n(on-device)] -- label ticks only --> B[(API)]
  subgraph Client
    A
  end
  subgraph Server
    B --> C[Redis Queue]
    C --> D[Parquet Raw]
    D --> E[Features & Metrics]
    E --> F[DP Aggregator\n(k-anon + noise)]
    E --> G[Model Scores]
    F --> H[Dashboard]
    G --> H
  end
  style A fill:#e0ffe0,stroke:#2e7
  style F fill:#e0f0ff,stroke:#27f