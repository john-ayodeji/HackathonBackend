# PDI Backend — Predictive Deterioration Index

A Flask-based REST API that supports ICU nurses in monitoring patients and detecting clinical deterioration early. The system combines rule-based vital-sign thresholds, a composite risk scoring algorithm, and an AI-powered nursing note generator (Claude + UMLS) to provide real-time, actionable patient assessments.

---

## What the project does

| Capability | Description |
|---|---|
| **Nurse authentication** | Username/password login with Bearer-token sessions stored in MongoDB |
| **Ward census** | Returns all pre-seeded ICU patients sorted by risk level (critical first) |
| **Vital-sign logging** | Appends HR, RR, SpO₂, Temp, SBP, DBP readings for any patient |
| **PDI scoring** | Computes a 0–100 composite Predictive Deterioration Index score from live vitals |
| **Trend projection** | Linear regression over past readings to project each vital 4 hours forward |
| **Alert generation** | Fires `crit` or `warn` alerts with specific recommended nursing actions |
| **AI note generation** | Uses Anthropic Claude + UMLS clinical terminology to auto-write SOAP nursing notes |
| **Note analysis** | Analyses manually entered nursing notes for deterioration risk markers |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (mobile / web — not in this repo)     │
└───────────────────┬─────────────────────────────┘
                    │ HTTP + Bearer token
┌───────────────────▼─────────────────────────────┐
│  app.py — Flask REST API (port 5000)            │
│   /auth/*   /patients/*   /thresholds           │
└──────┬────────────┬─────────────────────────────┘
       │            │
┌──────▼──────┐  ┌──▼──────────────────────────────┐
│  engine.py  │  │  nlp.py                         │
│  PDI core   │  │  Claude AI + UMLS note gen      │
│  - thresholds│  │  - generate_nursing_note()      │
│  - regression│  │  - analyze_nursing_note()       │
│  - scoring  │  └──────────────┬──────────────────┘
│  - alerts   │                 │ Anthropic API
└──────┬──────┘  ┌──────────────▼──────────────────┐
       │         │  Anthropic Claude claude-sonnet  │
┌──────▼──────────────────────────────────────────┐
│  db.py — MongoDB Atlas                          │
│  Collections: nurses · patients · sessions      │
└─────────────────────────────────────────────────┘
```

### Key modules

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, seed data, Bearer-token auth helper |
| `engine.py` | Clinical thresholds, linear regression, PDI composite score, alert generation |
| `nlp.py` | Anthropic Claude prompting, UMLS concept mapping, nursing note generation |
| `db.py` | MongoDB Atlas connection, nurse/patient/session CRUD, password hashing |
| `requirements.txt` | Python dependencies |
| `test_config.py` | Pre-flight configuration checker (env vars, packages, DB connectivity) |

---

## Clinical logic

### PDI composite score (0–100)

Each vital sign contributes a weighted portion to the composite score:

| Vital | Weight |
|---|---|
| SpO₂ | 25 |
| Heart Rate | 20 |
| Systolic BP | 18 |
| Temperature | 15 |
| Diastolic BP | 12 |
| Respiratory Rate | 10 |

A linear regression is fitted to the reading history of each vital and projected 4 hours ahead. Deviation from the normal range drives the score. An AI boost of up to +20 points is added when nursing notes flag high-risk clinical markers.

Risk levels:

| Score | Level |
|---|---|
| 0–29 | `ok` — routine monitoring |
| 30–59 | `warn` — escalate to senior nurse |
| 60–100 | `crit` — notify ICU physician STAT |

### Normal ranges

| Vital | Normal range |
|---|---|
| Heart Rate | 60–100 bpm |
| Respiratory Rate | 12–18 /min |
| SpO₂ | 95–100 % |
| Temperature | 36.5–37.3 °C |
| Systolic BP | 90–120 mmHg |
| Diastolic BP | 60–80 mmHg |

---

## Pre-seeded data

The application seeds the following records into MongoDB on startup (safe to restart — duplicates are skipped).

### Nurses

> ⚠️ **Development only.** These credentials are intentionally weak and are suitable for local development and hackathon demos only. Replace them with strong, unique passwords before any production or patient-facing deployment.

| Username | Password | Role |
|---|---|---|
| `amaka` | `nurse123` | nurse |
| `tunde` | `nurse123` | nurse |
| `chioma` | `nurse123` | nurse |
| `admin` | `admin123` | admin |

### Patients (ICU)

| ID | Name | Ward | Diagnosis |
|---|---|---|---|
| ICU-001 | Adaeze Okonkwo | ICU Bay A / Bed 1 | Post-operative monitoring |
| ICU-002 | Emeka Nwosu | ICU Bay A / Bed 2 | Respiratory failure |
| ICU-003 | Fatima Bello | ICU Bay A / Bed 3 | Haemodynamic instability |
| ICU-004 | Kolade Martins | ICU Bay B / Bed 1 | Cardiac monitoring |
| ICU-005 | Ngozi Adeleke | ICU Bay B / Bed 2 | Post-resuscitation care |
| ICU-006 | Seun Dada | ICU Bay B / Bed 3 | Trauma observation |

---

## Setup

### Prerequisites

- Python 3.11+
- A MongoDB Atlas cluster (free tier works)
- An Anthropic API key (for AI note generation)

### 1. Clone and install dependencies

```bash
git clone https://github.com/john-ayodeji/HackathonBackend.git
cd HackathonBackend
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root (rename `_env` if present):

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGO_DB=pdi
ANTHROPIC_API_KEY=sk-ant-...
```

> ⚠️ **Security:** Never commit your `.env` file to version control. Add `.env` to your `.gitignore`. Keep your MongoDB credentials and Anthropic API key confidential — rotate them immediately if they are ever exposed.

### 3. Verify configuration

```bash
python test_config.py
```

This checks that all required environment variables are set, required packages are installed, and MongoDB is reachable.

### 4. Run the API

```bash
python app.py
```

The API starts on `http://localhost:5000`.

---

## API reference

All endpoints except `/health` and `/auth/login` require an `Authorization: Bearer <token>` header.

### Health

```
GET /health
```

Returns service version and MongoDB connectivity status.

---

### Authentication

```
POST /auth/login
Body: { "username": "amaka", "password": "nurse123" }
Returns: { "token": "...", "nurse": { nurse_id, name, role, username } }

POST /auth/logout
Authorization: Bearer <token>

GET /auth/me
Authorization: Bearer <token>
Returns: { "nurse": { nurse_id, name, role, username } }
```

---

### Patients

```
GET /patients
Returns all patients with their latest PDI score and risk level, sorted critical-first.

GET /patients/<patient_id>
Returns full patient record, all vital sign history, notes, and current assessment.

PATCH /patients/<patient_id>/meta
Body: { "age": "...", "weight": "...", "bed": "...", "ward": "...", "diagnosis": "..." }
```

---

### Vital signs

```
POST /patients/<patient_id>/vitals
Body: {
  "time": "16:00",   (optional — defaults to current time)
  "hr":   95,        (required)
  "rr":   20,        (required)
  "spo2": 93,        (required)
  "temp": 37.8,      (required)
  "sbp":  128,       (optional)
  "dbp":  84         (optional)
}
Returns: { "message": "Vitals logged", "time": "...", "assessment": { ... } }
```

---

### Assessments and notes

```
GET /patients/<patient_id>/assess
Returns the current PDI assessment (vitals, pdi score, alert) without logging anything.

POST /patients/<patient_id>/generate-note
Body: { "additional_observations": "..." }   (optional)
Generates a UMLS-grounded SOAP nursing note using AI, appends it to the patient record,
and returns the note alongside an updated assessment.

GET /thresholds
Returns the full clinical threshold configuration used by the PDI engine.
```

---

## Assessment response shape

```json
{
  "vitals": [
    {
      "vital": "hr",
      "label": "Heart Rate",
      "unit": "bpm",
      "current_value": 95,
      "current_risk": "ok",
      "projected_value": 110,
      "projected_risk": "crit",
      "slope_per_hour": 3.5,
      "r2": 0.92,
      "worst_risk": "crit"
    }
  ],
  "pdi": {
    "score": 65,
    "numeric_score": 58.2,
    "ai_boost": 6.8,
    "risk_level": "crit",
    "breakdown": { ... }
  },
  "alert": {
    "level": "crit",
    "pdi_score": 65,
    "triggered_by": [ ... ],
    "actions": ["Notify ICU Physician STAT", ...]
  }
}
```
