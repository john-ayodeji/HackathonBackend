"""
PDI FastAPI — Nurse authentication, pre-seeded patients, neural-network note generation

Functionally equivalent to app.py (Flask) but built on FastAPI + uvicorn.
Swap the runner command:
  Flask  : python app.py          (or gunicorn app:app)
  FastAPI: uvicorn main:app       (or gunicorn -k uvicorn.workers.UvicornWorker main:app)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine import THRESHOLDS, run_full_assessment
from nlp import analyze_nursing_note, generate_nursing_note

from db import (
    append_note,
    append_vitals,
    authenticate_nurse,
    create_session,
    delete_session,
    get_nurse,
    get_patient,
    list_patients_summary,
    ping,
    resolve_session,
    seed_nurses,
    seed_patients,
    update_patient_meta,
)


# ── Seed data — nurses and patients loaded at startup ─────────────────────────

SEED_NURSE_LIST = [
    {"name": "Nurse Amaka Obi",    "username": "amaka",  "password": "nurse123", "role": "nurse"},
    {"name": "Nurse Tunde Adeyemi","username": "tunde",  "password": "nurse123", "role": "nurse"},
    {"name": "Nurse Chioma Eze",   "username": "chioma", "password": "nurse123", "role": "nurse"},
    {"name": "Admin",              "username": "admin",  "password": "admin123", "role": "admin"},
]

SEED_PATIENT_LIST = [
    {"patient_id": "ICU-001", "name": "Adaeze Okonkwo",  "age": "45", "weight": "68kg", "ward": "ICU Bay A", "bed": "Bed 1", "diagnosis": "Post-operative monitoring"},
    {"patient_id": "ICU-002", "name": "Emeka Nwosu",     "age": "62", "weight": "82kg", "ward": "ICU Bay A", "bed": "Bed 2", "diagnosis": "Respiratory failure"},
    {"patient_id": "ICU-003", "name": "Fatima Bello",    "age": "38", "weight": "61kg", "ward": "ICU Bay A", "bed": "Bed 3", "diagnosis": "Haemodynamic instability"},
    {"patient_id": "ICU-004", "name": "Kolade Martins",  "age": "55", "weight": "90kg", "ward": "ICU Bay B", "bed": "Bed 1", "diagnosis": "Cardiac monitoring"},
    {"patient_id": "ICU-005", "name": "Ngozi Adeleke",   "age": "71", "weight": "58kg", "ward": "ICU Bay B", "bed": "Bed 2", "diagnosis": "Post-resuscitation care"},
    {"patient_id": "ICU-006", "name": "Seun Dada",       "age": "29", "weight": "75kg", "ward": "ICU Bay B", "bed": "Bed 3", "diagnosis": "Trauma observation"},
]


# ── Lifespan — seed DB on startup ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_nurses(SEED_NURSE_LIST)
    seed_patients(SEED_PATIENT_LIST)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDI Backend",
    description="Predictive Deterioration Index — ICU nursing support API",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _resolve_token(authorization: Optional[str]) -> str:
    """
    Extract Bearer token, resolve to nurse_id.
    Raises HTTP 401 if the token is missing or invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized — please log in")
    token  = authorization.removeprefix("Bearer ").strip()
    nid    = resolve_session(token)
    if not nid:
        raise HTTPException(status_code=401, detail="Unauthorized — please log in")
    return nid


# ── Request / response models ─────────────────────────────────────────────────

class LoginBody(BaseModel):
    username: str
    password: str


class VitalsBody(BaseModel):
    time:  Optional[str] = None
    hr:    Optional[float] = None
    rr:    Optional[float] = None
    spo2:  Optional[float] = None
    temp:  Optional[float] = None
    sbp:   Optional[float] = None
    dbp:   Optional[float] = None


class GenerateNoteBody(BaseModel):
    additional_observations: Optional[str] = ""


class MetaBody(BaseModel):
    age:       Optional[str] = None
    weight:    Optional[str] = None
    bed:       Optional[str] = None
    ward:      Optional[str] = None
    diagnosis: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "PDI Backend", "version": "4.0.0", "mongodb": ping()}


# ── Nurse auth ────────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(body: LoginBody):
    """
    Nurse login by username + password.
    Returns: { "token": "...", "nurse": { name, role, nurse_id } }
    """
    username = (body.username or "").strip()
    password = (body.password or "").strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    nurse = authenticate_nurse(username, password)
    if not nurse:
        raise HTTPException(status_code=401, detail="Incorrect password — please try again")

    token = create_session(nurse["nurse_id"])
    return {
        "token": token,
        "nurse": {
            "nurse_id": nurse["nurse_id"],
            "name":     nurse["name"],
            "role":     nurse["role"],
            "username": nurse["username"],
        },
    }


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token:
        delete_session(token)
    return {"message": "Logged out"}


@app.get("/auth/me")
def me(authorization: Optional[str] = Header(default=None)):
    """Validate token and return nurse info."""
    nurse_id = _resolve_token(authorization)
    nurse = get_nurse(nurse_id)
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    return {"nurse": nurse}


# ── Ward census — all patients ────────────────────────────────────────────────

@app.get("/patients")
def list_patients(authorization: Optional[str] = Header(default=None)):
    """
    Return all pre-seeded patients with their latest risk assessment.
    Used by the Ward Census screen — nurses click a card to open a patient.
    """
    _resolve_token(authorization)

    patients = list_patients_summary()
    result = []
    for p in patients:
        assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
        result.append({
            "patient_id":    p["patient_id"],
            "name":          p["name"],
            "age":           p.get("age", ""),
            "ward":          p.get("ward", ""),
            "bed":           p.get("bed", ""),
            "diagnosis":     p.get("diagnosis", ""),
            "admitted":      p.get("admitted", ""),
            "latest_vitals": {k: v[-1] for k, v in p["vitals"].items() if v},
            "has_vitals":    any(v for v in p["vitals"].values()),
            "pdi_score":     assessment["pdi"]["score"],
            "risk_level":    assessment["pdi"]["risk_level"],
            "alert":         assessment["alert"],
        })

    # Sort by risk level — critical first
    order = {"crit": 0, "warn": 1, "ok": 2}
    result.sort(key=lambda x: order.get(x["risk_level"], 3))
    return result


# ── Individual patient ────────────────────────────────────────────────────────

@app.get("/patients/{patient_id}")
def get_patient_data(patient_id: str, authorization: Optional[str] = Header(default=None)):
    """Full patient record + current assessment for the selected patient."""
    _resolve_token(authorization)

    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
    return {
        "patient": {
            "patient_id": p["patient_id"], "name": p["name"],
            "age": p.get("age", ""),       "weight": p.get("weight", ""),
            "ward": p.get("ward", ""),     "bed": p.get("bed", ""),
            "diagnosis": p.get("diagnosis", ""), "admitted": p.get("admitted", ""),
        },
        "vitals":    p["vitals"],
        "times":     p["times"],
        "notes":     p.get("notes", []),
        "ai_weight": p.get("ai_weight", 0.0),
        "assessment": assessment,
    }


@app.patch("/patients/{patient_id}/meta")
def update_meta(
    patient_id: str,
    body: MetaBody,
    authorization: Optional[str] = Header(default=None),
):
    _resolve_token(authorization)
    update_patient_meta(patient_id, body.model_dump(exclude_none=True))
    return {"message": "Updated"}


# ── Vitals logging ────────────────────────────────────────────────────────────

@app.post("/patients/{patient_id}/vitals")
def log_vitals(
    patient_id: str,
    body: VitalsBody,
    authorization: Optional[str] = Header(default=None),
):
    """
    Log a new set of vital signs for the specified patient.
    Body: { "time": "16:00", "hr": 95, "rr": 20, "spo2": 93, "temp": 37.8, "sbp": 128, "dbp": 84 }
    """
    _resolve_token(authorization)

    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    errors: list[str] = []
    vitals: dict[str, float] = {}

    for key in ("hr", "rr", "spo2", "temp"):
        val = getattr(body, key, None)
        if val is None:
            errors.append(f"Missing: {key}")
        else:
            vitals[key] = float(val)

    for key in ("sbp", "dbp"):
        val = getattr(body, key, None)
        if val is not None:
            vitals[key] = float(val)

    if errors:
        raise HTTPException(status_code=400, detail={"error": "Validation failed", "details": errors})

    time_label = body.time or datetime.now().strftime("%H:%M")
    append_vitals(patient_id, vitals, time_label)

    p = get_patient(patient_id)
    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
    return {"message": "Vitals logged", "time": time_label, "assessment": assessment}


# ── Neural-network nursing note generation ────────────────────────────────────

@app.post("/patients/{patient_id}/generate-note")
def generate_note(
    patient_id: str,
    body: GenerateNoteBody = GenerateNoteBody(),
    authorization: Optional[str] = Header(default=None),
):
    """
    Auto-generate a UMLS-grounded nursing note from the patient's current vitals
    using the local neural network.  No external API calls are made.
    Optionally accepts: { "additional_observations": "..." } for supplementary context.
    """
    _resolve_token(authorization)

    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    if not any(v for v in p["vitals"].values()):
        raise HTTPException(
            status_code=400,
            detail="No vitals recorded yet — log vitals before generating a note",
        )

    extra      = (body.additional_observations or "").strip()
    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))

    patient_meta = {
        "name":      p["name"],
        "age":       p.get("age", ""),
        "weight":    p.get("weight", ""),
        "ward":      p.get("ward", ""),
        "bed":       p.get("bed", ""),
        "diagnosis": p.get("diagnosis", ""),
    }

    result = generate_nursing_note(p["vitals"], assessment, patient_meta)

    if extra and result.get("generated_note"):
        result["generated_note"] += f"\n\nAdditional nurse observations: {extra}"

    append_note(patient_id, result.get("generated_note", ""), result)
    p = get_patient(patient_id)
    updated_assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))

    return {
        "note":               result,
        "updated_assessment": updated_assessment,
    }


@app.get("/patients/{patient_id}/assess")
def assess(patient_id: str, authorization: Optional[str] = Header(default=None)):
    _resolve_token(authorization)
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))


@app.get("/thresholds")
def get_thresholds():
    return THRESHOLDS


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
