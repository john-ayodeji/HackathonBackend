"""
PDI NLP Layer — Local neural network for risk scoring + UMLS template-based note generation

Replaces the Anthropic/Claude API with a locally trained MLP neural network
(see train_model.py) for offline, cost-free inference.  The network predicts:

  • risk_weight  — continuous float 0.0–1.0   (MLPRegressor)
  • severity     — low / moderate / high        (MLPClassifier)

SOAP nursing notes are generated from UMLS-grounded clinical templates rather
than a large language model, keeping the output deterministic and explainable.

If the trained model artefacts are not yet present the module automatically
runs training (train_model.py) on first import so the API is always usable.
"""

import os

import joblib
import numpy as np

# ── UMLS concept clusters used for clinical language grounding ────────────────
# These map vital sign deviations to standardised UMLS clinical descriptors
# so generated notes use consistent, medically accurate terminology.

UMLS_VITAL_CONCEPTS = {
    "hr": {
        "high": {
            "umls_concept": "C0039231",  # Tachycardia
            "descriptors": ["tachycardic", "elevated heart rate", "increased pulse rate"],
            "clinical_concern": "possible sympathetic activation, pain, haemorrhage, or early sepsis",
        },
        "low": {
            "umls_concept": "C0004610",  # Bradycardia
            "descriptors": ["bradycardic", "decreased heart rate", "slow pulse"],
            "clinical_concern": "possible vagal response, medication effect, or cardiac conduction issue",
        },
    },
    "rr": {
        "high": {
            "umls_concept": "C0231835",  # Tachypnoea
            "descriptors": ["tachypnoeic", "increased respiratory rate", "rapid breathing"],
            "clinical_concern": "possible respiratory compromise, metabolic acidosis, or pain",
        },
        "low": {
            "umls_concept": "C0700067",  # Bradypnoea
            "descriptors": ["bradypnoeic", "decreased respiratory rate", "slow respirations"],
            "clinical_concern": "possible sedation effect, CNS depression, or fatigue",
        },
    },
    "spo2": {
        "low": {
            "umls_concept": "C0242184",  # Hypoxaemia
            "descriptors": ["hypoxaemic", "decreased oxygen saturation", "desaturating"],
            "clinical_concern": "possible respiratory failure, airway compromise, or V/Q mismatch",
        },
    },
    "temp": {
        "high": {
            "umls_concept": "C0015967",  # Fever
            "descriptors": ["febrile", "pyrexial", "elevated temperature"],
            "clinical_concern": "possible infection, inflammatory process, or drug reaction",
        },
        "low": {
            "umls_concept": "C0020672",  # Hypothermia
            "descriptors": ["hypothermic", "low body temperature", "subnormal temperature"],
            "clinical_concern": "possible environmental exposure, septic shock, or metabolic disorder",
        },
    },
    "sbp": {
        "high": {
            "umls_concept": "C0020538",  # Hypertension
            "descriptors": ["hypertensive", "elevated systolic pressure"],
            "clinical_concern": "possible pain, anxiety, or hypertensive crisis",
        },
        "low": {
            "umls_concept": "C0020649",  # Hypotension
            "descriptors": ["hypotensive", "decreased systolic pressure"],
            "clinical_concern": "possible haemodynamic compromise, haemorrhage, or distributive shock",
        },
    },
    "dbp": {
        "high": {
            "umls_concept": "C0020538",
            "descriptors": ["elevated diastolic pressure"],
            "clinical_concern": "possible hypertensive state",
        },
        "low": {
            "umls_concept": "C0020649",
            "descriptors": ["decreased diastolic pressure", "widened pulse pressure"],
            "clinical_concern": "possible septic or distributive shock",
        },
    },
}

THRESHOLDS_NORMAL = {
    "hr":   {"low": 60,   "high": 100,  "unit": "bpm"},
    "rr":   {"low": 12,   "high": 18,   "unit": "/min"},
    "spo2": {"low": 95,   "high": 100,  "unit": "%"},
    "temp": {"low": 36.5, "high": 37.3, "unit": "°C"},
    "sbp":  {"low": 90,   "high": 120,  "unit": "mmHg"},
    "dbp":  {"low": 60,   "high": 80,   "unit": "mmHg"},
}

VITAL_KEYS    = ["hr", "rr", "spo2", "temp", "sbp", "dbp"]
SEVERITY_LABELS = ["low", "moderate", "high"]

# ── Model registry — lazy-loaded on first inference call ──────────────────────
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
_scaler     = None
_regressor  = None
_classifier = None


def _ensure_models() -> bool:
    """
    Lazy-load trained model artefacts.  If they do not exist, run the trainer
    automatically so the API is always functional without a manual setup step.
    Returns True when models are available.
    """
    global _scaler, _regressor, _classifier
    if _scaler is not None:
        return True

    scaler_path     = os.path.join(_MODEL_DIR, "scaler.joblib")
    regressor_path  = os.path.join(_MODEL_DIR, "risk_regressor.joblib")
    classifier_path = os.path.join(_MODEL_DIR, "severity_classifier.joblib")

    if not all(os.path.exists(p) for p in (scaler_path, regressor_path, classifier_path)):
        print("[nlp] Model artefacts not found — running train_model.py …")
        try:
            from train_model import train_and_save  # noqa: PLC0415
            train_and_save(output_dir=_MODEL_DIR)
        except Exception as exc:
            print(f"[nlp] Auto-training failed: {exc}")
            return False

    try:
        _scaler     = joblib.load(scaler_path)
        _regressor  = joblib.load(regressor_path)
        _classifier = joblib.load(classifier_path)
        print("[nlp] Neural network models loaded.")
        return True
    except Exception as exc:
        print(f"[nlp] Failed to load models: {exc}")
        return False


def _extract_features(vitals: dict, assessment: dict) -> list:
    """
    Build the 18-element feature vector used by the neural network:
      [current_values×6, slopes×6, projections×6]
    Missing vitals are imputed with the midpoint of their normal range.
    """
    vital_assessments = {v["vital"]: v for v in assessment.get("vitals", [])}
    values, slopes, projections = [], [], []

    for key in VITAL_KEYS:
        thresh = THRESHOLDS_NORMAL[key]
        normal_mid = (thresh["low"] + thresh["high"]) / 2.0
        readings = vitals.get(key, [])

        if readings:
            current = float(readings[-1])
            va      = vital_assessments.get(key, {})
            slope   = float(va.get("slope_per_hour", 0.0))
            proj    = float(va.get("projected_value", current))
        else:
            current = normal_mid
            slope   = 0.0
            proj    = normal_mid

        values.append(current)
        slopes.append(slope)
        projections.append(proj)

    return values + slopes + projections


def _nn_predict(vitals: dict, assessment: dict) -> tuple:
    """
    Run neural-network inference.
    Returns (risk_weight: float, severity: str).
    Falls back to rule-based scoring when models are unavailable.
    """
    if not _ensure_models():
        return _fallback_risk(vitals, assessment)

    features = _extract_features(vitals, assessment)
    X = np.array([features], dtype=np.float32)
    X_s = _scaler.transform(X)

    risk_weight = float(np.clip(_regressor.predict(X_s)[0], 0.0, 1.0))
    severity_idx = int(_classifier.predict(X_s)[0])
    severity = SEVERITY_LABELS[min(severity_idx, 2)]
    return risk_weight, severity


def _fallback_risk(vitals: dict, assessment: dict) -> tuple:
    """
    Rule-based fallback when NN models are unavailable.
    Mirrors the deviation logic from train_model.py.
    """
    from train_model import compute_risk_weight  # noqa: PLC0415
    features = _extract_features(vitals, assessment)
    values      = features[0:6]
    slopes      = features[6:12]
    projections = features[12:18]
    risk_weight = compute_risk_weight(values, slopes, projections)
    severity = "low" if risk_weight < 0.2 else ("moderate" if risk_weight < 0.5 else "high")
    return risk_weight, severity


# ── Recommended actions ───────────────────────────────────────────────────────

_RECOMMENDED_ACTIONS = {
    "high": [
        "Notify ICU physician STAT",
        "Increase monitoring frequency to every 15 minutes",
        "Draw blood cultures and urgent bloods",
        "Prepare resuscitation equipment",
        "Escalate via rapid-response protocol",
    ],
    "moderate": [
        "Notify senior nurse and document concerns",
        "Increase monitoring to every 30 minutes",
        "Review recent nursing notes and medication chart",
        "Prepare documentation for escalation if trend continues",
    ],
    "low": [
        "Continue routine monitoring every 2 hours",
        "Document observations in patient record",
    ],
}


def _get_recommended_actions(severity: str, flagged_terms: list) -> list:
    actions = list(_RECOMMENDED_ACTIONS.get(severity, _RECOMMENDED_ACTIONS["low"]))
    if "hypotensive" in flagged_terms or "decreased systolic pressure" in flagged_terms:
        actions.insert(0, "Assess fluid status — consider IV fluid challenge per protocol")
    if "hypoxaemic" in flagged_terms or "desaturating" in flagged_terms:
        actions.insert(0, "Apply supplemental oxygen and assess airway patency")
    return actions


def _build_reasoning(flagged_terms: list, severity: str, pdi_score) -> str:
    severity_map = {
        "low":      "is stable and within acceptable parameters",
        "moderate": "warrants increased monitoring and senior nurse review",
        "high":     "indicates high risk of imminent deterioration requiring immediate escalation",
    }
    if not flagged_terms:
        return (f"All vital signs are within normal limits; PDI score {pdi_score}/100 "
                f"indicates a clinically stable condition.")
    terms_str = ", ".join(flagged_terms[:3])
    return (f"Presence of {terms_str} with PDI score {pdi_score}/100 "
            f"{severity_map.get(severity, 'requires clinical review')}.")


# ── SOAP note template builder ────────────────────────────────────────────────

def _build_soap_note(
    vitals: dict,
    assessment: dict,
    patient_meta: dict,
    risk_weight: float,
    severity: str,
) -> dict:
    """
    Construct a structured SOAP nursing note using UMLS-aligned clinical
    templates.  All clinical terminology is grounded in UMLS concept codes.
    """
    vital_assessments = {v["vital"]: v for v in assessment.get("vitals", [])}
    pdi               = assessment.get("pdi", {})
    pdi_score         = pdi.get("score", "N/A")

    name      = patient_meta.get("name", "Patient")
    age       = patient_meta.get("age", "N/A")
    ward      = patient_meta.get("ward", "ICU")
    bed       = patient_meta.get("bed", "N/A")
    diagnosis = patient_meta.get("diagnosis", "under observation")

    flagged_terms: list[str] = []
    abnormal_lines: list[str] = []
    normal_lines:   list[str] = []

    for key in VITAL_KEYS:
        readings = vitals.get(key, [])
        if not readings:
            continue

        thresh  = THRESHOLDS_NORMAL[key]
        va      = vital_assessments.get(key, {})
        current = readings[-1]
        slope   = va.get("slope_per_hour", 0)
        proj    = va.get("projected_value", current)

        if current > thresh["high"]:
            direction = "high"
        elif current < thresh["low"]:
            direction = "low"
        else:
            direction = None

        if direction:
            concepts   = UMLS_VITAL_CONCEPTS.get(key, {}).get(direction, {})
            descriptor = concepts.get("descriptors", [key])[0]
            concern    = concepts.get("clinical_concern", "")
            umls_code  = concepts.get("umls_concept", "")

            trend = ("worsening"
                     if (direction == "high" and slope > 0) or (direction == "low" and slope < 0)
                     else "stable")

            line = (f"{key.upper()} {current} {thresh['unit']} — {descriptor} "
                    f"[UMLS:{umls_code}], trend: {trend}; "
                    f"projected {proj} {thresh['unit']} in 4 h")
            if concern:
                line += f"; concern: {concern}"

            flagged_terms.append(descriptor)
            abnormal_lines.append(line)
        else:
            normal_lines.append(
                f"{key.upper()} {current} {thresh['unit']} — within normal limits; "
                f"projected {proj} {thresh['unit']} in 4 h"
            )

    # ── S ─────────────────────────────────────────────────────────────────────
    s_section = (
        f"S: {name}, {age} years old, admitted to {ward} (Bed {bed}) "
        f"for {diagnosis}. Nursing assessment performed."
    )

    # ── O ─────────────────────────────────────────────────────────────────────
    o_lines = ["O: Current vital signs with 4-hour trend analysis:"]
    for line in abnormal_lines + normal_lines:
        o_lines.append(f"   {line}")
    o_lines.append(
        f"   PDI Risk Score: {pdi_score}/100 "
        f"({pdi.get('risk_level', 'ok').upper()}) — "
        f"neural network risk weight: {risk_weight:.2f}"
    )
    o_section = "\n".join(o_lines)

    # ── A ─────────────────────────────────────────────────────────────────────
    severity_desc = {
        "low":      "low clinical risk; patient is haemodynamically stable",
        "moderate": "moderate risk of physiological deterioration",
        "high":     "high risk of imminent deterioration",
    }
    a_section = f"A: Patient assessment indicates {severity_desc.get(severity, severity)}. "
    if flagged_terms:
        a_section += f"Clinical concerns identified: {', '.join(flagged_terms)}."
    else:
        a_section += "No significant clinical concerns identified at this time."

    # ── P ─────────────────────────────────────────────────────────────────────
    actions = _get_recommended_actions(severity, flagged_terms)
    p_section = "P: " + "; ".join(actions[:4]) + "."

    soap_note = "\n\n".join([s_section, o_section, a_section, p_section])
    reasoning = _build_reasoning(flagged_terms, severity, pdi_score)

    return {
        "generated_note":      soap_note,
        "flagged_terms":       flagged_terms,
        "risk_weight":         round(risk_weight, 4),
        "severity":            severity,
        "reasoning":           reasoning,
        "recommended_actions": actions,
    }


def generate_nursing_note(vitals: dict, assessment: dict, patient_meta: dict) -> dict:
    """
    Generate a UMLS-grounded SOAP nursing note using the local neural network
    for risk scoring and clinical templates for text generation.
    No external API calls are made.
    """
    if not vitals or not any(v for v in vitals.values()):
        return {
            "generated_note": "Insufficient vital signs data to generate note.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "No vital signs recorded.",
            "recommended_actions": [],
        }

    try:
        risk_weight, severity = _nn_predict(vitals, assessment)
        return _build_soap_note(vitals, assessment, patient_meta, risk_weight, severity)
    except Exception:
        return {
            "generated_note": "Note generation failed — manual documentation required.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "Automated note generation encountered an internal error. Clinical review required.",
            "recommended_actions": ["Manual clinical review required"],
            "error": True,
        }


def analyze_nursing_note(note: str) -> dict:
    """
    Analyse a manually entered nursing note for clinical risk markers using
    UMLS-aligned keyword matching.  No external API calls are made.
    """
    if not note or not note.strip():
        return {
            "generated_note": "",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "No note provided.",
            "recommended_actions": [],
        }

    # Ordered by clinical severity weight — longer phrases matched first to
    # avoid partial-match issues (e.g. "decreased systolic" before "systolic").
    TERM_WEIGHTS: dict[str, float] = {
        # Oxygenation — highest weight
        "hypoxaemic": 0.18, "hypoxemic": 0.18,
        "desaturating": 0.18, "desaturation": 0.18,
        "decreased oxygen saturation": 0.18,
        # Haemodynamic compromise
        "hypotensive": 0.16, "hypotension": 0.16,
        "decreased systolic pressure": 0.14,
        # General deterioration markers
        "deteriorating": 0.20, "deterioration": 0.20,
        "imminent": 0.20, "escalation": 0.15,
        "critical": 0.20, "emergent": 0.20,
        "unstable": 0.18, "worsening": 0.15, "urgent": 0.14,
        # Cardiac
        "tachycardic": 0.14, "tachycardia": 0.14,
        "elevated heart rate": 0.14, "increased pulse rate": 0.14,
        "bradycardic": 0.14, "bradycardia": 0.14,
        "decreased heart rate": 0.14, "slow pulse": 0.14,
        # Respiratory
        "tachypnoeic": 0.12, "tachypneic": 0.12,
        "increased respiratory rate": 0.12,
        "bradypnoeic": 0.12, "bradypneic": 0.12,
        # Temperature
        "febrile": 0.10, "pyrexial": 0.10, "pyrexia": 0.10,
        "elevated temperature": 0.10,
        "hypothermic": 0.10, "hypothermia": 0.10,
        # Blood pressure (hypertensive side — lower weight)
        "hypertensive": 0.09, "hypertension": 0.09,
        "elevated systolic": 0.09,
    }

    note_lower = note.lower()
    flagged_terms: list[str] = []
    risk_score = 0.0

    # TERM_WEIGHTS keys are unique so each term is matched at most once.
    for term, weight in TERM_WEIGHTS.items():
        if term in note_lower:
            flagged_terms.append(term)
            risk_score = min(1.0, risk_score + weight)

    risk_weight = min(1.0, risk_score)
    severity = "low" if risk_weight < 0.2 else ("moderate" if risk_weight < 0.5 else "high")
    actions   = _get_recommended_actions(severity, flagged_terms)

    if not flagged_terms:
        reasoning = "No UMLS-aligned clinical risk markers detected in the nursing note."
    else:
        terms_str = ", ".join(flagged_terms[:3])
        severity_map = {
            "low":      "requires routine monitoring",
            "moderate": "warrants increased monitoring and senior nurse review",
            "high":     "indicates elevated deterioration risk requiring immediate clinical review",
        }
        reasoning = (f"Detected clinical markers: {terms_str}. "
                     f"Risk level {severity_map.get(severity, 'requires clinical review')}.")

    return {
        "generated_note":      note.strip(),
        "flagged_terms":       flagged_terms,
        "risk_weight":         round(risk_weight, 4),
        "severity":            severity,
        "reasoning":           reasoning,
        "recommended_actions": actions,
    }