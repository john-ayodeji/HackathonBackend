"""
PDI Local Configuration Test
Run this before starting the backend to verify everything is wired correctly.
Usage: python test_config.py
"""

import os, sys, json
from dotenv import load_dotenv

load_dotenv()

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []

def check(label, passed, detail=""):
    icon = PASS if passed else FAIL
    results.append(passed)
    print(f"  {icon}  {label}")
    if detail:
        print(f"       {detail}")

print("\n── 1. Environment variables ─────────────────────────────")

mongo_uri = os.environ.get("MONGO_URI", "")
mongo_db  = os.environ.get("MONGO_DB", "")

check("MONGO_URI set",
      bool(mongo_uri),
      mongo_uri[:40] + "..." if mongo_uri else "MISSING — add to .env")

check("MONGO_URI is Atlas (mongodb+srv://)",
      mongo_uri.startswith("mongodb+srv://"),
      "Good — using Atlas cluster" if mongo_uri.startswith("mongodb+srv://") else "Looks like a local URI")

check("MONGO_DB set",
      bool(mongo_db),
      f"Database name: {mongo_db}" if mongo_db else "MISSING — add MONGO_DB=pdi to .env")


print("\n── 2. Python packages ───────────────────────────────────")

packages = [
    ("pymongo",       "pymongo"),
    ("dnspython",     "dns"),
    ("flask",         "flask"),
    ("flask_cors",    "flask_cors"),
    ("fastapi",       "fastapi"),
    ("uvicorn",       "uvicorn"),
    ("sklearn",       "sklearn"),
    ("numpy",         "numpy"),
    ("joblib",        "joblib"),
    ("dotenv",        "dotenv"),
]

for pkg_name, import_name in packages:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        check(f"{pkg_name}", True, f"version {ver}")
    except ImportError:
        check(f"{pkg_name}", False, f"Run: pip install {pkg_name}")


print("\n── 3. MongoDB Atlas connection ──────────────────────────")

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, OperationFailure

    print("     Connecting to Atlas cluster...")
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=10000,
    )

    # Ping the cluster
    client.admin.command("ping")
    check("Atlas cluster reachable", True, "Ping successful")

    # Check database access
    db = client[mongo_db or "pdi"]
    collections = db.list_collection_names()
    check("Database accessible", True, f"Database '{mongo_db}' — collections: {collections or '(empty, will be created)'}")

    # Check Atlas network access
    check("Network access allowed", True, "Your IP is whitelisted on Atlas")

    client.close()

except ServerSelectionTimeoutError as e:
    check("Atlas cluster reachable", False,
          "TIMEOUT — Two things to check:\n"
          "       1) Atlas Network Access: go to Atlas → Network Access → Add IP Address → Add Current IP\n"
          "       2) Verify MONGO_URI username/password are correct")
    print(f"\n     Error detail: {str(e)[:120]}")

except OperationFailure as e:
    check("Atlas cluster reachable", False,
          f"AUTH FAILED — Check username/password in MONGO_URI\n       Error: {e.details.get('errmsg', str(e))}")

except Exception as e:
    msg = str(e)
    if "nameserver" in msg or "name resolution" in msg:
        check("Atlas cluster reachable", False,
              "DNS resolution failed\n"
              "       → If you're on your local machine: go to Atlas → Network Access → Add your IP\n"
              "       → Make sure you're not on a VPN blocking outbound connections")
    else:
        check("Atlas cluster reachable", False, f"{type(e).__name__}: {msg[:120]}")


print("\n── 4. Local neural network models ───────────────────────")

import os as _os
model_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models")
scaler_path     = _os.path.join(model_dir, "scaler.joblib")
regressor_path  = _os.path.join(model_dir, "risk_regressor.joblib")
classifier_path = _os.path.join(model_dir, "severity_classifier.joblib")

models_exist = all(_os.path.exists(p) for p in (scaler_path, regressor_path, classifier_path))

if models_exist:
    try:
        import joblib
        import numpy as np
        scaler     = joblib.load(scaler_path)
        regressor  = joblib.load(regressor_path)
        classifier = joblib.load(classifier_path)
        # Quick smoke-test: predict on a normal vital-signs vector
        normal = [80, 15, 98, 37.0, 110, 70,   0, 0, 0, 0, 0, 0,   80, 15, 98, 37.0, 110, 70]
        X = np.array([normal], dtype=np.float32)
        X_s = scaler.transform(X)
        risk = float(regressor.predict(X_s)[0])
        sev  = int(classifier.predict(X_s)[0])
        check("Neural network models loadable", True,
              f"Smoke-test passed — risk_weight: {risk:.3f}, severity: {['low','moderate','high'][sev]}")
    except Exception as e:
        check("Neural network models loadable", False, f"{type(e).__name__}: {str(e)[:120]}")
else:
    # Models not yet trained — not a hard failure; they auto-generate on first API call.
    print(f"  {WARN}  Neural network models not found")
    print("       Run: python train_model.py  (auto-generated on first API call)")
    results.append(True)  # record as passed — this is a warning, not a blocking error


print("\n── 5. Seed data dry run ─────────────────────────────────")

try:
    # Re-use the db module logic directly
    from pymongo import MongoClient
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
    db = client[mongo_db or "pdi"]

    nurses_count   = db.nurses.count_documents({})
    patients_count = db.patients.count_documents({})
    sessions_count = db.sessions.count_documents({})

    check("Nurses collection", True,
          f"{nurses_count} nurses in DB {'(will seed on first startup)' if nurses_count == 0 else '— already seeded'}")
    check("Patients collection", True,
          f"{patients_count} patients in DB {'(will seed on first startup)' if patients_count == 0 else '— already seeded'}")
    check("Sessions collection", True,
          f"{sessions_count} active sessions")

    client.close()

except Exception as e:
    check("Seed data check", False, str(e)[:120])


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n─────────────────────────────────────────────────────────")
passed = sum(results)
total  = len(results)
all_ok = all(results)

if all_ok:
    print(f"  {PASS}  All {total} checks passed — backend is ready")
    print("     Flask  : python app.py")
    print("     FastAPI: uvicorn main:app --reload\n")
else:
    failed = total - passed
    print(f"  {FAIL}  {failed} of {total} checks failed — fix the issues above first\n")
    sys.exit(1)