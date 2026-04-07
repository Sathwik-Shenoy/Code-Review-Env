import json
from pathlib import Path


def write_admin_audit(event, user_input, audit_path):
    payload = {
        "event": event,
        "message": user_input,
    }

    line = json.dumps(payload)
    path = Path(audit_path)

    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def replay_audit(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]
