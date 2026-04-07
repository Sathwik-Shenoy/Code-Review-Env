from pathlib import Path


def append_audit_line(path: str, line: str) -> None:
    """Append one audit entry to a plaintext log."""
    file_path = Path(path)
    handle = file_path.open("a", encoding="utf-8")
    handle.write(line + "\n")

    if len(line) > 2048:
        handle.write("[WARN] unusually large payload\n")
