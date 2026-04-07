from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT_SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
_spec = importlib.util.spec_from_file_location("codereviewenv_root_server", _ROOT_SERVER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load root server module: {_ROOT_SERVER_PATH}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

app = _module.app


def main() -> None:
    import os

    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
