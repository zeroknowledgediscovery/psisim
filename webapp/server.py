#!/usr/bin/env python3
"""PsiSim webapp HTTP server.

Run from the repository root:

  python3 webapp/server.py --host 127.0.0.1 --port 8000

The model is resolved (and, if needed, downloaded) once at startup. Every
browser session owns one resident native state (by default a propagation-only
``PropagationPsiState``); an answer is
applied with ``hard_observe(..., clamp=True)`` and its perturbation is then
propagated wave by wave (only newly induced deltas); every snapshot is
streamed to the browser as one NDJSON line.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine as eng

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


class NewSession(BaseModel):
    seed: int | None = Field(default=None, ge=0)
    dynamics: str = eng.DEFAULT_DYNAMICS


class Answer(BaseModel):
    column: int
    max_steps: int = eng.DEFAULT_MAX_STEPS
    empirical_n: int = eng.DEFAULT_EMPIRICAL_N  # finite-n dynamics only


def create_app(model: eng.Model | None = None) -> FastAPI:
    model = model or eng.Model(fetch=os.environ.get("PSISIM_NO_FETCH") != "1")
    engine = eng.Engine(model)
    model_json = json.dumps(model.describe(), separators=(",", ":"))

    app = FastAPI(title="PsiSim", docs_url=None, redoc_url=None)
    app.state.engine = engine

    def session_or_404(sid: str) -> eng.Session:
        s = engine.get(sid)
        if s is None:
            raise HTTPException(404, "session not found or expired")
        return s

    @app.get("/api/model")
    def get_model():
        return Response(
            model_json,
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "model": model.key,
            "model_path": str(model.path),
            "sessions": len(engine.sessions),
            "threads": eng.THREADS,
        }

    @app.post("/api/session")
    def new_session(body: NewSession | None = None):
        body = body or NewSession()
        try:
            s = engine.create(seed=body.seed, dynamics=body.dynamics)
        except eng.AnswerError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return engine.session_view(s)

    @app.get("/api/session/{sid}")
    def get_session(sid: str):
        return engine.session_view(session_or_404(sid))

    @app.delete("/api/session/{sid}")
    def delete_session(sid: str):
        engine.drop(sid)
        return {"ok": True}

    @app.get("/api/session/{sid}/export")
    def export_session(sid: str):
        data = engine.export(session_or_404(sid))
        return JSONResponse(
            data,
            headers={
                "Content-Disposition": f'attachment; filename="psisim_{sid}.json"'
            },
        )

    @app.post("/api/session/{sid}/answer")
    def answer(sid: str, body: Answer):
        s = session_or_404(sid)
        try:
            frames = engine.answer(
                s,
                body.column,
                max_steps=body.max_steps,
                empirical_n=body.empirical_n,
            )
        except eng.AnswerError as exc:
            raise HTTPException(409, str(exc)) from exc

        def lines():
            for item in frames:
                yield json.dumps(item, separators=(",", ":")) + "\n"

        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("PSISIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    import uvicorn

    print("loading model, Psi0, dependency graph and layout ...", flush=True)
    app = create_app()
    m = app.state.engine.model
    print(
        f"ready: {m.key} at {m.path} "
        f"({m.width} columns, {len(m.edges)} learned links, {m.load_seconds:.1f}s)",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
