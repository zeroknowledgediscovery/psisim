#!/usr/bin/env python3
"""PsiSim webapp HTTP server.

Run from the repository root:

  python3 webapp/server.py --host 127.0.0.1 --port 8000

The model is resolved (and, if needed, downloaded) once at startup. Every
browser session owns one resident native state of the survey model it was
started with (GSS 2018 by default; other countries/years are downloaded from
the public DTAG model release on demand); an answer is
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
    model: str | None = None   # catalog key, e.g. "afrobarometer/r7"; default GSS 2018


class Answer(BaseModel):
    column: int
    max_steps: int = eng.DEFAULT_MAX_STEPS
    empirical_n: int = eng.DEFAULT_EMPIRICAL_N  # finite-n dynamics only


def create_app(registry: eng.ModelRegistry | None = None) -> FastAPI:
    registry = registry or eng.ModelRegistry(fetch=os.environ.get("PSISIM_NO_FETCH") != "1")
    engine = eng.Engine(registry)
    described: dict[str, str] = {}

    app = FastAPI(title="PsiSim", docs_url=None, redoc_url=None)
    app.state.engine = engine

    def session_or_404(sid: str) -> eng.Session:
        s = engine.get(sid)
        if s is None:
            raise HTTPException(404, "session not found or expired")
        return s

    def model_key(family: str, name: str) -> str:
        try:
            return registry.validate(f"{family}/{name}")
        except eng.ModelError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/catalog")
    def catalog():
        """Every survey model of the public release: family, period, countries."""
        return registry.catalog()

    @app.get("/api/model")
    def get_model(key: str = eng.DEFAULT_MODEL):
        """Items, answer categories and Psi0 of a loaded model."""
        try:
            model = registry.get(registry.validate(key))
        except eng.ModelError as exc:
            raise HTTPException(409, str(exc)) from exc
        if key not in described:
            described[key] = json.dumps(model.describe(), separators=(",", ":"))
        return Response(described[key], media_type="application/json",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/models/{family}/{name}/status")
    def model_status(family: str, name: str):
        return registry.status(model_key(family, name))

    @app.post("/api/models/{family}/{name}/load")
    def load_model(family: str, name: str):
        """Download from the public release if needed, then compute Psi0."""
        return registry.start(model_key(family, name))

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "default_model": eng.DEFAULT_MODEL,
            "model_root": str(eng.model_root()),
            "sessions": len(engine.sessions),
            "threads": eng.THREADS,
        }

    @app.post("/api/session")
    def new_session(body: NewSession | None = None):
        body = body or NewSession()
        try:
            s = engine.create(seed=body.seed, dynamics=body.dynamics, model_key=body.model)
        except eng.ModelError as exc:
            raise HTTPException(409, str(exc)) from exc
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

    print("loading the default model and its Psi0 ...", flush=True)
    app = create_app()
    m = app.state.engine.default
    print(
        f"ready: {m.key} at {m.path} "
        f"({m.width} columns, {m.load_seconds:.1f}s); "
        f"{len(eng.CATALOG.get('models', {}))} models available on demand",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
