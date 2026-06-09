# LEGO Digital Twin / MINIVERSE

MINIVERSE is an exhibition system where visitors photograph a LEGO character and object, generate 3D assets through Tripo, and display the results inside a Unity VR plaza.

## Team

| Name | Affiliation | Role |
| --- | --- | --- |
| Yejin Kim | Chung-Ang University | Project Manager / Team Leader |
| Jinseo Kim | Chung-Ang University | Software Engineer / Lead Developer |
| Yumin Kim | Chung-Ang University | Unity Developer / Lead Designer |
| Seoyeon Lee | Chung-Ang University | Animator / Sub Project Manager |

## Development Roles

- System Architect, Backend, Frontend, System Automation: Jinseo Kim
- Unity Design, Unity Script: Yumin Kim

## Development Environment

- Linux or WSL for backend, workers, and frontend
- Windows for running the Unity project
- Python environment at `/home/jskim/anaconda3/envs/triposr`
- Node.js and npm
- Docker Desktop or Docker Engine for PostgreSQL and Redis

## What This Repository Contains

- `apps/backend`
  FastAPI API server, PostgreSQL models, Redis/RQ worker tasks, Tripo integration, Unity-facing APIs
- `apps/frontend`
  React + Vite tablet/kiosk capture app, plus the admin dashboard
- `unity-client`
  Unity VR client that loads the current visitor session and the accumulated plaza data
- `scripts`
  Local automation helpers such as IP sync and stack startup
- `history`
  team notes, troubleshooting logs, and internal implementation records

## System Overview

The project is split into persistent storage, async processing, and runtime clients.

- `FastAPI`
  Owns session APIs, uploads, profile updates, Unity data endpoints, and WebSocket broadcast
- `PostgreSQL`
  Stores durable exhibition data such as sessions, likes, `bubble_text`, signature motions, and generated asset URLs
- `Redis`
  Backs the RQ job queues, the Unity waiting queue, and realtime pub/sub events
- `RQ workers`
  Run heavy Tripo generation jobs in the background
- `Unity`
  Polls the active session, loads runtime models, and builds the plaza from stored visitor data

## Runtime Data Flow

1. The frontend creates a visitor session with `POST /sessions`.
2. The visitor uploads multi-view images for character and object assets.
3. FastAPI stores images under `apps/backend/storage/images/...` and enqueues background jobs.
4. RQ workers call Tripo, download generated files, and store outputs under `apps/backend/storage/models/...`.
5. Generated asset URLs are saved into PostgreSQL.
6. When a session is fully ready, the backend adds its session id to the Redis list `lego:unity_queue`.
7. Unity fetches the current active session with `/sessions/active` and `/unity/sessions/{id}`.
8. When Unity enters plaza mode, it loads previous visitor data from `/unity/plaza/sessions`.

## Queues and Realtime Channels

RQ queues:

- `lego-character`
- `lego-object`

Redis list:

- `lego:unity_queue`
  ordered list of fully completed sessions that Unity should consume

Redis pub/sub channel:

- `lego:events`
  broadcasts `session_ready` and `likes_updated`

## Current Execution Model

This repository currently uses Docker Compose for infrastructure only:

- PostgreSQL
- Redis

The application processes themselves run locally:

- FastAPI via the `triposr` Python environment
- RQ workers via `run_worker.py`
- frontend via Vite
- admin dashboard via a second Vite entry on port `3001`

This matches the current codebase better than the old full-container setup, because the repo no longer contains backend/frontend Dockerfiles.

## Environment Variables

Main backend configuration lives in:

- `apps/backend/.env`
- `apps/frontend/.env.local`

Important variables:

- `DATABASE_URL`
- `REDIS_URL`
- `TRIPO_API_KEY`
- `BACKEND_HOST`
- `VITE_API_URL`

## Quick Start

### 1. One-command startup

```bash
bash scripts/start_exhibition.sh <Windows-IP>
```

This automation script performs the full startup flow:

- updates `apps/frontend/.env.local` and `apps/backend/.env`
- updates Unity `StreamingAssets/Config/server_config.json`
- optionally updates DB asset URLs through `scripts/update_network_ip.py`
- starts `db` and `redis` via Docker Compose
- starts the backend, worker supervisor, frontend, and admin dashboard

It will prompt for the Windows IP if you omit `<Windows-IP>`.

Before running it, update Windows portproxy manually in an Administrator PowerShell using the current WSL IP.

Use this as the default exhibition startup command. It already includes the IP update step internally, so you do not need to run `scripts/update_network_ip.py` separately beforehand.

### 2. Manual step-by-step startup

```bash
docker compose up -d db redis
python scripts/update_network_ip.py <Windows-IP>
bash scripts/run_exhibition.sh
```

Use this manual path only when you want to update IP-sensitive config without immediately starting the whole stack, or when you need to start each part separately for debugging.

This starts:

- backend on `:8000`
- worker supervisor
- frontend on `:3000`
- admin dashboard on `:3001`

### 3. Open the UIs

- frontend: `http://localhost:3000`
- admin: `http://localhost:3001`
- backend health: `http://localhost:8000/health`

## Manual Start Commands

If you do not want to use the startup script:

### Backend

```bash
/home/jskim/anaconda3/envs/triposr/bin/python -m uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000
```

### Workers

```bash
/home/jskim/anaconda3/envs/triposr/bin/python run_worker.py
```

### Frontend

```bash
cd apps/frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

### Admin Dashboard

```bash
cd apps/frontend
npm run dev:admin -- --host 0.0.0.0 --port 3001
```

## Unity Notes

- Unity should be run from `unity-client/AURA_Lego`
- The Unity client consumes:
  - `/sessions/active`
  - `/unity/sessions/{session_id}`
  - `/unity/plaza/sessions`
  - WebSocket like updates
- Plaza data includes:
  - generated character and object models
  - likes
  - `bubble_text`
  - signature motions

## Admin / Exhibition Operations

The admin dashboard is intended for live exhibition operation and debugging.

Typical tasks:

- inspect session queue state
- inspect worker state
- inspect Redis / backend health
- cancel bad visitor sessions
- remove sessions from the Unity queue
- reset exhibition state when needed

## Important Scripts

- `scripts/start_exhibition.sh`
  default exhibition startup entrypoint; runs IP update + Docker + backend/worker/frontend/admin
- `scripts/run_exhibition.sh`
  process startup helper for backend, workers, frontend, and admin only
- `scripts/update_network_ip.py`
  maintenance helper for updating IP-sensitive local config when the Windows IP changes
- `run_worker.py`
  starts and supervises the two RQ workers

## Important Paths

- `apps/backend/storage/images`
  uploaded visitor images
- `apps/backend/storage/models`
  generated Tripo outputs
- `history/network_situation.txt`
  Windows/WSL networking notes for mobile or tablet access

## Cleanup Notes

Legacy paths that used to describe an older Tripo-only prototype have been removed from the active architecture. The current source of truth is:

- `apps/backend` for the live backend
- `apps/frontend` for the live web apps
- `unity-client` for the VR client

## Development Guidance

- Treat PostgreSQL as the durable record of sessions and asset metadata
- Treat Redis as transient orchestration state
- Treat RQ workers as the only place that should perform heavy Tripo generation
- Treat `SessionManager` in Unity as the write entry point for visitor-facing session edits such as `bubble_text`
