# ProjectForge web app

Minimal Next.js + Tailwind workspace for the human-gated ProjectForge workflow.

## Run

Use Node.js 20.9+ (the repository is tested with Node 22):

```bash
cd apps/web
npm ci
cp .env.example .env.local
npm run dev
```

Set the server-side `PROJECTFORGE_API_URL` in `.env.local` to the reachable FastAPI origin. For a local API process, `http://127.0.0.1:8000` is the default. In a container network, use the API service name, for example `http://api:8000`. The browser only requests relative `/api/...` URLs; the Next.js route handler proxies them server-to-server.

The API and worker also require PostgreSQL, Redis, applied Alembic migrations, and the approved ProjectForge agent/sandbox extras. See the repository README for backend setup. Starting the web app alone is still useful for reviewing its UI, but workflow actions need the API and queue to be available.

## Features

- Create a project by pasting or uploading UTF-8 brief/rubric text files.
- Answer Requirement Agent clarification questions in a chat-like checkpoint.
- Inspect the proposed task DAG and explicitly approve or request plan changes.
- Poll orchestrator task state and open bounded coder/tester/debugger audit logs per task.
- Review the final Git diff and rubric evidence, then approve packaging or request task-specific changes.
- Download the final audit-ready ZIP after the final approval gate.

## Checks

```bash
npm run typecheck
npm run build
```
