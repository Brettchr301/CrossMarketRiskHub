# quant-dashboard

React dashboard for the standalone quant API.

## Setup

```bash
npm install
```

## Run

```bash
npm run dev
```

## Desktop App Mode (Electron)

```bash
npm run app:dev
```

This opens the dashboard in a native desktop window while Vite runs locally.

If you already built static assets and want to start the desktop shell directly:

```bash
npm run app:start
```

Optional:

```bash
cp .env.example .env
```

Then set `VITE_API_BASE` if your API host differs from `http://localhost:8100`.
