# aegra-cli

Aegra CLI - Command-line interface for managing self-hosted agent deployments.

Aegra is an open-source, self-hosted alternative to LangSmith Deployments. Use this CLI to initialize projects, run development servers, and manage Docker services.

## Installation

### From PyPI

```bash
pip install aegra-cli
```

### From Source

```bash
# Clone the repository
git clone https://github.com/aegra/aegra.git
cd aegra

# Install all workspace packages
uv sync --all-packages
```

## Quick Start

```bash
# Initialize a new Aegra project (prompts for location, template, and name)
aegra init

# Follow the printed next steps
cd <your-project>
cp .env.example .env     # Add your OPENAI_API_KEY
uv sync                  # Install dependencies
uv run aegra dev         # Start PostgreSQL + dev server
```

## Commands

### `aegra version`

Show version information for aegra-cli and aegra-api.

```bash
aegra version
```

Output displays a table with versions for both packages.

---

### `aegra init`

Initialize a new Aegra project with configuration files and directory structure.

```bash
aegra init [PATH] [OPTIONS]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `PATH` | `.` | Project directory to initialize |

**Options:**

| Option | Description |
|--------|-------------|
| `-t, --template INT` | Template number (1 = "New Aegra Project", 2 = "ReAct Agent") |
| `-n, --name STR` | Project name |
| `--force` | Overwrite existing files if they exist |

**Templates:**

| # | Name | Description |
|---|------|-------------|
| 1 | New Aegra Project | Simple chatbot with basic graph structure |
| 2 | ReAct Agent | Tool-calling agent with a tools module |

**Examples:**

```bash
# Initialize in current directory
aegra init

# Initialize in a specific directory
aegra init my-project

# Initialize with a specific template and name
aegra init my-project --template 2 --name "My ReAct Agent"

# Initialize in current directory, overwriting existing files
aegra init --force
```

**Created Files:**

- `aegra.json` - Graph configuration
- `pyproject.toml` - Python project configuration
- `.env.example` - Environment variable template
- `.gitignore` - Git ignore rules
- `README.md` - Project readme
- `src/<slug>/__init__.py` - Package init file
- `src/<slug>/graph.py` - Graph implementation
- `src/<slug>/state.py` - State schema definition
- `src/<slug>/prompts.py` - Prompt templates
- `src/<slug>/context.py` - Context configuration
- `src/<slug>/utils.py` - Utility functions
- `src/<slug>/tools.py` - Tool definitions (ReAct template only)
- `docker-compose.yml` - Docker Compose for PostgreSQL and API services
- `Dockerfile` - Container build

---

### `aegra dev`

Run the development server with hot reload.

```bash
aegra dev [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `127.0.0.1` | Host to bind the server to |
| `--port PORT` | `2026` | Port to bind the server to |
| `--app APP` | `aegra_api.main:app` | Application import path |
| `-c, --config PATH` | | Path to aegra.json config file |
| `-e, --env-file PATH` | | Path to .env file |
| `-f, --file PATH` | | Path to docker-compose.yml file |
| `--no-db-check` | | Skip the automatic database check |

**Examples:**

```bash
# Start with defaults (localhost:2026)
aegra dev

# Start on all interfaces, port 3000
aegra dev --host 0.0.0.0 --port 3000

# Start with a custom app
aegra dev --app myapp.main:app

# Start with a specific config and env file
aegra dev --config ./aegra.json --env-file ./.env

# Start without automatic database check
aegra dev --no-db-check
```

The server automatically restarts when code changes are detected.

---

### `aegra serve`

Run the production server (no hot reload). On Linux and macOS the command replaces itself with uvicorn, so the server receives the container's stop signal directly; on Windows uvicorn runs as a child process.

```bash
aegra serve [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Host to bind the server to |
| `--port PORT` | `2026` | Port to bind the server to |
| `--app APP` | `aegra_api.main:app` | Application import path |
| `-c, --config PATH` | | Path to aegra.json config file |

**Examples:**

```bash
# Start with defaults (0.0.0.0:2026)
aegra serve

# Start with a custom config
aegra serve --config ./aegra.json
```

---

### `aegra up`

Start services with Docker Compose.

```bash
aegra up [SERVICES...] [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `SERVICES` | Optional list of specific services to start |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --file FILE` | | Path to docker-compose.yml file |
| `--build / --no-build` | `--build` | Build images before starting containers (on by default) |

**Examples:**

```bash
# Start all services (builds by default)
aegra up

# Start only postgres
aegra up postgres

# Start without building
aegra up --no-build

# Start with a custom compose file
aegra up -f ./docker-compose.yml
```

---

### `aegra down`

Stop services with Docker Compose.

```bash
aegra down [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-f, --file FILE` | Path to docker-compose.yml file |
| `-v, --volumes` | Remove named volumes declared in the compose file |

**Examples:**

```bash
# Stop all services
aegra down

# Stop and remove volumes (WARNING: data will be lost)
aegra down -v

# Stop with a custom compose file
aegra down -f ./docker-compose.yml
```

---

## Environment Variables

The CLI respects the following environment variables (typically set via `.env` file):

```bash
# Database
POSTGRES_USER=aegra
POSTGRES_PASSWORD=aegra_secret
POSTGRES_HOST=localhost
POSTGRES_DB=aegra

# Authentication
AUTH_TYPE=noop  # Options: noop, custom

# Server (for aegra dev)
HOST=0.0.0.0
PORT=2026

# Configuration
AEGRA_CONFIG=aegra.json
```

## Requirements

- Python 3.12+
- Docker (for `aegra up` and `aegra down` commands)
- PostgreSQL (or use Docker)

## Related Packages

- **aegra-api**: Core API package providing the Agent Protocol server
- **aegra**: Meta-package that installs both aegra-cli and aegra-api
