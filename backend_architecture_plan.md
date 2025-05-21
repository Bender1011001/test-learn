# CAMEL Extensions GUI Backend Architecture Plan

This document outlines the comprehensive architecture and implementation plan for the backend of the CAMEL Extensions GUI, which will support the self-improving agent system based on mutual ranking and Direct Preference Optimization (DPO).

## 1. Architecture Overview

### API Layer

- **Technology**: FastAPI + Uvicorn
- **Responsibilities**:
  - Expose REST/WebSocket endpoints for GUI actions:
    - Start/stop workflows
    - Stream logs
    - Trigger DPO training
    - CRUD operations for configs and annotations
  - Use Pydantic schemas for request/response validation

### Core Services

Each service will be implemented as its own submodule:

- **Workflow Manager**
  - Wraps the CAMEL AI library
  - Manages workflow execution lifecycle
  - Streams interaction data to UI and database

- **Config Manager**
  - Loads and validates `agents.yaml` via Pydantic
  - Uses ruamel.yaml for round-trip edits (preserving comments)
  - Handles configuration persistence and validation

- **DB Manager**
  - SQLAlchemy + SQLite for development
  - PostgreSQL for production
  - Alembic for database migrations

- **DPO Trainer**
  - Implements Celery tasks or subprocess wrapper around `scripts/train_dpo.py`
  - Manages training job lifecycle
  - Handles adapter artifact storage and registration

### Persistence

- **Configurations**: `configs/agents.yaml`
- **Database**:
  - `InteractionLogs` table
  - `DPOAnnotations` table
  - Schema managed via SQLAlchemy models and Alembic migrations
- **Artifacts**: Trained adapters stored under `models/adapters/...`

### UI

- **Technology**:
  - Streamlit for MVP
  - Optional migration to React/Next.js in future iterations
- **State Management**:
  - Streamlit session_state for keeping handles to active workflows
  - WebSockets for live data feeds (workflow interactions, training status)

## 2. DevOps & CI/CD

### Containerization

- **Two Dockerfiles**:
  - `api/`: FastAPI + Uvicorn
  - `ui/`: Streamlit
- **docker-compose.yml**:
  - Spins up Postgres/Redis/RabbitMQ for local development
  - Configurable for different environments (dev, staging, prod)

### CI/CD Pipeline (GitHub Actions)

- **Code Quality**:
  - Lint (black, flake8, isort, mypy)
  - Static analysis
- **Testing**:
  - Run unit tests (pytest + coverage)
  - Generate coverage reports
- **Build & Deployment**:
  - Build & push Docker images
  - Run basic smoke tests against a docker-compose stack
  - Deploy to target environment

### Migrations

- **Alembic** to manage DB schema changes:
  - Version-controlled schema migrations
  - Automated migration generation
  - Migration scripts included in CI/CD pipeline

## 3. Testing & Quality

### Unit Tests

- **Testing Framework**: pytest
- **Coverage**: Aim for >80%
- **Mock CAMEL workflows & DB** with pytest fixtures
- **Test Core Services** in isolation

### Integration Tests

- **Test API endpoints** using httpx.AsyncClient
- **Validate DB interactions** with test database
- **Ensure contract compliance** between services

### UI E2E

- **Playwright scripts** for core flows:
  - Starting a workflow
  - Annotating logs
  - Triggering DPO training
  - Configuration updates

### Pre-commit Hooks

- **pre-commit** with:
  - Code linters
  - Type checks
  - Unit test verification
  - Commit message formatting

## 4. Background Processing & Scalability

### Task Queue

- **Technology**: Celery + RabbitMQ (or Redis) for:
  - Long-running workflows
  - DPO training jobs
  - Periodic maintenance tasks
- **Features**:
  - Configurable retries
  - Timeouts
  - Result backends
  - Task monitoring

### Caching

- **Technology**: Redis
- **Use Cases**:
  - Caching heavy configuration loads
  - Repeated LLM calls
  - Session data

### Future Autoscaling

- **Kubernetes manifests** with:
  - Horizontal Pod Autoscaler (HPA) based on CPU/memory usage
  - Configurable resource limits and requests
  - Readiness/liveness probes

## 5. Observability & Monitoring

### Logging

- **Technology**: loguru
- **Format**: JSON-structured logs
- **Context**: Include request/task IDs for traceability
- **Storage**: Log rotation and archival

### Metrics

- **Technology**: Prometheus client in FastAPI
- **Endpoint**: Expose `/metrics`
- **Metrics**:
  - API latency
  - Workflow execution time
  - Training throughput
  - Error rates

### Health Checks

- **Endpoint**: `/health`
- **Validations**:
  - Database connectivity
  - Redis connectivity
  - RabbitMQ connectivity
  - File system access

### Error Tracking

- **Optional**: Sentry integration for uncaught exceptions
- **Alerts**: Configure alerts for critical failures

## 6. Robust State Management

### Session Store

- **Technology**: Redis-backed store
- **Purpose**: Back Streamlit's in-memory state with persistent storage
- **Benefits**: Avoids stale handles across UI refreshes

### Workflow Registry

- **Implementation**: DB table of active workflow runs
- **Features**:
  - TTL cleanup jobs
  - Orphaned workflow detection
  - Workflow state serialization/deserialization

## 7. Revised Phasing

| Phase | Deliverable | Description |
|-------|-------------|-------------|
| 1. Foundations | • FastAPI + Uvicorn stack<br>• Pydantic + ruamel.yaml for configs<br>• SQLite + Alembic<br>• Docker + docker-compose | Set up the basic infrastructure and project structure with minimal functionality |
| 2. "Hello World" Slice | • Streamlit "Start workflow" button → FastAPI WS<br>• One log entry in DB | Create an end-to-end minimal viable feature showing UI-API-DB interaction |
| 3. Config & Logs | • Edit/save agents.yaml in UI<br>• /logs REST + filters<br>• UI table + annotation CRUD | Implement configuration management and log exploration/annotation |
| 4. Background & DPO | • Celery tasks for workflows & training<br>• Real-time stdout via WS<br>• Adapter artifact drop | Add background processing and DPO training functionality |
| 5. Testing & CI/CD | • pytest suite + coverage<br>• pre-commit hooks<br>• GitHub Actions pipeline | Improve code quality and add automated testing |
| 6. Observability & Hardening | • Prometheus metrics + health<br>• loguru JSON logs<br>• Redis session store | Enhance production readiness with monitoring and resilience |

This phased approach allows for incremental delivery of working features while building toward the complete architecture.