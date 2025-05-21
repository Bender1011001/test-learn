# CAMEL Extensions Project Finalization Plan

This document outlines the detailed plan for integrating the backend with the frontend, testing, and finalizing the CAMEL Extensions project.

## 0. Adopt "Walking Skeleton" Release Strategy

To ensure manageable progress, we'll adopt an incremental "Walking Skeleton" approach with multiple stages:

| Walk | Focus | Description |
|------|-------|------------|
| Walk-1 | **Basic Integration** | API + GUI talk over HTTP, single docker-compose, SQLite, NO Celery. |
| Walk-2 | **Async Features** | Add Celery/RabbitMQ + WebSocket feeds. |
| Walk-3 | **Observability & QA** | Add monitoring, logging, and E2E testing. |
| Walk-4+ | **Scaling & Polish** | Redis sessions, Kubernetes YAML, etc. |

This approach allows us to build a fully functional system early while preserving hooks for future enhancements.

## 1. API Client Hardening Adjustments

The API client should be robust with proper error handling and retries:

```python
from tenacity import retry, stop_after_attempt, wait_exponential
DEFAULT_TIMEOUT = (3, 10)  # (connect_timeout, read_timeout)

class APIClient:
    # ...
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    def _request(self, method: str, path: str, **kwargs):
        self.logger.debug(f"API Request: {method} {path}")
        response = self.session.request(method,
                                        f"{self.base_url}{path}",
                                        timeout=DEFAULT_TIMEOUT,
                                        **kwargs)
        self.logger.debug(f"API Response: {response.status_code}")
        return response
        
    def get(self, path, **kwargs):
        return self._request("GET", path, **kwargs)
    
    def post(self, path, json=None, **kwargs):
        return self._request("POST", path, json=json, **kwargs)
    
    # Similar methods for PUT, DELETE, etc.
    
    # Important: Add close() method to clean up resources
    def close(self):
        """Close the API client and any associated resources."""
        # Close any active WebSocket connections
        for ws in self._active_websockets:
            try:
                ws.close()
            except:
                pass
        self.session.close()
```

The client should be used in Streamlit's `on_session_end` to ensure proper cleanup.

## 2. State & Cache Strategy

Implement a strategic approach to state and caching in Streamlit:

| Data Type | Strategy | Invalidation |
|-----------|----------|--------------|
| Config YAML | `st.cache_data(ttl=30)` | Explicit "Reload" button calls `cache_clear()`. |
| Logs | No cache – always paginate API | N/A |
| Adapter list | `ttl=60` | WS "dpo_finished" event triggers clear. |

Example implementation:

```python
@st.cache_data(ttl=30)
def fetch_agent_configs():
    """Fetch agent configurations with caching."""
    return api_client.get_all_agent_configs()

# In the UI code
if st.button("Reload Configurations"):
    # Clear the cache for this function
    fetch_agent_configs.clear()
    st.success("Configuration reloaded!")
```

## 3. Robust WebSocket Wrapper

Implement a production-ready WebSocket wrapper with these features:

- Heartbeat ping every 15 seconds to keep connections alive
- Exponential backoff reconnect strategy (maximum 5 minutes)
- Message queue with size limit (`queue.Queue(maxsize=500)`)
- Drop oldest messages if queue is full (log warning)
- Clean shutdown with threading `Event` for background thread termination

```python
import queue
import threading
import time
import websocket
import json
from loguru import logger

class RobustWebSocket:
    def __init__(self, url, on_message, max_queue_size=500):
        self.url = url
        self.user_callback = on_message
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.ws = None
        self.connected = False
        self.should_reconnect = True
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_reconnect_delay = 300  # Maximum 5 minutes
        self.stop_event = threading.Event()
        
        # Start message processing thread
        self.process_thread = threading.Thread(target=self._process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Connect
        self._connect()
    
    def _connect(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start the WebSocket connection in a thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                'ping_interval': 15,
                'ping_timeout': 10
            })
            self.ws_thread.daemon = True
            self.ws_thread.start()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._schedule_reconnect()
    
    def _on_message(self, ws, message):
        try:
            # Try to add to queue, discard if full
            try:
                self.message_queue.put_nowait(message)
            except queue.Full:
                logger.warning("WebSocket message queue full, dropping oldest message")
                # Discard oldest and add new
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait(message)
                except:
                    pass
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        if self.should_reconnect:
            self._schedule_reconnect()
    
    def _on_open(self, ws):
        logger.info(f"WebSocket connected to {self.url}")
        self.connected = True
        self.reconnect_delay = 1  # Reset reconnect delay
    
    def _schedule_reconnect(self):
        if not self.should_reconnect or self.stop_event.is_set():
            return
            
        logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
        time.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        # Try to reconnect
        self._connect()
    
    def _process_messages(self):
        while not self.stop_event.is_set():
            try:
                # Block for 0.5 seconds, then check if we should stop
                try:
                    message = self.message_queue.get(timeout=0.5)
                    # Parse JSON and deliver to user callback
                    data = json.loads(message)
                    self.user_callback(data)
                except queue.Empty:
                    continue
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
            
        logger.info("WebSocket message processor stopped")
    
    def send(self, message):
        """Send a message through the WebSocket."""
        if not self.connected:
            logger.warning("Cannot send message, WebSocket not connected")
            return False
        
        try:
            self.ws.send(message)
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    def stop(self):
        """Stop the WebSocket connection and message processor."""
        self.should_reconnect = False
        self.stop_event.set()
        
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        logger.info("WebSocket stopped")
```

## 4. True Health Endpoints

Add a FastAPI sidecar in the same Streamlit container to provide proper health checks:

```python
# gui/health.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
```

Update the GUI Dockerfile:

```dockerfile
# Run both the FastAPI health endpoint and Streamlit app
CMD ["bash", "-c", "uvicorn health:app --port 8502 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
```

Update `docker-compose.yml` to include health checks:

```yaml
services:
  ui:
    # ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8502/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 5. Log & Trace Aggregation

Implement comprehensive logging and tracing:

1. **Structured JSON Logging**: Configure all services to output structured JSON logs to stdout for container collection.

2. **Log Aggregation**: Recommend using [Vector](https://vector.dev) sidecars to fan-in logs locally:
   ```yaml
   vector:
     image: timberio/vector:latest
     volumes:
       - ./vector.toml:/etc/vector/vector.toml
       - /var/log:/var/log
     depends_on:
       - api
       - ui
   ```

3. **Trace ID Propagation**: Add HTTP header propagation for traceability:
   ```python
   # In GUI client
   def _request(self, method, path, **kwargs):
       headers = kwargs.get('headers', {})
       # Generate or propagate trace ID
       if 'X-Trace-ID' not in headers:
           headers['X-Trace-ID'] = str(uuid.uuid4())
       kwargs['headers'] = headers
       # Log with trace ID
       self.logger.debug(f"API Request: {method} {path} (Trace: {headers['X-Trace-ID']})")
       # Continue with request
   ```

   Backend API middleware to capture trace IDs:
   ```python
   @app.middleware("http")
   async def add_trace_id(request: Request, call_next):
       trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
       request.state.trace_id = trace_id
       logger.bind(trace_id=trace_id).debug(f"Request: {request.method} {request.url.path}")
       response = await call_next(request)
       return response
   ```

## 6. Developer Ergonomics

Add developer conveniences to the project root:

### Makefile

```makefile
.PHONY: compose dev-api dev-gui fixtures

compose:
	docker-compose --profile dev up --build

dev-api:
	cd backend && alembic upgrade head && uvicorn api.main:app --reload

dev-gui:
	API_BASE_URL=http://localhost:8000/api streamlit run gui/app.py

fixtures:
	python backend/scripts/load_demo_data.py

test:
	cd backend && pytest

e2e:
	cd e2e && npx playwright test
```

### .env.example

```
# API Configuration
API_PORT=8000
DEBUG=true
DATABASE_URL=sqlite:///./camel_extensions.db

# GUI Configuration
STREAMLIT_SERVER_PORT=8501
API_BASE_URL=http://localhost:8000/api

# Optional: Uncomment for PostgreSQL
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/camel_extensions

# Optional: Uncomment for Redis
# REDIS_URL=redis://localhost:6379/0
```

### load_demo_data.py

Create a script to populate the database with sample data for development:

```python
# backend/scripts/load_demo_data.py
"""
Load demo data into the database for development.
Creates sample workflows, logs, and annotations.
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import random
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import models and database
from backend.db.base import SessionLocal, engine
from backend.db.models import InteractionLog, DPOAnnotation

def main():
    """Load demo data into the database."""
    db = SessionLocal()
    try:
        # Generate 5 workflow runs
        workflow_ids = ["proposer_executor_review_loop"]
        agent_names = ["Proposer", "Executor", "PeerReviewer", "system"]
        
        # Create 100 log entries across 5 workflow runs
        workflow_runs = [str(uuid.uuid4()) for _ in range(5)]
        
        print(f"Creating {len(workflow_runs)} workflow runs with logs...")
        
        for i, run_id in enumerate(workflow_runs):
            # System log for workflow start
            start_time = datetime.utcnow() - timedelta(days=i, hours=random.randint(0, 12))
            
            # Create start log
            start_log = InteractionLog(
                workflow_run_id=run_id,
                timestamp=start_time,
                agent_name="system",
                agent_type="system",
                input_data={"workflow_id": workflow_ids[0]},
                output_data={"status": "started", "initial_goal": f"Demo goal {i+1} for testing"}
            )
            db.add(start_log)
            
            # Create agent interaction logs
            for j in range(20):  # 20 logs per workflow
                agent_idx = j % (len(agent_names) - 1)  # Skip "system" for regular logs
                log_time = start_time + timedelta(seconds=j*30)  # 30 seconds between logs
                
                log = InteractionLog(
                    workflow_run_id=run_id,
                    timestamp=log_time,
                    agent_name=agent_names[agent_idx],
                    agent_type=agent_names[agent_idx],
                    input_data={"step": j, "agent": agent_names[agent_idx]},
                    output_data={"response": f"Sample response from {agent_names[agent_idx]} for step {j}", "status": "success"}
                )
                db.add(log)
                
                # Add annotations to some Proposer logs
                if agent_names[agent_idx] == "Proposer" and random.random() < 0.3:
                    db.flush()  # Ensure log has an ID
                    annotation = DPOAnnotation(
                        log_entry_id=log.id,
                        rating=random.uniform(1.0, 5.0),
                        rationale=f"Sample annotation rationale for log {log.id}",
                        chosen_prompt=f"Better prompt for goal: {log.input_data.get('step')}",
                        rejected_prompt=f"Worse prompt for goal: {log.input_data.get('step')}",
                        dpo_context=f"Context for DPO training with goal: {log.input_data.get('step')}",
                        user_id="demo_user",
                        timestamp=log_time + timedelta(minutes=random.randint(5, 60))
                    )
                    db.add(annotation)
            
            # Create completion log
            end_log = InteractionLog(
                workflow_run_id=run_id,
                timestamp=start_time + timedelta(seconds=20*30 + 10),
                agent_name="system",
                agent_type="system",
                input_data={"workflow_id": workflow_ids[0]},
                output_data={"status": "completed"}
            )
            db.add(end_log)
        
        db.commit()
        print(f"Created {len(workflow_runs) * 22} log entries with annotations.")
        
    except Exception as e:
        print(f"Error loading demo data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
```

## 7. Testing Pipeline Tweaks

Establish a comprehensive testing strategy:

| Frequency | Tests |
|-----------|-------|
| Per-PR | Lint + unit tests + smoke Playwright (1 path) |
| Nightly | Full Playwright matrix (all browsers) + docker-compose E2E |

**CI Configuration:**
- Split backend & GUI jobs to run on parallel workers
- Cache pip installations to reduce CI runtime
- Set up matrix testing for browsers in Playwright

Example GitHub Actions workflow:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'
        cache-dependency-path: 'backend/requirements.txt'
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    - name: Lint
      run: |
        cd backend
        pip install black flake8 mypy
        black --check .
        flake8 .
        mypy .
    - name: Test
      run: |
        cd backend
        pytest

  frontend:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Install dependencies
      run: |
        pip install -r gui/requirements.txt
    - name: Lint
      run: |
        pip install black flake8
        black --check gui
        flake8 gui

  e2e-smoke:
    runs-on: ubuntu-latest
    needs: [backend, frontend]
    steps:
    - uses: actions/checkout@v3
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Install Playwright
      run: |
        npx playwright install --with-deps chromium
    - name: Run smoke tests
      run: |
        npx playwright test -c e2e/smoke-config.js
```

## 8. Documentation Must-Haves

Essential documentation to include:

### CONTRIBUTING.md

```markdown
# Contributing to CAMEL Extensions

## Local Development Setup

### Quick Start

1. Clone the repository
2. Run `make compose` to start the entire stack in Docker
3. Or for local development:
   - In one terminal: `make dev-api`
   - In another terminal: `make dev-gui`
   - Load demo data: `make fixtures`

### Coding Standards

- Python: We follow PEP 8 guidelines with a 100-character line limit
- Use [Black](https://github.com/psf/black) for formatting Python code
- Use [isort](https://github.com/PyCQA/isort) to sort imports
- Add type hints to all Python functions

### Testing

- Backend: `cd backend && pytest`
- E2E: `cd e2e && npx playwright test`

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if necessary
3. Add yourself to CONTRIBUTORS.md if it's your first contribution
4. Create a PR against the `main` branch
```

### Architecture Diagram (docs/architecture.svg)

Create a simple architecture diagram showing:
- GUI ⇄ API ⇄ DB/Celery/Redis
- WebSocket connections
- Data flow between components

## 9. Deferred Items (ROADMAP.md)

Create a `ROADMAP.md` file listing future enhancements:

```markdown
# CAMEL Extensions Roadmap

Items deferred to future releases:

## Near-term (Next 1-2 releases)
- Redis session store for Streamlit
- Improved error handling and recovery
- Comprehensive logging and monitoring

## Mid-term (3-6 months)
- Kubernetes manifests
- Model & Adapter Hub UX improvements
- Security layer (JWT authentication, RBAC)

## Long-term (6+ months)
- Distributed training support
- Multi-tenant deployment option
- Integration with model registry systems