# CAMEL Extensions Backend

This is the backend implementation for the CAMEL Extensions GUI, providing API services for workflow execution, configuration management, log exploration, and DPO training.

## Architecture

The backend follows a layered architecture:

- **API Layer**: FastAPI-based RESTful API and WebSocket endpoints
- **Service Layer**: Core business logic services (Config Manager, Workflow Manager, DB Manager, DPO Trainer)
- **Data Access Layer**: SQLAlchemy models and database interactions

## Setup and Installation

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, SQLite can be used for development)
- Redis (optional, used for caching and task queue)

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (or create a `.env` file):
```
DATABASE_URL=sqlite:///./camel_extensions.db  # For development
API_PORT=8000
DEBUG=true
```

4. Apply database migrations:
```bash
cd backend
alembic upgrade head
```

## Running the Server

### Development Mode

```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### Production Mode

```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker

### Running with Docker Compose

The entire stack (API, database, Redis, and UI) can be run using Docker Compose:

```bash
docker-compose up
```

### Building and Running API Container Only

```bash
cd backend
docker build -t camel-extensions-api .
docker run -p 8000:8000 camel-extensions-api
```

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Database Migrations

### Creating a Migration

```bash
cd backend
alembic revision --autogenerate -m "Description of changes"
```

### Applying Migrations

```bash
cd backend
alembic upgrade head
```

## Directory Structure

- `api/`: FastAPI application and routers
- `core/`: Core services and business logic
  - `services/`: Service implementations
- `db/`: Database models and utilities
  - `models/`: SQLAlchemy models
- `migrations/`: Alembic migration scripts