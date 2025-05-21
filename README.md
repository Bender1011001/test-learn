# CAMEL Extensions

This project provides a comprehensive system for managing, observing, and improving CAMEL AI agents. It includes a FastAPI backend and Streamlit GUI frontend, focusing on facilitating workflows like the proposer-executor loop and streamlining the Direct Preference Optimization (DPO) training process.

![CAMEL Logo](misc/logo_light.png)

## Features

- **Workflow Execution:** Initiate and observe pre-defined CAMEL agent workflows in real-time
- **Configuration Management:** View and modify configurations for agents and workflows via a user-friendly interface
- **Log Exploration & Annotation:** Review historical agent interactions and create preference data for DPO training
- **DPO Training:** Configure and start DPO training runs for agents using annotated data
- **Real-time Updates:** WebSocket connections for live workflow and training monitoring
- **API Backend:** RESTful API for programmatic access to all functionality

## Architecture Overview

This project follows a modern architecture with a clear separation between frontend and backend:

- **Frontend**: Streamlit-based GUI with API client integration
- **Backend**: FastAPI server providing RESTful endpoints and WebSocket connections
- **Database**: SQLAlchemy ORM with migrations (Alembic) supporting SQLite or PostgreSQL
- **Services**: Core business logic services for configuration, workflows, database, and DPO training

See the [architecture documentation](docs/architecture.md) for a detailed overview.

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11 (camel-ai is not compatible with Python 3.12+)
- pip (for installing dependencies)
- Docker and Docker Compose (optional, for containerized deployment)

### Quick Start with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/camel-ai/camel.git
   cd camel
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

3. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/camel-ai/camel.git
   cd camel
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

5. Initialize the database:
   ```bash
   cd backend
   alembic upgrade head
   ```

6. Load demo data (optional):
   ```bash
   python backend/scripts/load_demo_data.py
   ```

7. Start the backend:
   ```bash
   cd backend
   uvicorn api.main:app --reload
   ```

8. Start the frontend (in a new terminal):
   ```bash
   streamlit run gui/app.py
   ```

9. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### API Keys (Optional)

Set up necessary API keys as environment variables (e.g., `OPENAI_API_KEY`). The specific keys required will depend on the Large Language Models (LLMs) you configure for your agents in `configs/agents.yaml`.

## Project Structure

```
camel/
├── backend/                # Backend API and services
│   ├── api/                # FastAPI application
│   │   ├── routers/        # API endpoints
│   │   └── main.py         # API entry point
│   ├── core/               # Core business logic
│   │   └── services/       # Service implementations
│   ├── db/                 # Database models and access
│   │   └── models/         # SQLAlchemy models
│   └── migrations/         # Alembic migrations
├── configs/                # Configuration files
│   └── agents.yaml         # Agent configuration
├── gui/                    # Frontend Streamlit application
│   ├── views/              # UI view components
│   ├── api_client.py       # API client for backend communication
│   ├── websocket_client.py # WebSocket client for real-time updates
│   └── app.py              # GUI entry point
├── docs/                   # Documentation
│   └── architecture.md     # Architecture overview
├── models/                 # Default directory for trained models
├── docker-compose.yml      # Docker Compose configuration
├── Makefile                # Development utilities
├── CONTRIBUTING.md         # Contribution guidelines
└── ROADMAP.md              # Future development plans
```

## Development with Make

For easier development, we provide a Makefile with common tasks:

```bash
# Setup dependencies
make setup

# Start both API and GUI in development mode
make run-all

# Start only API in development mode
make dev-api

# Start only GUI in development mode
make dev-gui

# Load demo data
make fixtures

# Run tests
make test

# Run end-to-end tests
make e2e

# Clean up temporary files
make clean
```

## API Documentation

When the backend is running, you can access the API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

See the [Roadmap](ROADMAP.md) for information about planned features and improvements.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.