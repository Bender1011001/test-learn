# Contributing to CAMEL Extensions

Thank you for your interest in contributing to the CAMEL Extensions project! This document provides guidelines and instructions for contributing to the project.

## Local Development Setup

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd camel
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   make setup
   ```

4. Start the full stack using Docker Compose:
   ```bash
   make compose
   ```

5. Alternatively, for local development without Docker:
   - In one terminal: `make dev-api` (Starts the backend API)
   - In another terminal: `make dev-gui` (Starts the Streamlit GUI)
   - Load demo data: `make fixtures`

## Project Structure

```
camel/
├── backend/                 # Backend API and services
│   ├── api/                 # FastAPI application
│   │   ├── routers/         # API endpoints
│   │   └── main.py          # API entry point
│   ├── core/                # Core business logic
│   │   └── services/        # Service implementations
│   ├── db/                  # Database models and access
│   └── migrations/          # Alembic migrations
├── configs/                 # Configuration files
│   └── agents.yaml          # Agent configuration
├── gui/                     # Frontend Streamlit application
│   ├── views/               # UI view components
│   ├── api_client.py        # API client for backend communication
│   ├── websocket_client.py  # WebSocket client for real-time updates
│   └── app.py               # GUI entry point
├── models/                  # Model files and adapters
└── scripts/                 # Utility scripts
```

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `dev`: Development branch for integration
- Feature branches: `feature/<feature-name>`
- Bug fix branches: `fix/<bug-name>`

### Pull Request Process

1. Fork the repository and create a feature/bugfix branch
2. Make your changes, ensuring code quality and test coverage
3. Update documentation if necessary
4. Submit a pull request to the `dev` branch
5. Address any review comments
6. Once approved, your changes will be merged

## Coding Standards

- **Python**: We follow PEP 8 guidelines with a line limit of 100 characters.
- Use [Black](https://github.com/psf/black) for Python code formatting.
- Use [isort](https://github.com/PyCQA/isort) to sort imports.
- Add type hints to all Python functions.
- Write docstrings for all modules, classes, and functions.

## Testing

- Write unit tests for all new functionality.
- Ensure all tests pass before submitting a pull request.
- Run tests with: `make test`
- Run end-to-end tests with: `make e2e`

## Documentation

- Update documentation for any API, UI, or behavior changes.
- Document complex algorithms and design decisions with comments.
- Keep the README and other documentation files up to date.

## Commit Messages

Follow the conventional commits specification for commit messages:

- `feat: add new feature`
- `fix: fix bug`
- `docs: update documentation`
- `style: formatting changes`
- `refactor: code restructuring`
- `test: add tests`
- `chore: maintenance tasks`

## Getting Help

If you have questions or need help with your contribution:

- Create an issue with your question.
- Reach out to project maintainers.

## Code of Conduct

- Be respectful and inclusive in all interactions.
- Provide constructive feedback.
- Focus on what's best for the community and the project.

Thank you for contributing to CAMEL Extensions!
