# CAMEL Extensions Project Makefile

.PHONY: compose dev-api dev-gui fixtures test e2e clean run-all setup

# Default target
all: setup run-all

# Setup tasks
setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r backend/requirements.txt

# Docker Compose for full stack
compose:
	docker-compose --profile dev up --build

# Run the backend API
dev-api:
	@echo "Starting backend API..."
	cd backend && alembic upgrade head && uvicorn api.main:app --reload --port 8000

# Run the frontend GUI
dev-gui:
	@echo "Starting Streamlit GUI..."
	API_BASE_URL=http://localhost:8000/api streamlit run gui/app.py

# Run both backend and GUI in separate terminals (use with tmux or screen)
run-all:
	@echo "Starting both backend and GUI..."
	$(MAKE) dev-api & $(MAKE) dev-gui

# Load demo data
fixtures:
	@echo "Loading demo data..."
	python backend/scripts/load_demo_data.py

# Run backend tests
test:
	@echo "Running tests..."
	cd backend && pytest

# Run end-to-end tests
e2e:
	@echo "Running E2E tests..."
	cd e2e && npx playwright test

# Clean up
clean:
	@echo "Cleaning up..."
	rm -f *.db
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

# Show help
help:
	@echo "CAMEL Extensions Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup      - Install required dependencies"
	@echo "  compose    - Run full stack via Docker Compose"
	@echo "  dev-api    - Run backend API server in development mode"
	@echo "  dev-gui    - Run Streamlit GUI in development mode"
	@echo "  run-all    - Run both backend and GUI (in background)"
	@echo "  fixtures   - Load demo data"
	@echo "  test       - Run tests"
	@echo "  e2e        - Run end-to-end tests"
	@echo "  clean      - Clean up temporary files"
