# CAMEL Extensions Project Roadmap

This document outlines the planned future enhancements and features for the CAMEL Extensions project, organized by timeline and priority.

## Current Release (0.2.0)

The current release implements the "Walking Skeleton" version of the application:

- ✅ Basic FastAPI backend with SQLite database
- ✅ Streamlit GUI with API client integration
- ✅ Workflow execution and visualization
- ✅ Configuration management
- ✅ Log exploration and annotation
- ✅ DPO training setup and monitoring
- ✅ WebSocket support for real-time updates
- ✅ Redis integration for PubSub (e.g., DPO training, workflow status/log updates)

## Near Term (Next 1-2 Releases)

### Walk-2: Add Celery/RabbitMQ + WebSocket feeds

- 🔲 Celery integration for background task processing (may leverage existing Redis for message broking or introduce RabbitMQ)
  - Workflow execution as background tasks
  - DPO training as background tasks
  - Task monitoring and management
- 🔲 RabbitMQ for message queuing (alternative/addition to Redis for Celery)
  - Reliable message delivery
  - Task distribution
  - Event-driven architecture
- 🔲 Enhanced WebSocket support
  - Centralized event dispatch
  - Client reconnection improvements
  - Better error handling

### Walk-3: Observability & Hardening

- 🔲 Redis for advanced caching/session management (e.g., Streamlit session store)
  - Session persistence across restarts
  - Shared state in multi-worker deployments
- 🔲 Comprehensive logging and monitoring
  - Structured logging (JSON format)
  - Vector.dev integration for log aggregation
  - Metrics collection with Prometheus
  - Dashboards with Grafana
- 🔲 Error handling and recovery improvements
  - Graceful degradation
  - Circuit breakers for external dependencies
  - Error tracing and correlation IDs
- 🔲 Security hardening
  - API authentication (JWT)
  - Role-based access control
  - Input validation and sanitization

## Mid Term (3-6 Months)

### Scaling & Polish

- 🔲 Kubernetes deployment
  - Helm charts
  - Horizontal Pod Autoscaling
  - Resource limit configuration
  - Readiness/liveness probes
- 🔲 Model & Adapter Hub
  - Model registry integration
  - Versioning of models and adapters
  - Model sharing and distribution
  - Model metadata and provenance tracking
- 🔲 Advanced DPO training features
  - Hyperparameter optimization
  - Training visualization and introspection
  - A/B testing of models and adapters
  - Training job scheduling and prioritization
- 🔲 Enhanced UI/UX
  - Responsive design improvements
  - Real-time collaboration features
  - Dark mode support
  - User preferences and customization

## Long Term (6+ Months)

- 🔲 Distributed training support
  - Multi-GPU training
  - Training job distribution across nodes
  - CheckpointIO and model sharding
  - Distributed evaluation
- 🔲 Multi-tenant deployment
  - Organizational boundaries
  - Team-based access control
  - Usage quotas and metering
  - Multi-region support
- 🔲 Integration with model registry systems
  - Hugging Face Hub integration
  - MLflow integration
  - Custom model registry
  - Model versioning and lifecycle management
- 🔲 Advanced agent orchestration
  - Complex workflow definitions
  - RLHF (Reinforcement Learning from Human Feedback)
  - Multi-step improvement workflows
  - Autonomous agent optimization
- 🔲 Web-based UI alternative
  - React/Next.js frontend
  - GraphQL API support
  - Real-time collaboration
  - Rich visualizations and dashboards

## Wishlist (Priorities TBD)

- 🔲 OpenAPI and Swagger UI documentation
- 🔲 API client libraries in multiple languages
- 🔲 Batch processing of DPO annotations
- 🔲 Training data augmentation 
- 🔲 Integration with external data sources
- 🔲 Support for quantized inference
- 🔲 Mobile-friendly UI
- 🔲 Progressive Web App (PWA) support
- 🔲 Workflow templates and sharing
- 🔲 Configuration version control
- 🔲 Import/export of workflows and configurations
- 🔲 Notification system for long-running tasks
- 🔲 Scheduled workflow runs

---

*Note: This roadmap is subject to change based on community feedback, project priorities, and resource availability.*