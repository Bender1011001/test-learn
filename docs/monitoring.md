# Monitoring and Observability System

This document describes the monitoring and observability features implemented in the CAMEL Extensions project.

## Overview

The monitoring system provides comprehensive visibility into the performance, health, and behavior of the CAMEL Extensions application. It tracks API requests, database operations, cache performance, errors, and system health metrics.

## Components

### Backend Components

1. **MonitoringService** (`backend/core/services/monitoring_service.py`)
   - Core service that collects and manages metrics
   - Provides methods for recording various types of metrics
   - Optionally exports metrics to Prometheus

2. **Database Monitoring** (`backend/core/services/db_monitoring.py`)
   - Decorators for monitoring database operations
   - Automatic instrumentation of DBManager methods
   - Tracks query performance and cache effectiveness

3. **Monitoring Middleware** (`backend/api/middleware/monitoring.py`)
   - FastAPI middleware for tracking API requests
   - Custom route class for per-endpoint timing
   - Request/response logging

4. **Metrics API Endpoints** (`backend/api/routers/metrics.py`)
   - Endpoints for retrieving metrics data
   - Health check endpoint
   - Prometheus metrics endpoint

### Frontend Components

1. **Monitoring Dashboard** (`gui/views/monitoring_view.py`)
   - Interactive visualization of metrics data
   - System health display
   - Performance analysis tools

## Metrics Collected

The monitoring system collects the following types of metrics:

1. **API Metrics**
   - Request count by endpoint
   - Latency (min, max, avg, p95, p99)
   - Error count and rate

2. **Database Metrics**
   - Query count by operation type
   - Query latency (min, max, avg, p95, p99)
   - Transaction success/failure rate

3. **Cache Metrics**
   - Hit/miss count by cache key
   - Hit rate percentage
   - Cache size and eviction rate

4. **Error Metrics**
   - Error count by type and source
   - Error rate
   - Error stack traces (in logs)

5. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk usage
   - Process information

## Using the Monitoring System

### Viewing Metrics in the GUI

1. Launch the CAMEL Extensions GUI:
   ```
   cd gui
   streamlit run app.py
   ```

2. Navigate to the "Monitoring" section in the sidebar.

3. Explore the different tabs to view various metrics:
   - API Performance
   - Database Performance
   - Cache Performance
   - Error Tracking
   - System Health

4. Use the refresh button or enable auto-refresh to get the latest metrics.

### Accessing Metrics via API

The following API endpoints are available for accessing metrics:

- `GET /api/metrics` - Get a summary of all metrics
- `GET /api/metrics/api` - Get API performance metrics
- `GET /api/metrics/db` - Get database performance metrics
- `GET /api/metrics/cache` - Get cache performance metrics
- `GET /api/metrics/errors` - Get error metrics
- `GET /api/metrics/health` - Get system health metrics
- `GET /api/metrics/prometheus` - Get metrics in Prometheus format

Example:
```bash
curl http://localhost:8000/api/metrics/health
```

### Testing the Monitoring System

A test script is provided to generate load and test the monitoring system:

```bash
cd scripts
python test_monitoring.py --all
```

Options:
- `--api-url` - API base URL (default: http://localhost:8000/api)
- `--all` - Run all tests
- `--random-load` - Generate random load
- `--num-requests` - Number of random requests
- `--delay` - Delay between requests
- `--cache` - Test cache performance
- `--errors` - Test error handling
- `--concurrent` - Test concurrent requests
- `--threads` - Number of threads for concurrent requests
- `--concurrent-requests` - Number of requests per thread
- `--display` - Display metrics

## Prometheus Integration

The monitoring system can export metrics to Prometheus for integration with industry-standard monitoring tools.

### Requirements

- Prometheus server
- `prometheus_client` Python package (included in requirements.txt)

### Configuration

1. Set the `PROMETHEUS_PORT` environment variable to enable the Prometheus HTTP server:
   ```
   export PROMETHEUS_PORT=9090
   ```

2. Configure Prometheus to scrape metrics from the application:
   ```yaml
   scrape_configs:
     - job_name: 'camel_extensions'
       scrape_interval: 15s
       static_configs:
         - targets: ['localhost:9090']
   ```

3. Alternatively, use the `/api/metrics/prometheus` endpoint with Prometheus HTTP scraping.

## Extending the Monitoring System

### Adding New Metrics

To add new metrics to the monitoring system:

1. Add the metric to the `_metrics` dictionary in `MonitoringService.__init__`.
2. Add methods to record the metric in `MonitoringService`.
3. Update the `get_metrics_summary` method to include the new metric.
4. Add the metric to the Prometheus initialization if needed.

### Adding Custom Dashboards

To add custom dashboards to the frontend:

1. Create a new view file in `gui/views/`.
2. Implement the dashboard using Streamlit components.
3. Add the view to `gui/app.py`.

## Best Practices

1. **Use Decorators**: Use the provided decorators to monitor new functions.
2. **Monitor Critical Paths**: Ensure all critical code paths are monitored.
3. **Set Appropriate TTLs**: Configure cache TTLs based on data volatility.
4. **Review Metrics Regularly**: Regularly review metrics to identify issues.
5. **Set Up Alerts**: Configure alerts for critical metrics.

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check database query performance
   - Review API endpoint implementations
   - Check for resource contention

2. **Low Cache Hit Rate**
   - Review cache key generation
   - Adjust TTL values
   - Check for cache invalidation issues

3. **High Error Rate**
   - Check error logs for details
   - Review error handling in code
   - Check for external service failures

### Logs

Detailed logs are available in the `logs/` directory:

- `logs/api_{time}.log` - API logs
- `logs/app_{time}.log` - Application logs

## Future Enhancements

Planned enhancements for the monitoring system:

1. Historical metrics storage for trend analysis
2. Alerting system for critical issues
3. Distributed tracing for complex request flows
4. Custom dashboards for specific use cases
5. Automated performance testing with metric validation