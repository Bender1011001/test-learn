#!/usr/bin/env python3
"""
Test script for the monitoring system.

This script generates test load on the API to demonstrate the monitoring features.
It makes a series of API calls to different endpoints and displays the metrics.
"""
import requests
import time
import random
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default API base URL
DEFAULT_API_URL = "http://localhost:8000/api"

def make_request(endpoint, method="GET", data=None, params=None):
    """Make a request to the API and return the response."""
    url = f"{args.api_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            logger.error(f"Unsupported method: {method}")
            return None
        
        # Log the request
        logger.info(f"{method} {endpoint} - Status: {response.status_code}")
        
        return response
    except Exception as e:
        logger.error(f"Error making request to {url}: {str(e)}")
        return None

def generate_random_load(num_requests=50, delay=0.1):
    """Generate random load on the API."""
    # List of endpoints to test
    endpoints = [
        ("/health", "GET", None, None),
        ("/metrics", "GET", None, None),
        ("/metrics/api", "GET", None, None),
        ("/metrics/db", "GET", None, None),
        ("/metrics/cache", "GET", None, None),
        ("/metrics/errors", "GET", None, None),
        ("/metrics/health", "GET", None, None),
        ("/workflows", "GET", None, None),
        ("/configs/agents", "GET", None, None),
        ("/logs", "GET", None, {"limit": 10}),
        ("/logs/summary", "GET", None, None),
    ]
    
    # Make random requests
    for _ in range(num_requests):
        endpoint, method, data, params = random.choice(endpoints)
        make_request(endpoint, method, data, params)
        time.sleep(delay)

def test_cache_performance():
    """Test cache performance by making repeated requests."""
    logger.info("Testing cache performance...")
    
    # Make multiple requests to the same endpoint to test caching
    for _ in range(10):
        make_request("/logs/summary", "GET")
        time.sleep(0.1)
    
    logger.info("Cache performance test completed")

def test_error_handling():
    """Test error handling by making requests that will fail."""
    logger.info("Testing error handling...")
    
    # Make requests to non-existent endpoints
    make_request("/non-existent-endpoint", "GET")
    
    # Make requests with invalid parameters
    make_request("/logs", "GET", None, {"limit": "invalid"})
    
    # Make requests with invalid data
    make_request("/configs/agents/invalid", "PUT", {"invalid": "data"})
    
    logger.info("Error handling test completed")

def test_concurrent_requests(num_threads=5, num_requests=10):
    """Test concurrent requests using multiple threads."""
    logger.info(f"Testing concurrent requests with {num_threads} threads...")
    
    def worker():
        for _ in range(num_requests):
            endpoint = random.choice([
                "/metrics",
                "/logs/summary",
                "/health",
                "/configs/agents"
            ])
            make_request(endpoint, "GET")
            time.sleep(random.uniform(0.1, 0.5))
    
    # Create and start threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    logger.info("Concurrent requests test completed")

def display_metrics():
    """Display the current metrics."""
    logger.info("Fetching current metrics...")
    
    response = make_request("/metrics", "GET")
    if response and response.status_code == 200:
        metrics = response.json()
        print("\n=== Current Metrics ===")
        print(json.dumps(metrics, indent=2))
    else:
        logger.error("Failed to fetch metrics")

def main():
    """Main function."""
    logger.info(f"Starting monitoring test with API URL: {args.api_url}")
    
    # Run the tests
    if args.all or args.random_load:
        generate_random_load(args.num_requests, args.delay)
    
    if args.all or args.cache:
        test_cache_performance()
    
    if args.all or args.errors:
        test_error_handling()
    
    if args.all or args.concurrent:
        test_concurrent_requests(args.threads, args.concurrent_requests)
    
    # Display the metrics
    if args.all or args.display:
        display_metrics()
    
    logger.info("Monitoring test completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the monitoring system")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--random-load", action="store_true", help="Generate random load")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of random requests")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests")
    parser.add_argument("--cache", action="store_true", help="Test cache performance")
    parser.add_argument("--errors", action="store_true", help="Test error handling")
    parser.add_argument("--concurrent", action="store_true", help="Test concurrent requests")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads for concurrent requests")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="Number of requests per thread")
    parser.add_argument("--display", action="store_true", help="Display metrics")
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not any([args.all, args.random_load, args.cache, args.errors, args.concurrent, args.display]):
        args.all = True
    
    main()