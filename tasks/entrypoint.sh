#!/bin/bash

# Entrypoint script for A4S Evaluation Service

set -e

# Function to wait for dependencies
wait_for_dependencies() {
    # Wait for Redis
    if [ -n "$REDIS_HOST" ]; then
        echo "Waiting for Redis at $REDIS_HOST:${REDIS_PORT:-6379}..."
        timeout 30 bash -c "until nc -z $REDIS_HOST ${REDIS_PORT:-6379}; do sleep 1; done" || {
            echo "Redis not reachable at $REDIS_HOST:${REDIS_PORT:-6379}"
        }
    fi
    
    # Wait for RabbitMQ
    if [ -n "$MQ_HOST" ]; then
        echo "Waiting for RabbitMQ at $MQ_HOST..."
        timeout 30 bash -c "until nc -z $MQ_HOST $MQ_PORT; do sleep 1; done" || {
            echo "RabbitMQ not reachable at $MQ_HOST:$MQ_PORT"
        }
    fi
}

# Function to start Celery worker
start_worker() {
    echo "Starting Celery worker..."
    exec celery -A a4s_eval.celery_worker worker --loglevel=info --concurrency=1 --hostname=worker@%h
}

# Main function
main() {
    # Wait for dependencies
    wait_for_dependencies

    start_worker
}

# Execute main function with all arguments
main "$@"
