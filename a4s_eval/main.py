"""Main entry point for the A4S Evaluation API service.

This module initializes and configures the FastAPI application that serves
as the main entry point for the A4S (AI Audit as a Service) evaluation service.
"""

from fastapi import FastAPI

# Initialize FastAPI application instance
app = FastAPI(
    title="A4S Evaluation API",
    description="API service for evaluation module of A4S",
    version="0.0.1"
)


@app.get("/")
def read_root():
    """Root endpoint that returns a simple health check response.

    Returns:
        dict: A simple hello world response indicating the service is running.
    """
    return {"Hello": "World"}
