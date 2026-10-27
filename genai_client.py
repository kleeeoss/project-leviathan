import os
import logging
from typing import Optional
from google import genai

logger = logging.getLogger("genai_factory")


def get_vertex_client(strict_production: bool = False) -> Optional[genai.Client]:
    """
    Initializes a Google GenAI Client connecting strictly to Google Cloud Vertex AI via ADC.
    Zero Developer API keys or AI Studio fallbacks.
    - If strict_production=True: raises RuntimeError if project or ADC authentication fails.
    - If strict_production=False: returns None when run in offline unit test environments.
    """
    project_id = os.getenv("VERTEX_PROJECT_ID", "project-6d3877ae-baae-4159-a77")
    location = os.getenv("VERTEX_LOCATION", "global")

    try:
        return genai.Client(vertexai=True, project=project_id, location=location)
    except Exception as e:
        if strict_production:
            raise RuntimeError(
                f"Actionable Vertex AI configuration error for project \"{project_id}\" "
                f"in location \"{location}\": {e}. Verify ADC via \"gcloud auth application-default print-access-token\"."
            ) from e
        logger.warning(f"Vertex AI Client offline degradation for unit tests: {e}")
        return None
