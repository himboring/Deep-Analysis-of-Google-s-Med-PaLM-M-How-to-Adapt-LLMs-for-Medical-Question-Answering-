```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
from typing import Dict, Optional

# Initialize FastAPI app
app = FastAPI(title="Medical Text Summarization API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load T5 model and tokenizer for summarization
try:
    model_name = "t5-small"  # For demo; replace with fine-tuned medical T5 model in production
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    logger.info(f"T5 model ({model_name}) loaded successfully")
except Exception as e:
    logger.error(f"Error loading T5 model: {str(e)}")
    raise

# Pydantic model for request body
class SummaryRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150
    min_length: Optional[int] = 50

# API endpoint for summarizing clinical text
@app.post("/summarize")
async def summarize_text(request: SummaryRequest) -> Dict[str, any]:
    try:
        text = request.text
        max_length = request.max_length
        min_length = request.min_length
        logger.info(f"Processing text for summarization: {text[:50]}...")

        # Validate input
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        if max_length < min_length:
            raise HTTPException(status_code=400, detail="max_length must be greater than min_length")

        # Generate summary
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]["summary_text"]

        return {
            "status": "success",
            "original_text": text,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error processing summarization request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
```

<xaiArtifact artifact_id="c0c3901e-c659-4251-bc50-16be24d2beed" artifact_version_id="45a99352-14bc-4c07-bd52-f60b5fe0c02a" title="Dockerfile" contentType="text/dockerfile">
```dockerfile
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY medical_summary_deployment.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "medical_summary_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

<xaiArtifact artifact_id="5d77c7cd-9207-4111-abe6-cd4f6f959adb" artifact_version_id="f792a63c-41bd-4955-80f1-3bd33c331aff" title="requirements.txt" contentType="text/plain">
```
fastapi==0.68.1
uvicorn==0.15.0
transformers==4.12.5
pydantic==1.8.2
torch==1.9.0
```

### Deployment Instructions
1. **Prerequisites**:
   - Install Docker: Ensure Docker is installed on your system.
   - Python Environment: The code uses Python 3.9, FastAPI, and Hugging Face's `transformers` library with a T5 model.

2. **Setup**:
   - Save the above files (`medical_summary_deployment.py`, `Dockerfile`, `requirements.txt`) in a project directory.
   - Build the Docker image:
     ```bash
     docker build -t medical-summary-api .
     ```
   - Run the Docker container:
     ```bash
     docker run -p 8000:8000 medical-summary-api
     ```

3. **Usage**:
   - Send a POST request to `http://localhost:8000/summarize` with a JSON body:
     ```json
     {
       "text": "Patient presents with a history of type 2 diabetes mellitus, diagnosed 5 years ago. Currently managed with metformin 500 mg twice daily. Recent complaints of fatigue, polyuria, and blurred vision. HbA1c level is 8.2%, indicating poor glycemic control. No history of cardiovascular disease. Plan to adjust medication and recommend lifestyle changes.",
       "max_length": 100,
       "min_length": 30
     }
     ```
   - Example response:
     ```json
     {
       "status": "success",
       "original_text": "Patient presents with a history of type 2 diabetes mellitus, diagnosed 5 years ago. Currently managed with metformin 500 mg twice daily. Recent complaints of fatigue, polyuria, and blurred vision. HbA1c level is 8.2%, indicating poor glycemic control. No history of cardiovascular disease. Plan to adjust medication and recommend lifestyle changes.",
       "summary": "Patient with type 2 diabetes, diagnosed 5 years ago, on metformin. Reports fatigue, polyuria, and blurred vision. HbA1c 8.2% shows poor control. Plan to adjust medication and recommend lifestyle changes."
     }
     ```

4. **Notes**:
   - This example uses `t5-small` for demonstration. In production, replace with a T5 model fine-tuned on medical datasets (e.g., MIMIC-III clinical notes) for better accuracy.
   - The API summarizes clinical text, such as patient notes or discharge summaries, into concise outputs, useful for clinicians or patient reports.
   - Ensure compliance with HIPAA/GDPR when processing sensitive medical data.
   - For production, deploy with orchestration (e.g., Kubernetes) and add authentication to secure the API.
   - To enhance performance, consider integrating with a medical knowledge base or fine-tuning on domain-specific datasets for more precise summaries.

This deployment provides a practical example of a medical text summarization system, leveraging T5 for concise and accurate summaries of clinical notes.