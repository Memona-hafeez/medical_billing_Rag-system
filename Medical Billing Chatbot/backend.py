from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import time
from dotenv import load_dotenv

# Import your existing code components
from connect_memory import route_query

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Make sure required API keys are available
required_keys = ["GROQ_API_KEY", "HF_TOKEN"]
for key in required_keys:
    if not os.getenv(key):
        logger.warning(f"Environment variable {key} is not set!")

# Create FastAPI app
app = FastAPI(
    title="Medical Billing Chatbot API",
    description="API for answering healthcare data queries using LangChain and Groq",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # In production, specify your Streamlit app's URL instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    response: str
    processing_time: float

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Medical Billing Chatbot API is running"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query about healthcare data.
    
    The system will automatically route the query to either:
    - A RAG (Retrieval Augmented Generation) system for simple lookups
    - A Pandas agent for queries requiring computation
    """
    start_time = time.time()
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing query: {query}")
        
        # Use your existing route_query function to process the query
        result = route_query(query)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return QueryResponse(
            response=result,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your query: {str(e)}"
        )

# Additional endpoints could be added here for dataset statistics, etc.

@app.get("/dataset/stats")
async def get_dataset_stats():
    """Get basic statistics about the healthcare dataset."""
    try:
        # This is a placeholder - you would calculate these from your dataframe
        # You can implement this by adding functions to your existing code
        from create_memory import df
        
        stats = {
            "total_patients": len(df),
            "hospitals": df["Hospital"].nunique(),
            "average_billing": float(df["Billing Amount"].mean()),
            "most_common_condition": df["Medical Condition"].value_counts().index[0],
            "admission_types": df["Admission Type"].value_counts().to_dict(),
            "gender_distribution": df["Gender"].value_counts().to_dict()
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset statistics: {str(e)}"
        )

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Set to False in production
    )