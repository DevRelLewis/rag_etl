import os
import yaml
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vectorstore import VectorStore
from multisource_etl import MultiSourceETL
from query_service import QueryService, QueryRequest, QueryResponse


class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None


class StatsResponse(BaseModel):
    total_documents: int
    index_size: int
    source_breakdown: dict


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_app():
    config = load_config()

    app = FastAPI(
        title=config['app']['name'],
        version=config['app']['version']
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    vector_store = VectorStore(
        embedding_model=config['embedding']['model_name'],
        index_path=config['vectorstore']['index_path'],
        dimension=config['vectorstore']['dimension']
    )

    etl_service = MultiSourceETL(config, vector_store)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    query_service = QueryService(vector_store, config['openai'], openai_api_key)

    @app.get("/")
    def read_root():
        return {"message": "Portfolio RAG API is running", "version": config['app']['version']}

    @app.post("/api/query", response_model=QueryResponse)
    def query_documents(request: QueryRequest):
        try:
            return query_service.process_query(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

    # Add the RAG endpoint that your frontend expects
    @app.post("/api/rag/query", response_model=QueryResponse)
    def rag_query_documents(request: QueryRequest):
        try:
            return query_service.process_query(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

    @app.post("/api/upload", response_model=UploadResponse)
    async def upload_document(
            file: UploadFile = File(...),
            source_system: str = "ats"
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.filename.endswith(('.txt', '.md', '.json')):
            raise HTTPException(status_code=400, detail="Only .txt, .md, and .json files are supported")

        try:
            content = await file.read()
            content_str = content.decode('utf-8')

            success = etl_service.process_uploaded_file(content_str, file.filename, source_system)

            if success:
                return UploadResponse(
                    success=True,
                    message="File processed and indexed successfully",
                    filename=file.filename
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to process file")

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")

    @app.post("/build_index")
    def build_index():
        try:
            data_sources_path = os.path.join(os.path.dirname(__file__), '..', 'data_sources')
            etl_service.process_data_sources(data_sources_path)
            stats = vector_store.get_stats()
            return {
                "message": "Index built successfully",
                "stats": stats
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Index building error: {str(e)}")

    @app.get("/stats", response_model=StatsResponse)
    def get_stats():
        try:
            stats = vector_store.get_stats()
            return StatsResponse(**stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "server:app",
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['reload']
    )