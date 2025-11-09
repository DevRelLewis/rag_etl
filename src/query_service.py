from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from openai import OpenAI
from vectorstore import VectorStore


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.7


class Citation(BaseModel):
    content: str
    source_system: str
    classification: str
    file_path: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    query: str


class QueryService:

    def __init__(self, vector_store: VectorStore, openai_config: Dict[str, Any], openai_api_key: str):
        self.vector_store = vector_store
        self.openai_config = openai_config
        self.client = OpenAI(api_key=openai_api_key)

    def process_query(self, request: QueryRequest) -> QueryResponse:
        retrieved_docs = self.vector_store.weighted_search(
            request.query,
            k=request.top_k,
            threshold=request.threshold
        )

        if not retrieved_docs:
            return QueryResponse(
                answer="I couldn't find relevant information to answer your query.",
                citations=[],
                query=request.query
            )

        context = self._build_context(retrieved_docs)
        answer = self._generate_answer(request.query, context)
        citations = self._build_citations(retrieved_docs)

        return QueryResponse(
            answer=answer,
            citations=citations,
            query=request.query
        )

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        context_parts = []

        for doc in docs:
            metadata = doc['metadata']
            classification = metadata['classification']
            source = metadata['source_system']
            content = doc['content']

            context_parts.append(
                f"[{classification}] From {source}: {content}"
            )

        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        system_prompt = """You are a helpful assistant that answers questions based on provided resume and career documents. 
        Use only the information provided in the context to answer questions. 
        If you cannot answer based on the context, say so clearly.
        Be concise and professional."""

        user_prompt = f"""Context information:
{context}

Question: {query}

Please provide a clear, concise answer based only on the context information provided."""

        try:
            response = self.client.chat.completions.create(
                model=self.openai_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.openai_config.get('temperature', 0.1),
                max_tokens=self.openai_config.get('max_tokens', 500)
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _build_citations(self, docs: List[Dict[str, Any]]) -> List[Citation]:
        citations = []

        for doc in docs:
            metadata = doc['metadata']
            citation = Citation(
                content=doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                source_system=metadata['source_system'],
                classification=metadata['classification'],
                file_path=metadata['file_path'],
                score=doc.get('weighted_score', doc.get('score', 0.0))
            )
            citations.append(citation)

        return citations