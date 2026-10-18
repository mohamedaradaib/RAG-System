import pydantic


class RAGChunk(pydantic.BaseModel):
    text: str
    page: str | int | None = None


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[RAGChunk]
    source_id: str = None


class RAGCitation(pydantic.BaseModel):
    id: int
    source: str
    text: str
    chunk_index: int | None = None
    page: str | int | None = None
    score: float


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    citations: list[RAGCitation] = pydantic.Field(default_factory=list)
    error: str | None = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int
    error: str | None = None


class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    error: str | None = None