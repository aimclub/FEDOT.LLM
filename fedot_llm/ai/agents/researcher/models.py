from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class Citation(BaseModel):
    page_title: str = Field(...,
                            description="The title of the page where the citation was found."
                            )
    url: str = Field(...,
                     description="The URL source of the page where the citation was found.")
    number: int = Field(...,
                        description="The number of the citation."
                        )
    relevant_passages: List[str] = Field(...,
                                         description="A list of every relevant passage on a single documentation page."
                                         )


class GenerateWithCitations(BaseModel):
    """Generate a response with citations to relevant passages in the documentation."""

    citations: List[Citation] = Field(...,
                                      description="A list of citations to relevant passages in the documentation."
                                      )

    answer: str = Field(...,
                        description="A plain text answer, formatted as Markdown."
                        )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )


class RewriteQuestion(BaseModel):
    """Rewrite a question to be more optimized for vectorstore retrieval."""
    question: str = Field(description="A re-phrased question")