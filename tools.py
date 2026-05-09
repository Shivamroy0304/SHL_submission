"""LangChain tools that expose catalog search and lookup operations."""

from langchain.tools import Tool

from catalog import CatalogManager


def make_catalog_tools(catalog_manager: CatalogManager) -> list[Tool]:
    """Build tool list used by AgentExecutor during compare/recommend support."""
    return [
        Tool(
            name="search_assessments",
            func=lambda query: str(catalog_manager.search(query, n_results=20)),
            description=(
                "Search SHL catalog by semantic similarity. "
                "Input: descriptive hiring query. Output: list of up to 20 assessments."
            ),
        ),
        Tool(
            name="get_assessment_details",
            func=lambda names_csv: str(
                catalog_manager.get_by_names([n.strip() for n in names_csv.split(",") if n.strip()])
            ),
            description=(
                "Get full catalog details for exact assessment names. "
                "Input: comma-separated names. Output: detailed assessment records."
            ),
        ),
    ]
