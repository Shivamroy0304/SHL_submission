"""Catalog loading, parsing, indexing, and retrieval for SHL assessments."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class CatalogManager:
    """Manages SHL catalog cache, parsing, Chroma indexing, and retrieval."""

    def __init__(self) -> None:
        """Initialize manager with environment-based paths and defaults."""
        self.catalog_url: str = os.getenv(
            "CATALOG_URL",
            "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
        )
        self.cache_path: Path = Path(
            os.getenv("CATALOG_CACHE_PATH", "./catalog_data/shl_catalog.json")
        )
        self.collection_name: str = "shl_assessments"
        self.startup_complete: bool = False
        self.catalog_items: list[dict[str, Any]] = []
        self.chroma: Chroma | None = None
        self.embeddings: HuggingFaceEmbeddings | None = None
        self._by_name: dict[str, dict[str, Any]] = {}

    def initialize(self) -> None:
        """Load catalog, bootstrap embeddings/vector DB, and mark startup complete."""
        try:
            self._ensure_cache_exists()
            try:
                raw_data = self._load_cached_catalog()
            except (json.JSONDecodeError, Exception):
                logger.info("Cache corrupted - deleting and re-downloading...")
                if self.cache_path.exists():
                    self.cache_path.unlink()
                self._ensure_cache_exists()
                raw_data = self._load_cached_catalog()
            self.catalog_items = self._parse_individual_assessments(raw_data)
            self._by_name = {item["name"].lower(): item for item in self.catalog_items}
            self._init_embeddings()
            self._init_chroma()
            logger.info("Assessments loaded: %s", len(self.catalog_items))
        except Exception as exc:
            logger.error("Catalog initialization failed: %s", exc, exc_info=True)
            self.catalog_items = []
            self._by_name = {}
            self._init_embeddings(best_effort=True)
            self._init_chroma(best_effort=True)
        finally:
            self.startup_complete = True

    def _ensure_cache_exists(self) -> None:
        """Download catalog JSON once if local cache is missing."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            return
        logger.info("Catalog cache missing; downloading from source URL.")
        try:
            import re

            response = requests.get(self.catalog_url, timeout=40)
            response.raise_for_status()
            content = response.text
            content = re.sub(r'[\x00-\x1f\x7f]', '', content)
            json.loads(content)
            self.cache_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to download catalog: %s", exc, exc_info=True)
            raise

    def _load_cached_catalog(self) -> Any:
        """Load cached catalog JSON, stripping invalid control characters."""
        import re

        try:
            content = self.cache_path.read_text(encoding="utf-8", errors="replace")
            # Remove invalid JSON control characters (keep \t \n \r which are valid)
            content = re.sub(r'[\x00-\x1f\x7f]', '', content)
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed, deleting cache to force re-download: %s", exc)
            # Delete corrupted cache so it re-downloads fresh next startup
            try:
                self.cache_path.unlink()
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.error("Failed to read cached catalog: %s", exc, exc_info=True)
            raise

    def _parse_individual_assessments(self, data: Any) -> list[dict[str, Any]]:
        """Extract individual test solutions and report items from raw catalog JSON."""
        records = self._collect_candidate_records(data)
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in records:
            parsed = self._normalize_record(item)
            if not parsed:
                continue
            key = parsed["name"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(parsed)
        return normalized

    def _collect_candidate_records(self, node: Any) -> list[dict[str, Any]]:
        """Recursively collect dict nodes likely representing catalog products."""
        found: list[dict[str, Any]] = []
        if isinstance(node, dict):
            if self._is_product_like(node):
                found.append(node)
            for value in node.values():
                found.extend(self._collect_candidate_records(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(self._collect_candidate_records(value))
        return found

    def _is_product_like(self, node: dict[str, Any]) -> bool:
        """Return True for entries that look like assessment product records."""
        keys = {k.lower() for k in node.keys()}
        has_name = any(k in keys for k in ("name", "title", "productname", "product_name"))
        has_url = any(k in keys for k in ("url", "producturl", "product_url", "link"))
        has_desc = any(
            k in keys for k in ("description", "shortdescription", "longdescription")
        )
        return has_name and has_url and has_desc

    def _normalize_record(self, node: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize one raw record into expected assessment schema."""
        name = self._pick_first(node, ["name", "title", "productName", "product_name"])
        url = self._pick_first(node, ["url", "productUrl", "product_url", "link"])
        description = self._pick_first(
            node, ["description", "shortDescription", "longDescription"]
        )
        if not name or not url or "shl.com" not in str(url):
            return None

        lower_name = str(name).lower()
        if self._is_prepackaged_solution(node, lower_name):
            return None

        test_type = self._extract_test_type(node)
        job_levels = self._extract_list_value(node, ["jobLevels", "job_levels", "levels"])
        languages = self._extract_list_value(node, ["languages", "language"])
        is_report = "report" in lower_name or "development report" in lower_name

        return {
            "name": str(name).strip(),
            "url": str(url).strip(),
            "test_type": test_type,
            "description": str(description).strip(),
            "job_levels": job_levels,
            "languages": languages,
            "is_report": is_report,
        }

    def _is_prepackaged_solution(self, node: dict[str, Any], lower_name: str) -> bool:
        """Filter out bundled/job-solution packages while keeping reports."""
        if "report" in lower_name:
            return False
        bundle_markers = ("job solution", "solution bundle", "package", "bundled")
        if any(marker in lower_name for marker in bundle_markers):
            return True
        category = str(
            self._pick_first(node, ["category", "productCategory", "type", "productType"]) or ""
        ).lower()
        return "job solution" in category or "bundle" in category

    def _extract_test_type(self, node: dict[str, Any]) -> str:
        """Extract test type with robust fallback to letter-like values."""
        raw = self._pick_first(
            node,
            [
                "testType",
                "test_type",
                "assessmentType",
                "assessment_type",
                "typeCode",
                "type_code",
            ],
        )
        if raw:
            return str(raw).strip()
        type_hint = str(self._pick_first(node, ["type", "productType", "category"]) or "")
        letters = []
        for part in type_hint.replace("/", ",").split(","):
            clean = part.strip().upper()
            if len(clean) <= 3 and clean.isalpha():
                letters.append(clean)
        return ",".join(letters) if letters else "UNKNOWN"

    def _extract_list_value(self, node: dict[str, Any], keys: list[str]) -> list[str]:
        """Extract a string list from possible keys and normalize values."""
        raw = self._pick_first(node, keys)
        if raw is None:
            return []
        if isinstance(raw, str):
            return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, list):
            result = []
            for item in raw:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
                elif isinstance(item, dict):
                    name = self._pick_first(item, ["name", "label", "value"])
                    if isinstance(name, str) and name.strip():
                        result.append(name.strip())
            return result
        return []

    def _pick_first(self, node: dict[str, Any], keys: list[str]) -> Any:
        """Return first existing key value across case-variant candidates."""
        for key in keys:
            if key in node:
                return node[key]
            for existing in node.keys():
                if existing.lower() == key.lower():
                    return node[existing]
        return None

    def _init_embeddings(self, best_effort: bool = False) -> None:
        """Initialize local HuggingFace embedding model. No API key needed."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embeddings initialized with all-MiniLM-L6-v2")
        except Exception as exc:
            logger.error("Failed to initialize embeddings: %s", exc, exc_info=True)
            if not best_effort:
                raise

    def _init_chroma(self, best_effort: bool = False) -> None:
        """Initialize in-memory Chroma collection and index catalog data."""
        try:
            client = chromadb.EphemeralClient()
            self.chroma = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            self._index_catalog()
        except Exception as exc:
            logger.error("Failed to initialize Chroma: %s", exc, exc_info=True)
            if not best_effort:
                raise

    def _safe_count_collection(self) -> int:
        """Get collection document count safely, returning zero on failure."""
        if self.chroma is None:
            return 0
        try:
            collection = self.chroma._collection
            return int(collection.count()) if collection else 0
        except Exception as exc:
            logger.error("Failed to count Chroma collection: %s", exc, exc_info=True)
            return 0

    def _index_catalog(self) -> None:
        """Batch-index parsed catalog assessments into persistent Chroma."""
        if self.chroma is None:
            return
        if not self.catalog_items:
            logger.info("No catalog items available for indexing.")
            return

        texts = [self._build_embedding_text(item) for item in self.catalog_items]
        metadatas = [self._build_metadata(item) for item in self.catalog_items]
        ids = [f"assessment-{idx}" for idx, _ in enumerate(self.catalog_items)]
        batch_size = 64

        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            try:
                self.chroma.add_texts(
                    texts=texts[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )
            except Exception as exc:
                logger.error("Batch indexing failed [%s:%s]: %s", start, end, exc, exc_info=True)
                continue
        logger.info("Chroma indexing complete for %s assessments.", len(self.catalog_items))

    def _build_embedding_text(self, item: dict[str, Any]) -> str:
        """Compose dense retrieval text used for semantic embedding."""
        return (
            f"{item['name']}. {item['description']}. "
            f"Test type: {item['test_type']}. "
            f"Levels: {item.get('job_levels', [])}. "
            f"Languages: {item.get('languages', [])}"
        )

    def _build_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """Construct Chroma metadata payload for each assessment."""
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"],
            "job_levels": ", ".join(item.get("job_levels", [])),
            "languages": ", ".join(item.get("languages", [])),
            "is_report": bool(item.get("is_report", False)),
            "description": item.get("description", ""),
        }

    def search(self, query: str, n_results: int = 20) -> list[dict[str, Any]]:
        """Return top semantically similar assessment candidates."""
        if not query.strip():
            return []
        if self.chroma is None:
            return []
        try:
            docs = self.chroma.similarity_search(query=query, k=n_results)
            return [self._doc_to_assessment(doc.metadata) for doc in docs]
        except Exception as exc:
            logger.error("Catalog search failed: %s", exc, exc_info=True)
            return []

    def _doc_to_assessment(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert Chroma metadata record to normalized assessment dict."""
        return {
            "name": metadata.get("name", ""),
            "url": metadata.get("url", ""),
            "test_type": metadata.get("test_type", "UNKNOWN"),
            "description": metadata.get("description", ""),
            "job_levels": self._to_list(metadata.get("job_levels", "")),
            "languages": self._to_list(metadata.get("languages", "")),
            "is_report": bool(metadata.get("is_report", False)),
        }

    def _to_list(self, value: Any) -> list[str]:
        """Normalize potentially serialized metadata list representation."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def get_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        """Return full catalog rows for given exact names (case-insensitive)."""
        results: list[dict[str, Any]] = []
        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            try:
                item = self._by_name.get(key)
                if item:
                    results.append(item)
            except Exception as exc:
                logger.error("Lookup failed for name '%s': %s", name, exc, exc_info=True)
        return results

    def get_all(self) -> list[dict[str, Any]]:
        """Return full parsed catalog data."""
        try:
            return list(self.catalog_items)
        except Exception as exc:
            logger.error("Failed to return all catalog items: %s", exc, exc_info=True)
            return []
