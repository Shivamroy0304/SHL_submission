from __future__ import annotations
import hashlib
import json, logging, os, re, requests
from pathlib import Path
from typing import Any
import chromadb

logger = logging.getLogger(__name__)


class _LocalEmbeddingFunction:
    """Small deterministic embedding function with no external downloads."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

class CatalogManager:
    def __init__(self):
        self.catalog_url = os.getenv("CATALOG_URL",
            "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json")
        self.cache_path = Path(os.getenv("CATALOG_CACHE_PATH",
            "./catalog_data/shl_catalog.json"))
        self.startup_complete = False
        self.catalog_items = []
        self._by_name = {}
        self._collection = None

    def initialize(self):
        try:
            self._ensure_cache_exists()
            try:
                raw_data = self._load_cached_catalog()
            except Exception:
                logger.info("Cache bad, re-downloading...")
                self.cache_path.unlink(missing_ok=True)
                self._ensure_cache_exists()
                raw_data = self._load_cached_catalog()
            self.catalog_items = self._parse_individual_assessments(raw_data)
            self._by_name = {i["name"].lower(): i for i in self.catalog_items}
            self._init_chroma()
            self._index_catalog()
            logger.info("Assessments loaded: %s", len(self.catalog_items))
        except Exception as exc:
            logger.error("Catalog init failed: %s", exc, exc_info=True)
        finally:
            self.startup_complete = True

    def _ensure_cache_exists(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            return
        logger.info("Downloading catalog...")
        resp = requests.get(self.catalog_url, timeout=60)
        resp.raise_for_status()
        content = re.sub(r'[\x00-\x1f\x7f]', '', resp.text)
        json.loads(content)  # validate
        self.cache_path.write_text(content, encoding="utf-8")

    def _load_cached_catalog(self):
        content = self.cache_path.read_text(encoding="utf-8", errors="replace")
        content = re.sub(r'[\x00-\x1f\x7f]', '', content)
        return json.loads(content)

    def _init_chroma(self):
        client = chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection(
            name="shl_assessments",
            embedding_function=_LocalEmbeddingFunction()
        )

    def _index_catalog(self):
        if not self._collection or not self.catalog_items:
            return
        texts = [self._build_embedding_text(i) for i in self.catalog_items]
        metadatas = [self._build_metadata(i) for i in self.catalog_items]
        ids = [f"a-{i}" for i in range(len(self.catalog_items))]
        batch = 50
        for s in range(0, len(texts), batch):
            try:
                self._collection.add(
                    documents=texts[s:s+batch],
                    metadatas=metadatas[s:s+batch],
                    ids=ids[s:s+batch]
                )
            except Exception as e:
                logger.error("Batch failed [%s]: %s", s, e)
        logger.info("Indexing complete: %s assessments", len(self.catalog_items))

    def search(self, query: str, n_results: int = 20) -> list[dict]:
        if not query.strip() or not self._collection:
            return []
        try:
            r = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, len(self.catalog_items)))
            )
            return [self._meta_to_assessment(m) 
                    for m in (r.get("metadatas") or [[]])[0]]
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def get_by_names(self, names: list[str]) -> list[dict]:
        return [self._by_name[n.strip().lower()] 
                for n in names 
                if n.strip().lower() in self._by_name]

    def get_all(self) -> list[dict]:
        return list(self.catalog_items)

    def _build_embedding_text(self, item: dict) -> str:
        return (f"{item['name']}. {item['description']}. "
                f"Type: {item['test_type']}. "
                f"Levels: {item.get('job_levels',[])}.")

    def _build_metadata(self, item: dict) -> dict:
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"],
            "description": item.get("description","")[:500],
            "job_levels": ", ".join(item.get("job_levels",[])),
            "languages": ", ".join(item.get("languages",[])),
            "is_report": str(item.get("is_report", False))
        }

    def _meta_to_assessment(self, m: dict) -> dict:
        return {
            "name": m.get("name",""),
            "url": m.get("url",""),
            "test_type": m.get("test_type",""),
            "description": m.get("description",""),
            "job_levels": [x.strip() for x in 
                          m.get("job_levels","").split(",") if x.strip()],
            "languages": [x.strip() for x in 
                         m.get("languages","").split(",") if x.strip()],
            "is_report": m.get("is_report","False") == "True"
        }

    def _parse_individual_assessments(self, data):
        records = self._collect_records(data)
        seen, result = set(), []
        for r in records:
            p = self._normalize(r)
            if p and p["name"].lower() not in seen:
                seen.add(p["name"].lower())
                result.append(p)
        return result

    def _collect_records(self, node):
        found = []
        if isinstance(node, dict):
            if self._is_product(node): found.append(node)
            for v in node.values(): found.extend(self._collect_records(v))
        elif isinstance(node, list):
            for v in node: found.extend(self._collect_records(v))
        return found

    def _is_product(self, node):
        keys = {k.lower() for k in node}
        return (any(k in keys for k in ("name","title","productname")) and
                any(k in keys for k in ("url","producturl","link")) and
                any(k in keys for k in ("description","shortdescription")))

    def _normalize(self, node):
        name = self._get(node,["name","title","productName"])
        url  = self._get(node,["url","productUrl","link"])
        desc = self._get(node,["description","shortDescription","longDescription"])
        if not name or not url or "shl.com" not in str(url): return None
        ln = str(name).lower()
        if self._is_bundle(node, ln): return None
        return {
            "name": str(name).strip(),
            "url":  str(url).strip(),
            "test_type": self._get_type(node),
            "description": str(desc or "").strip(),
            "job_levels": self._get_list(node,["jobLevels","job_levels","levels"]),
            "languages":  self._get_list(node,["languages","language"]),
            "is_report": "report" in ln
        }

    def _is_bundle(self, node, ln):
        if "report" in ln: return False
        if any(m in ln for m in ("job solution","solution bundle","bundled")):
            return True
        cat = str(self._get(node,["category","productCategory","type"]) or "").lower()
        return "job solution" in cat or "bundle" in cat

    def _get_type(self, node):
        raw = self._get(node,["testType","test_type","assessmentType","typeCode"])
        return str(raw).strip() if raw else "UNKNOWN"

    def _get_list(self, node, keys):
        raw = self._get(node, keys)
        if raw is None: return []
        if isinstance(raw, str): return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, list):
            out = []
            for i in raw:
                if isinstance(i, str) and i.strip(): out.append(i.strip())
                elif isinstance(i, dict):
                    n = self._get(i,["name","label","value"])
                    if isinstance(n,str) and n.strip(): out.append(n.strip())
            return out
        return []

    def _get(self, node, keys):
        for k in keys:
            if k in node: return node[k]
            for ek in node:
                if ek.lower() == k.lower(): return node[ek]
        return None
