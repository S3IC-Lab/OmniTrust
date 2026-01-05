"""Util that calls Google Search using the Serper.dev API."""
import pdb
import requests
import asyncio
import aiohttp
import yaml
import os
from config import GET_API, LOAD_API_CONFIG


"""Util that calls Google Search using the Serper.dev API."""

import asyncio
import aiohttp
import json
import os
from typing import Any, Dict, List, Union


class GoogleSerperAPIWrapper:
    """Wrapper around the Serper.dev Google Search API.

    You can create a free API key at https://serper.dev.

    Usage:
        export SERPER_API_KEY="xxxx"
        from google_serper_api import GoogleSerperAPIWrapper
        search = GoogleSerperAPIWrapper()
        asyncio.run(search.run(...))
    """

    def __init__(
        self,
        snippet_cnt: int = 10,
        serper_api_key: str | None = None,
        gl: str = "us",
        hl: str = "en",
        max_concurrency: int = 5,
    ) -> None:
        self.k = snippet_cnt
        self.gl = gl
        self.hl = hl
        self.max_concurrency = max_concurrency

        
        model_name = 'google-serper'
        api_key, api_url = GET_API(model_name)
        if not api_url:
            raise ValueError(f"api_url is missing for model: {model_name}")
        if not api_key:
            raise ValueError(f"api_key is missing for model: {model_name}")

        self.serper_api_key = api_key

    async def _google_serper_search_results(
        self,
        session: aiohttp.ClientSession,
        search_term: str,
        gl: str,
        hl: str,
    ) -> Dict[str, Any]:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }

        payload = {"q": search_term, "gl": gl, "hl": hl}

        try:
            async with session.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=30,
            ) as response:
                text = await response.text()
                if response.status != 200:
                    print(
                        "[Serper HTTP ERROR]",
                        response.status,
                        text[:300],
                    )
                    return {
                        "error": f"HTTP {response.status}",
                        "raw": text,
                        "query": search_term,
                    }

                try:
                    return json.loads(text)
                except Exception as e:
                    print("[Serper JSON PARSE ERROR]", e, text[:300])
                    return {
                        "error": f"JSON parse error: {e}",
                        "raw": text[:300],
                        "query": search_term,
                    }

        except Exception as e:
            print("[Serper REQUEST EXCEPTION]", type(e), repr(e), "query:", search_term)
            return {
                "error": f"request exception: {type(e).__name__}: {e}",
                "query": search_term,
            }

    def _parse_results(self, results: Any) -> List[Dict[str, str]]:
        if isinstance(results, Exception):
            return [
                {
                    "content": f"Search error: {type(results).__name__}: {results}",
                    "source": "None",
                }
            ]
        if isinstance(results, dict) and "error" in results and "organic" not in results:
            return [
                {
                    "content": f"Search error: {results.get('error')}",
                    "source": "None",
                }
            ]
        if not isinstance(results, dict):
            return [
                {
                    "content": f"Unexpected result type: {type(results).__name__}",
                    "source": "None",
                }
            ]

        snippets: List[Dict[str, str]] = []

        answer_box = results.get("answerBox")
        if answer_box:
            if answer_box.get("answer"):
                element = {"content": answer_box.get("answer"), "source": "None"}
                return [element]
            elif answer_box.get("snippet"):
                element = {
                    "content": answer_box.get("snippet").replace("\n", " "),
                    "source": "None",
                }
                return [element]
            elif answer_box.get("snippetHighlighted"):
                element = {
                    "content": str(answer_box.get("snippetHighlighted")),
                    "source": "None",
                }
                return [element]

        kg = results.get("knowledgeGraph")
        if kg:
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                element = {"content": f"{title}: {entity_type}", "source": "None"}
                snippets.append(element)
            description = kg.get("description")
            if description:
                element = {"content": description, "source": "None"}
                snippets.append(element)
            for attribute, value in kg.get("attributes", {}).items():
                element = {
                    "content": f"{attribute}: {value}",
                    "source": "None",
                }
                snippets.append(element)

        organic_results = results.get("organic", []) or []
        for result in organic_results[: self.k]:
            link = result.get("link", "None")
            if "snippet" in result:
                element = {"content": result["snippet"], "source": link}
                snippets.append(element)
            for attribute, value in result.get("attributes", {}).items():
                element = {
                    "content": f"{attribute}: {value}",
                    "source": link,
                }
                snippets.append(element)

        if len(snippets) == 0:
            element = {
                "content": "No good Google Search Result was found",
                "source": "None",
            }
            return [element]

        snippets = snippets[: int(self.k / 2)]

        return snippets

    async def parallel_searches(
        self,
        search_queries: List[str],
        gl: str,
        hl: str,
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async with aiohttp.ClientSession() as session:
            async def bounded_search(query: str):
                async with sem:
                    return await self._google_serper_search_results(
                        session, query, gl, hl
                    )

            tasks = [bounded_search(query) for query in search_queries]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            clean_results: List[Dict[str, Any]] = []
            for q, r in zip(search_queries, raw_results):
                if isinstance(r, Exception):
                    print(
                        "[Serper GATHER EXCEPTION]",
                        type(r),
                        repr(r),
                        "query:",
                        q,
                    )
                    clean_results.append(
                        {
                            "error": f"gather exception: {type(r).__name__}: {r}",
                            "query": q,
                        }
                    )
                else:
                    clean_results.append(r)

            return clean_results

    async def run(
        self,
        queries: Union[
            str,
            List[str],
            List[List[str]],
        ],
    ) -> List[List[Dict[str, str]]]:
        flattened_queries: List[str] = []

        if isinstance(queries, str):
            flattened_queries = [queries]
        elif isinstance(queries, (list, tuple)):
            for sublist in queries:
                if sublist is None:
                    sublist = ["None", "None"]

                if isinstance(sublist, (list, tuple)):
                    for item in sublist:
                        flattened_queries.append(str(item))
                else:
                    flattened_queries.append(str(sublist))
        else:
            raise TypeError(
                f"Unsupported type for queries: {type(queries).__name__}. "
                "Expected str, List[str], or List[List[str]]."
            )

        results = await self.parallel_searches(
            flattened_queries,
            gl=self.gl,
            hl=self.hl,
        )

        snippets_list: List[List[Dict[str, str]]] = []
        for r in results:
            snippets_list.append(self._parse_results(r))

        snippets_split: List[List[Dict[str, str]]] = []
        i = 0
        while i < len(snippets_list):
            if i + 1 < len(snippets_list):
                snippets_split.append(snippets_list[i] + snippets_list[i + 1])
                i += 2
            else:
                snippets_split.append(snippets_list[i])
                i += 1

        return snippets_split


if __name__ == "__main__":
    async def _test_single():
        search = GoogleSerperAPIWrapper(snippet_cnt=10, max_concurrency=3)
        res = await search.run("What is the capital of the United States?")
        print("Result length:", len(res))
        print("First group:", res[0])

    asyncio.run(_test_single())
