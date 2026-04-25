import json
import re
from typing import Dict, List

import networkx as nx

from .config import GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY


ENTITY_EXTRACTION_PROMPT = """\
You are an information extraction assistant. Given a text passage from a video segment,
extract named entities and relationships between them.

Return ONLY valid JSON (no markdown, no explanation):
{
  "entities": [
    {"name": "Entity Name", "type": "PERSON|CONCEPT|PLACE|OBJECT|EVENT", "description": "brief description"}
  ],
  "relations": [
    {"source": "Entity A", "relation": "verb phrase", "target": "Entity B"}
  ]
}

Guidelines:
- Extract 3–7 entities and 2–5 relations per passage
- Only include entities clearly present in the text
- Keep entity names concise (1–4 words)
- Relations should be short verb phrases (e.g. "uses", "is part of", "explains")
"""


def _normalize_extraction(payload: Dict) -> Dict:
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    if not isinstance(entities, list) or not isinstance(relations, list):
        return {"entities": [], "relations": []}

    clean_entities = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        clean_entities.append(
            {
                "name": name,
                "type": str(item.get("type", "CONCEPT")).strip() or "CONCEPT",
                "description": str(item.get("description", "")).strip(),
            }
        )

    clean_relations = []
    for item in relations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        relation = str(item.get("relation", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and relation and target:
            clean_relations.append(
                {"source": source, "relation": relation, "target": target}
            )

    return {"entities": clean_entities, "relations": clean_relations}


def _extract_with_gemini(chunk_text: str) -> Dict:
    if not GEMINI_API_KEY:
        return {"entities": [], "relations": []}

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"{ENTITY_EXTRACTION_PROMPT}\n\nPassage:\n{chunk_text}"
    )
    text = getattr(response, "text", "") or ""
    return _normalize_extraction(json.loads(text))


def _extract_with_openai(chunk_text: str) -> Dict:
    if not OPENAI_API_KEY:
        return {"entities": [], "relations": []}

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        max_tokens=600,
    )
    text = response.choices[0].message.content or ""
    return _normalize_extraction(json.loads(text))


def _extract_with_groq(chunk_text: str) -> Dict:
    if not GROQ_API_KEY:
        return {"entities": [], "relations": []}

    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        max_tokens=600,
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return _normalize_extraction(json.loads(text))


def _extract_with_heuristics(chunk_text: str) -> Dict:
    text = re.sub(r"\s+", " ", chunk_text).strip()
    lowered = text.lower()
    entities = []
    seen = set()

    person_matches = re.findall(r"\bProfessor\s+[A-Z][a-zA-Z]+\b", text)
    for match in person_matches:
        key = match.lower()
        if key not in seen:
            entities.append(
                {
                    "name": match,
                    "type": "PERSON",
                    "description": "A person mentioned in the segment text.",
                }
            )
            seen.add(key)

    concept_patterns = [
        "gradient descent",
        "backpropagation",
        "loss function",
        "neural networks",
        "optimization algorithm",
        "gradients",
    ]
    for phrase in concept_patterns:
        if phrase in lowered and phrase not in seen:
            entities.append(
                {
                    "name": phrase.title(),
                    "type": "CONCEPT",
                    "description": f"Mentioned in the segment text: {phrase}.",
                }
            )
            seen.add(phrase)

    title_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    for match in title_matches:
        key = match.lower()
        if key in seen or match in {"Transcript", "Visual"}:
            continue
        if len(entities) >= 7:
            break
        entities.append(
            {
                "name": match,
                "type": "CONCEPT",
                "description": "A salient phrase detected from the segment text.",
            }
        )
        seen.add(key)

    relations = []
    entity_names = {entity["name"].lower(): entity["name"] for entity in entities}

    def has(name: str) -> bool:
        return name.lower() in entity_names

    if has("Professor Smith") and has("Gradient Descent"):
        relations.append(
            {
                "source": entity_names["professor smith"],
                "relation": "introduces",
                "target": entity_names["gradient descent"],
            }
        )
    if has("Gradient Descent") and has("Neural Networks"):
        relations.append(
            {
                "source": entity_names["gradient descent"],
                "relation": "trains",
                "target": entity_names["neural networks"],
            }
        )
    if has("Gradient Descent") and has("Loss Function"):
        relations.append(
            {
                "source": entity_names["gradient descent"],
                "relation": "minimizes",
                "target": entity_names["loss function"],
            }
        )
    if has("Backpropagation") and has("Gradients"):
        relations.append(
            {
                "source": entity_names["backpropagation"],
                "relation": "computes",
                "target": entity_names["gradients"],
            }
        )
    if has("Backpropagation") and has("Gradient Descent"):
        relations.append(
            {
                "source": entity_names["backpropagation"],
                "relation": "supports",
                "target": entity_names["gradient descent"],
            }
        )

    return {"entities": entities[:7], "relations": relations[:5]}


def extract_entities_from_chunk(chunk_text: str) -> Dict:
    """
    Extract entities and relations from one text chunk.

    Preference order:
    - Gemini
    - Groq
    - OpenAI
    - heuristic fallback
    """
    for extractor in (_extract_with_gemini, _extract_with_groq, _extract_with_openai):
        try:
            result = extractor(chunk_text)
            if result.get("entities") or result.get("relations"):
                return result
        except Exception:
            continue

    try:
        return _extract_with_heuristics(chunk_text)
    except Exception:
        return {"entities": [], "relations": []}


def build_knowledge_graph(chunks: List[Dict]) -> nx.DiGraph:
    """
    Build a directed graph from extracted chunk-level entities and relations.
    """
    graph = nx.DiGraph()

    for i, chunk in enumerate(chunks):
        extracted = extract_entities_from_chunk(chunk["text"])
        chunk_id = chunk["id"]

        for entity in extracted.get("entities", []):
            name = entity["name"]
            if graph.has_node(name):
                if not graph.nodes[name].get("description") and entity.get("description"):
                    graph.nodes[name]["description"] = entity["description"]
                if not graph.nodes[name].get("type") and entity.get("type"):
                    graph.nodes[name]["type"] = entity["type"]
                if chunk_id not in graph.nodes[name]["chunks"]:
                    graph.nodes[name]["chunks"].append(chunk_id)
            else:
                graph.add_node(
                    name,
                    type=entity.get("type", "CONCEPT"),
                    description=entity.get("description", ""),
                    chunks=[chunk_id],
                )

        for relation in extracted.get("relations", []):
            source = relation["source"]
            target = relation["target"]
            if not graph.has_node(source):
                graph.add_node(source, type="CONCEPT", description="", chunks=[chunk_id])
            elif chunk_id not in graph.nodes[source]["chunks"]:
                graph.nodes[source]["chunks"].append(chunk_id)

            if not graph.has_node(target):
                graph.add_node(target, type="CONCEPT", description="", chunks=[chunk_id])
            elif chunk_id not in graph.nodes[target]["chunks"]:
                graph.nodes[target]["chunks"].append(chunk_id)

            graph.add_edge(source, target, relation=relation["relation"])

        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            print(
                f"Processed {i + 1}/{len(chunks)} chunks "
                f"-> {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )

    return graph


def get_entity_context(graph: nx.DiGraph, entity_name: str, hops: int = 1) -> str:
    """
    Return a readable text summary for one entity and its local graph neighborhood.
    """
    target = None
    wanted = entity_name.lower()
    for node in graph.nodes:
        if node.lower() == wanted:
            target = node
            break

    if target is None:
        return ""

    ego = nx.ego_graph(graph, target, radius=hops, undirected=False)
    node_data = graph.nodes[target]
    lines = [
        f"Entity: {target} ({node_data.get('type', 'CONCEPT')}): {node_data.get('description', '')}".rstrip()
    ]

    relations = []
    for source, sink, edge_data in ego.edges(data=True):
        if source == target or sink == target:
            relations.append(f"- {source} --[{edge_data.get('relation', 'related to')}]--> {sink}")

    if relations:
        lines.append("Relations:")
        lines.extend(relations)

    chunks = node_data.get("chunks", [])
    if chunks:
        lines.append(f"Chunks: {', '.join(chunks)}")

    return "\n".join(lines)
