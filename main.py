"""
Member QA Service - Natural Language Question Answering System

A FastAPI-based question-answering system that answers natural-language questions
about member data from a conversational dataset.

Architecture:
1. Entity Extraction: Identifies person names, locations, and numbers from queries
2. BM25 Ranking: Fast full-text search to retrieve relevant messages
3. Pre-filtering: Filters by extracted entities (person, location)
4. Question-Type Handlers: Specific logic for WHO, WHEN, WHERE, HOW_MANY, etc.
5. Pattern Matching: Extracts specific information based on question type

No pre-trained models used.
"""

from fastapi import FastAPI, Query
import re
import requests
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_URL = "https://november7-730026606190.europe-west1.run.app/messages/"
NOT_FOUND_ANSWER = "This topic or answer does not exist in the conversation."

app = FastAPI(
    title="Member QA Service",
    version="3.0.0",
    description="Natural-language QA system for member conversations"
)

# In-memory cache for messages and BM25 index
_CACHE: Dict[str, Any] = {
    "messages": [],
    "doc_tokens": [],
    "bm25": None,
}

# ============================================================================
# TEXT UTILITIES
# ============================================================================

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"\w+", text.lower())


# ============================================================================
# DATA FETCHING & INDEXING
# ============================================================================

def fetch_messages() -> List[dict]:
    """
    Fetch all messages from local file or API.
    
    Priority:
    1. Load from local all_messages.json if exists
    2. Fallback to fetching from API with pagination
    
    Returns:
        List of messages with keys: user_name, message, timestamp
    """
    # Try local file first
    try:
        if os.path.exists("all_messages.json"):
            with open("all_messages.json", "r") as f:
                data = json.load(f)
                items = data.get("items", [])
                if items:
                    print(f"Loaded {len(items)} messages from local file.")
                    
                    # Parse and sort by timestamp
                    for it in items:
                        try:
                            it["_parsed_timestamp"] = datetime.fromisoformat(
                                it["timestamp"].replace("+00:00", "")
                            )
                        except (KeyError, ValueError):
                            it["_parsed_timestamp"] = datetime.now()
                    
                    items.sort(key=lambda x: x["_parsed_timestamp"])
                    
                    messages = [
                        {
                            "user_name": it["user_name"],
                            "message": it["message"],
                            "timestamp": it["_parsed_timestamp"],
                        }
                        for it in items
                    ]
                    return messages
    except Exception as e:
        print(f"Could not load local file: {e}")
    
    # Fallback to API
    print("Fetching from API...")
    skip = 0
    limit = 1000
    items: List[dict] = []

    while True:
        cur_url = f"{DATA_URL}?skip={skip}&limit={limit}"
        try:
            resp = requests.get(cur_url, timeout=15)
            if resp.status_code >= 400:
                print(f"Stopping at skip={skip}, status={resp.status_code}")
                break
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error at skip={skip}: {e}")
            break
        except ValueError as e:
            print(f"JSON decode error at skip={skip}: {e}")
            break

        batch = data.get("items", [])
        if not batch:
            break

        items.extend(batch)
        skip += limit

    # Parse timestamps
    for it in items:
        try:
            it["_parsed_timestamp"] = datetime.fromisoformat(it["timestamp"])
        except (KeyError, ValueError):
            it["_parsed_timestamp"] = datetime.now()
    
    items.sort(key=lambda x: x["_parsed_timestamp"])

    messages = [
        {
            "user_name": it["user_name"],
            "message": it["message"],
            "timestamp": it["_parsed_timestamp"],
        }
        for it in items
    ]

    print(f"Loaded {len(messages)} messages from API.")
    return messages


def build_index(messages: List[dict]):
    """
    Build BM25 index from messages.
    
    BM25 is a standard ranking function used in information retrieval.
    We tokenize user_name + message together for better matching.
    
    Args:
        messages: List of message dicts
        
    Returns:
        Tuple of (doc_tokens, bm25_index)
    """
    docs_tokens: List[List[str]] = []
    for m in messages:
        # Combine user name and message for indexing
        combined = f"{m['user_name']} {m['message']}"
        docs_tokens.append(tokenize(combined))

    bm25 = BM25Okapi(docs_tokens)
    return docs_tokens, bm25


def ensure_index():
    """Ensure BM25 index is built and cached."""
    if _CACHE["bm25"] is not None:
        return

    print("Fetching & indexing messages...")
    messages = fetch_messages()
    doc_tokens, bm25 = build_index(messages)

    _CACHE["messages"] = messages
    _CACHE["doc_tokens"] = doc_tokens
    _CACHE["bm25"] = bm25

    print("Index ready")


@app.on_event("startup")
async def startup_event():
    """Initialize index on app startup."""
    try:
        ensure_index()
    except Exception as e:
        print(f"Failed to load index on startup: {e}")


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

QUESTION_WORDS = {
    "what", "when", "why", "where", "how", "who", "which", "can",
    "please", "looking", "will", "should", "is", "are", "need",
    "thank", "book", "check", "send", "find", "get", "arrange",
    "could", "would", "do", "does"
}


def extract_person_names(q: str) -> List[str]:
    """
    Extract person names from query.
    
    Strategy:
    1. Try to find multi-word names (e.g., "Vikram Desai")
    2. Filter out common question words
    3. Fallback to single capitalized words
    
    Args:
        q: Query string
        
    Returns:
        List of extracted person names
    """
    # Try multi-word names first
    full_names = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", q)
    full_names = [n for n in full_names if n.lower() not in QUESTION_WORDS]
    if full_names:
        return full_names
    
    # Fallback to single capitalized words
    result = re.findall(r"\b[A-Z][a-z]+\b", q)
    return [w for w in result if w.lower() not in QUESTION_WORDS]


def extract_locations(q: str) -> List[str]:
    """
    Extract location names from query.
    
    Uses a list of known locations for exact matching + fallback to
    extracted capitalized words that might be locations.
    
    Args:
        q: Query string
        
    Returns:
        List of extracted location names
    """
    # Known locations in dataset
    location_keywords = [
        "london", "paris", "tokyo", "new york", "dubai", "singapore",
        "bangkok", "aspen", "maldives", "bali", "cannes", "monaco",
        "tuscany", "santorini", "riviera", "milan", "switzerland",
        "kyoto", "pebble beach"
    ]
    
    found = [loc for loc in location_keywords if loc.lower() in q.lower()]
    return found


def extract_numbers(text: str) -> List[str]:
    """Extract digits and word numbers from text."""
    digits = re.findall(r"\b(\d+)\b", text)
    
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
        "thirty": "30", "hundred": "100"
    }
    
    words = re.findall(
        r"\b(" + "|".join(word_to_num.keys()) + r")\b",
        text.lower()
    )
    words_converted = [word_to_num[w] for w in words]
    
    return digits + words_converted


def extract_question_type(q: str) -> str:
    """
    Determine the type of question.
    
    Returns one of: WHO, WHEN, WHERE, HOW_MANY, WHAT_ARE, WHICH, WHY, GENERIC
    """
    qlow = q.lower()
    
    if re.search(r"\bwho\b", qlow):
        return "WHO"
    elif re.search(r"\bwhen\b", qlow):
        return "WHEN"
    elif re.search(r"\bwhere\b", qlow):
        return "WHERE"
    elif re.search(r"\bhow many\b", qlow):
        return "HOW_MANY"
    elif re.search(r"\bwhat are\b", qlow):
        return "WHAT_ARE"
    elif re.search(r"\bwhich\b", qlow):
        return "WHICH"
    elif re.search(r"\bwhy\b", qlow):
        return "WHY"
    else:
        return "GENERIC"


def extract_number_strict(text: str) -> Optional[str]:
    """Extract the first number from text."""
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return m.group(1)
    
    word_to_num = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    
    t = text.lower()
    for w, v in word_to_num.items():
        if re.search(rf"\b{w}\b", t):
            return v
    return None


# ============================================================================
# RETRIEVAL & FILTERING
# ============================================================================

def get_context(idx: int, w: int = 2) -> List[int]:
    """Get message indices in context window around index idx."""
    n = len(_CACHE["messages"])
    return list(range(max(0, idx - w), min(n, idx + w + 1)))


def filter_candidates_by_entities(
    candidates: List[int], q: str, msgs: List[dict]
) -> List[int]:
    """
    Pre-filter candidates by extracted entities (person, location).
    
    This is crucial for accuracy: it ensures BM25 ranking is applied
    to a relevant subset, preventing irrelevant results from being returned.
    
    Strategy:
    1. Filter by person if extracted
    2. Further filter by location if available and we have many candidates
    3. Fallback to original candidates if filtering is too restrictive
    
    Args:
        candidates: List of message indices
        q: Query string
        msgs: List of messages
        
    Returns:
        Filtered list of candidate indices
    """
    persons = extract_person_names(q)
    locations = extract_locations(q)
    
    filtered = candidates[:]
    
    # Filter by person if mentioned
    if persons:
        person_matches = [
            i for i in filtered
            if any(p.lower() in msgs[i]["user_name"].lower() for p in persons)
        ]
        if person_matches:
            filtered = person_matches
    
    # Further filter by location if we have many candidates
    if locations and len(filtered) > 5:
        location_matches = [
            i for i in filtered
            if any(loc.lower() in msgs[i]["message"].lower() for loc in locations)
        ]
        if location_matches:
            filtered = location_matches
    
    # Fallback to original candidates if filtering was too restrictive
    return filtered if filtered else candidates


# ============================================================================
# QUESTION HANDLERS
# ============================================================================

def answer_question(q: str) -> str:
    """
    Answer a natural language question.
    
    Pipeline:
    1. Extract entities (person, location, numbers) from query
    2. Get BM25 ranking of top 20 candidates
    3. Extract message context around top hits
    4. Pre-filter by extracted entities
    5. Determine question type
    6. Apply type-specific handler (WHO, WHEN, WHERE, etc.)
    
    Args:
        q: Natural language question
        
    Returns:
        Answer string
    """
    ensure_index()
    
    msgs = _CACHE["messages"]
    bm25 = _CACHE["bm25"]
    
    if not bm25:
        return NOT_FOUND_ANSWER
    
    # --------- STEP 1: BM25 ranking ---------
    q_tokens = tokenize(q)
    if not q_tokens:
        return NOT_FOUND_ANSWER
    
    scores = bm25.get_scores(q_tokens)
    order = np.argsort(scores)[::-1]
    top_k_indices = [int(i) for i in order[:20]]
    
    # --------- STEP 2: Get candidates with context ---------
    cand_idx = []
    seen = set()
    for idx in top_k_indices:
        for j in get_context(idx, w=2):
            if j not in seen:
                seen.add(j)
                cand_idx.append(j)
    
    if not cand_idx:
        return NOT_FOUND_ANSWER
    
    # --------- STEP 3: Pre-filter by entities ---------
    cand_idx = filter_candidates_by_entities(cand_idx, q, msgs)
    
    # --------- STEP 4: Determine question type and handle ---------
    qtype = extract_question_type(q)
    qlow = q.lower()
    
    # WHO - Return person name
    if qtype == "WHO":
        if top_k_indices:
            return msgs[top_k_indices[0]]["user_name"]
        return NOT_FOUND_ANSWER
    
    # WHEN - Find date/time information
    elif qtype == "WHEN":
        date_patterns = [
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
            r"\b(today|tomorrow|tonight|next\s+\w+|this\s+\w+|last\s+\w+)\b",
            r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        ]
        
        # Priority 1: Look in pre-filtered candidates
        for idx in cand_idx:
            msg = msgs[idx]["message"]
            for pattern in date_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    return msg
        
        # Priority 2: Search all messages of extracted person
        persons = extract_person_names(q)
        locations = extract_locations(q)
        
        if persons:
            for idx, msg in enumerate(msgs):
                if any(p.lower() in msg["user_name"].lower() for p in persons):
                    has_date = any(
                        re.search(p, msg["message"], re.IGNORECASE)
                        for p in date_patterns
                    )
                    if has_date:
                        # Prefer if location also mentioned
                        if locations and any(
                            loc.lower() in msg["message"].lower()
                            for loc in locations
                        ):
                            return msg["message"]
                        elif not locations:
                            return msg["message"]
        
        return NOT_FOUND_ANSWER
    
    # WHERE - Find location information
    elif qtype == "WHERE":
        loc_regex = r"\b(?:to|in|at)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        for idx in cand_idx:
            match = re.search(loc_regex, msgs[idx]["message"])
            if match:
                return match.group(1)
        return NOT_FOUND_ANSWER
    
    # HOW_MANY - Extract and return numbers
    elif qtype == "HOW_MANY":
        noun_match = re.search(r"how many\s+(\w+)", qlow)
        noun = noun_match.group(1) if noun_match else None
        
        for idx in cand_idx:
            msg = msgs[idx]["message"]
            if noun and noun in msg.lower():
                num = extract_number_strict(msg)
                if num:
                    return num
        
        return NOT_FOUND_ANSWER
    
    # WHICH - Find specific items/places
    elif qtype == "WHICH":
        for idx in cand_idx:
            msg = msgs[idx]["message"]
            if " at " in msg.lower():
                after = msg.split(" at ", 1)[1]
                return after
        return NOT_FOUND_ANSWER
    
    # WHAT_ARE - Return lists/descriptions
    elif qtype == "WHAT_ARE":
        for idx in cand_idx:
            msg = msgs[idx]["message"]
            if " are " in msg.lower():
                after = msg.split(" are ", 1)[1]
                return after
        return NOT_FOUND_ANSWER
    
    # WHY - Find cause/reason
    elif qtype == "WHY":
        for idx in cand_idx:
            if "because" in msgs[idx]["message"].lower():
                return msgs[idx]["message"]
        return NOT_FOUND_ANSWER
    
    # GENERIC - Return most relevant message
    else:
        if cand_idx:
            return msgs[cand_idx[0]]["message"]
        elif top_k_indices:
            return msgs[top_k_indices[0]]["message"]
        
        return NOT_FOUND_ANSWER


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/ask")
def ask(q: str = Query(..., description="Natural language question")):
    """
    Answer a question about member conversations.
    
    Example: /ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F
    
    Returns:
        {"answer": "..."}
    """
    try:
        ans = answer_question(q)
        return {"answer": ans}
    except Exception as e:
        return {"error": str(e)}


@app.get("/refresh")
def refresh():
    """Clear cache and rebuild index."""
    _CACHE["messages"] = []
    _CACHE["doc_tokens"] = []
    _CACHE["bm25"] = None
    ensure_index()
    return {"status": "refreshed"}


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Member QA Service",
        "version": "3.0.0",
        "endpoints": {
            "ask": "/ask?q=<question>",
            "refresh": "/refresh",
            "docs": "/docs"
        }
    }
