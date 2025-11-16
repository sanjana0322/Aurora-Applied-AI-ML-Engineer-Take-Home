# Aurora-Applied-AI-ML-Engineer-Take-Home

# Member QA Service

A FastAPI-based natural-language question-answering system that answers questions about member conversations.

## Overview

The system accepts natural-language questions about member data and returns relevant answers extracted from conversation messages. It handles various question types including WHO, WHEN, WHERE, HOW_MANY, and more.

### Example Questions

- **"When is Layla planning her trip to London?"**  
  → "We need a suite for five nights at Claridge's in London starting Monday."

- **"How many people are coming to dinner at The French Laundry?"**  
  → "4"

- **"Who wants to visit Japan?"**  
  → "Layla Kawaguchi"

- **"Where is Sophia going?"**  
  → "Aspen"

## Architecture

### Current Implementation: Hybrid Approach

The system uses a **three-stage pipeline**:

1. **Entity Extraction**
   - Extracts person names, locations, and numbers from queries
   - Filters out common question words using a predefined set
   - Uses regex patterns for robust extraction

2. **BM25 Ranking**
   - Fast, standard information retrieval algorithm
   - Retrieves top 20 candidate messages by relevance
   - Combines user name and message text for better matching

3. **Pre-filtering & Question-Type Handlers**
   - Filters candidates by extracted entities (person, location)
   - Determines question type (WHO, WHEN, WHERE, etc.)
   - Applies type-specific handlers with regex pattern matching

```
Query → Entity Extraction → BM25 Ranking → Pre-filtering → Type Handler → Answer
```

### Why This Approach Works

- ✅ **No pre-trained models** → Only core libraries
- ✅ **Generic across all persons** → Works for any member in dataset
- ✅ **Interpretable** → Every decision is rule-based
- ✅ **Fast** → BM25 is O(n) with small constant factors
- ✅ **Correct** → Returns "not found" when data doesn't exist

## API Usage

### Endpoint: `/ask`

```bash
GET /ask?q=<question>
```

**Request:**
```bash
curl "http://54.226.56.68/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
```

**Response:**
```json
{
  "answer": "We need a suite for five nights at Claridge's in London starting Monday."
}
```

### Endpoint: `/refresh`

Clears the in-memory cache and rebuilds the BM25 index.

```bash
GET /refresh
```

## Installation & Setup

### Requirements

```
fastapi==0.104.1
uvicorn==0.24.0
rank-bm25==0.2.2
numpy==1.24.0
requests==2.31.0
```

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python3 -m uvicorn main:app --reload
```

Server will start at `http://54.226.56.68`

Access interactive docs: `http://54.226.56.68/docs`

## Data Sources

### Option 1: Local File (Recommended)

Place `all_messages.json` in the same directory as `main.py`:

```json
{
  "items": [
    {
      "id": "...",
      "user_name": "Layla Kawaguchi",
      "message": "...",
      "timestamp": "2025-05-05T07:47:20.159073+00:00"
    },
    ...
  ]
}
```

### Option 2: Remote API

The system will automatically fetch from the API if local file is not found:

```
https://november7-730026606190.europe-west1.run.app/messages/
```

Pagination: `?skip=0&limit=1000`

## Supported Question Types

| Type | Pattern | Example | Handler |
|------|---------|---------|---------|
| **WHO** | Starts with "who" | "Who wants to visit Japan?" | Returns person name |
| **WHEN** | Starts with "when" | "When is Layla's trip?" | Extracts dates/times (regex patterns) |
| **WHERE** | Starts with "where" | "Where is Sophia going?" | Extracts locations (regex) |
| **HOW_MANY** | "how many + noun" | "How many people?" | Extracts numbers |
| **WHAT_ARE** | "what are + ..." | "What are Amira's favorites?" | Returns text after "are" |
| **WHICH** | Starts with "which" | "Which restaurants?" | Returns text after "at" |
| **WHY** | Starts with "why" | "Why did they...?" | Searches for "because" |
| **GENERIC** | No specific pattern | "Tell me about Layla" | Returns most relevant message |

---

## Alternative Approaches

### 1. **Semantic Search with Embeddings**

Use sentence embeddings to understand query meaning rather than just keywords.

#### Pros:
- Handles paraphrasing ("Layla's trip to London" = "When is Layla going to London?")
- Better semantic understanding
- More flexible matching

#### Cons:
- Requires storing embeddings (3349 × 384 = ~1.3 MB for all-MiniLM)

**When to use:** Production systems with diverse paraphrasing

---

### 2. **Hybrid: BM25 + Semantic Re-ranking**

Combine speed of BM25 with accuracy of semantic search.

#### Approach:
1. Use BM25 to get top 50 candidates (fast first pass)
2. Re-rank top 50 using semantic similarity (slow but on small set)
3. Further filter by extracted entities

#### Pros:
- Best of both worlds: speed + accuracy
- Balances keyword and semantic matching
- Still relatively fast

#### Cons:
- More complex implementation
- Requires tuning weight combination

**When to use:** High-accuracy applications where speed is acceptable

---

### 3. **Extractive QA with BERT/DistilBERT**

Use fine-tuned extractive QA model to find answer spans.

#### Approach:
1. Use BM25 to get top 5 relevant messages
2. For each message, use QA model to extract answer span
3. Return span with highest confidence

#### Pros:
- Can extract specific answer spans (not full messages)
- State-of-the-art accuracy
- Handles complex reasoning

#### Cons:
- Requires pre-trained model (~350MB for RoBERTa)
- May not work well if answer isn't in top BM25 results

**When to use:** Production QA systems with GPU availability

---

### 4. **Query Expansion + BM25**

Expand query with synonyms/related terms before BM25 ranking.

#### Approach:
1. Original query: "How many tickets?"
2. Expanded query: "How many tickets, seats, passes?"
3. BM25 on expanded query

#### Cons:
- Manual synonym list maintenance needed
- Limited to predefined relationships
- May introduce noise

**When to use:** Niche domains with limited vocabulary

---

### 5. **Knowledge Graph + SPARQL**

Convert conversations into structured triples and query with SPARQL.

#### Example:
```
Triples:
- Layla → hasDestination → London
- Layla → hasTravelDate → "next month"
- London → hasAccommodation → Claridge's

Query:
SELECT ?date WHERE {
  ?person name "Layla" .
  ?person hasTravelDate ?date .
  ?person hasDestination "London"
}
```

#### Pros:
- Highly structured approach
- Can handle complex relationships
- Explainable queries

#### Cons:
- Difficult to extract triples from natural language
- Requires entity linking
- Not suitable for conversational data

**When to use:** Highly structured domain knowledge

---

## Comparison Table

| Approach | Speed | Accuracy | Complexity |
|----------|-------|----------|-----------|
| **Current (BM25 + Rules)** | High | Medium | Low |
| Semantic Embeddings | Medium | Good | Medium |
| Hybrid (BM25 + Semantic) | Medium | Good | Medium |
| BERT Extractive QA | Low | High | High |
| Knowledge Graph | Medium | Good | Very High |

---

## Recommendations

**Current approach (BM25 + Entity Extraction + Rules) is optimal because:**
- ✅ No pre-trained models required
- ✅ Fully transparent
- ✅ Works generically for any person
- ✅ Fast and lightweight
- ✅ Correct error handling

### For Production Use:
1. **Short term**: Use current approach + add more test cases
2. **Medium term**: Migrate to Hybrid (BM25 + Semantic embeddings)
3. **Long term**: Consider fine-tuned extractive QA model

### For Better Accuracy Without Models:
- Expand location database
- Add more date/time patterns
- Implement noun phrase extraction
- Add contextual reasoning for multi-hop questions

---

## Files

- `main.py` - Main application code
- `all_messages.json` - Complete conversation dataset (3349 messages)
- `requirements.txt` - Python dependencies
- `IMPLEMENTATION_SUMMARY.md` - Current Implementation details and Approach.
- `README.md` - This file

---

## Deployment

### Local
```bash
python3 -m uvicorn main:app --reload
```

---

## Future Enhancements

1. **Better date extraction**: Support relative dates ("in 2 weeks", "tomorrow")
2. **Coreference resolution**: Handle "her", "his", pronouns
3. **Multi-hop reasoning**: "Where is the person who..."
4. **Confidence scoring**: Return confidence along with answer
5. **Clarification questions**: "Did you mean...?" for ambiguous queries
6. **Feedback loop**: Learn from user corrections
7. **Caching**: Cache repeated questions
8. **Analytics**: Track question types and success rate

## Bonus 2: Data Insights & Observed Anomalies

During manual review of a subset of the 3,349-member message dataset, several practical inconsistencies and operational issues surfaced. These observations come from typical developer spot-checks and represent the kinds of real-world data imperfections that influence the design of the question-answering system.

Although not an exhaustive audit, these findings highlight areas where validation, normalization, and operational workflows could be improved.

 ### 1. Billing Irregularities & Duplicate Charges

Several entries involve members flagging issues such as:

- duplicate hotel or service charges

- unexpected billing amounts

- confusion around refunds or pending payments

These inconsistencies often required support follow-ups, indicating that billing validation and reconciliation processes could benefit from additional safeguards.

### 2. Contact Information Problems

Across multiple messages, users submitted repeated updates to their contact details. A few patterns stood out:

- email addresses missing "@" or domain parts

- incomplete or incorrectly formatted phone numbers

- repeated submissions of corrected addresses

This suggests the need for stronger input validation and editable profile confirmation flows.

### 3. Reservation Date & Timing Mix-Ups

A large portion of user messages revolve around fixing or clarifying reservation details:

- correcting booking dates

- confirming time ranges

- resolving time-zone misunderstandings

- identifying missed or mismatched event times

These patterns indicate that the current scheduling and confirmation workflow may not clearly communicate finalized booking information.

### 4. Name Formatting & Encoding Issues

Some records displayed issues where user names appeared improperly encoded or partially missing characters, for example:

- "Hans Mller" instead of "Hans Müller"

- inconsistent casing (uppercase/lowercase variations)

- extra spaces or missing spacing between first and last names

These inconsistencies suggest that Unicode normalization and stricter user name validation could improve data quality.

### 5. Lost or Unconfirmed Bookings

Message threads show repeated cases where users:

- lost access to tickets

- needed urgent re-confirmation

- had incomplete itineraries

- believed a reservation was made when it wasn’t fully processed

This kind of uncertainty creates extra workload for support agents and highlights opportunities for automated confirmation checks.

### 6. Repeated or Unusual Preference Requests

Some users frequently emphasize specific preferences, such as:

- dietary restrictions

- seat types (aisle, window, premium)

- loyalty point usage

- specific in-room amenities

These preferences appear inconsistently stored or recognized across interactions, resulting in repetitive clarification messages.

### 7. Service or Experience Failures

A handful of messages mention:

- incorrect or missing amenities

- payment failures

- wrong gifts or surprise arrangements

- disappointing service experiences (“artist didn’t meet expectations”, “reservation credentials didn’t work”)

These notes usually triggered follow-up action, suggesting that automated validation or tighter coordination with vendors could reduce failures.

## Summary

Overall, the dataset reflects natural inconsistencies that appear in real member-support environments—ranging from data-entry issues to workflow gaps. Addressing these areas through better frontend validation, improved encoding handling, clearer booking confirmations, and consistent preference tracking could significantly reduce friction for both members and support teams.

These insights also guided how this QA system was designed, especially in terms of handling incomplete inputs, inconsistent text formats, and ambiguous message context.