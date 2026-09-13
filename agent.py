import json
import os
import re
import time
from typing import Callable, Dict, List, Any

import requests
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

QUALIFICATION_SCHEMA = {
    "type": "json_schema",
    "name": "tvb_company_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "domain": {"type": "string"},
                        "description": {"type": "string"},
                        "industry": {"type": "string"},
                        "hq_location": {"type": "string"},
                        "funding_or_revenue": {"type": "string"},
                        "funding_or_revenue_usd": {"type": "number"},
                        "funding_or_revenue_type": {"type": "string"},
                        "us_presence": {"type": "string"},
                        "executive_name": {"type": "string"},
                        "executive_title": {"type": "string"},
                        "executive_email": {"type": "string"},
                        "company_source": {"type": "string"},
                        "contact_source": {"type": "string"},
                        "qualification_reason": {"type": "string"},
                        "meets_all_hard_filters": {"type": "boolean"}
                    },
                    "required": [
                        "company_name", "domain", "description", "industry",
                        "hq_location", "funding_or_revenue", "funding_or_revenue_usd",
                        "funding_or_revenue_type", "us_presence", "executive_name",
                        "executive_title", "executive_email", "company_source",
                        "contact_source", "qualification_reason", "meets_all_hard_filters"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["companies"],
        "additionalProperties": False
    }
}

def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=key)

def _json_response(instructions: str, prompt: str):
    client = _client()
    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        instructions=instructions,
        input=prompt,
        text={"format": QUALIFICATION_SCHEMA},
    )
    return json.loads(response.output_text)

def discover_candidates(max_candidates: int, progress: Callable[[str], None]):
    progress("1/4 Discovering companies from multiple independent search angles...")
    queries = [
        "technology platform startup raised between $1 million and $5 million outside the United States",
        "SaaS startup funding $1m $5m Europe Asia Africa Latin America platform",
        "AI platform startup raised $1m $5m outside US",
        "fintech platform startup raised $1m $5m outside US",
        "healthtech platform startup raised $1m $5m outside US",
        "cybersecurity platform startup raised $1m $5m outside US",
        "edtech platform startup raised $1m $5m outside US",
        "travel technology platform startup raised $1m $5m outside US",
    ]
    query_text = "\n".join(f"- {q}" for q in queries)
    instructions = """
You are the discovery researcher for The Venture Build (TVB).
The task is to discover NEW company candidates, not to use a fixed list.
Search the public web using multiple independent searches.

Hard target:
1) revenue OR funding raised is between USD 1M and USD 5M inclusive;
2) the company operates a technology-related platform;
3) minimal to no presence in the United States;
4) CEO or co-founder name and a direct professional email must be discoverable.

For discovery, be conservative. Do not invent facts or emails. Return blank strings where a fact is not supported.
Prefer primary company sources and reputable funding/news databases for company facts.
Do not return companies you cannot reasonably investigate further.
Return at least 2x the requested final lead count when possible.
"""
    prompt = f"""
Find up to {max_candidates} candidate companies for TVB.

Use these search angles, and vary your searches beyond them when useful:
{query_text}

For each candidate, capture evidence URLs. Do not claim that a generic info@ or contact@ address belongs to the CEO/co-founder.
"""
    data = _json_response(instructions, prompt)
    return data["companies"]

def hunter_domain_search(domain: str):
    key = os.getenv("HUNTER_API_KEY")
    if not key or not domain:
        return []
    try:
        r = requests.get(
            "https://api.hunter.io/v2/domain-search",
            params={"domain": domain, "api_key": key, "limit": 10},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("data", {}).get("emails", []) or []
    except Exception:
        return []

def hunter_verify(email: str):
    key = os.getenv("HUNTER_API_KEY")
    if not key or not email:
        return {"status": "not_checked", "score": None}
    try:
        r = requests.get(
            "https://api.hunter.io/v2/email-verifier",
            params={"email": email, "api_key": key},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        return {
            "status": data.get("status", "unknown"),
            "score": data.get("score"),
            "result": data.get("result"),
            "sources": data.get("sources", []),
        }
    except Exception as e:
        return {"status": "error", "score": None, "error": str(e)}

def choose_executive_email(candidate: Dict[str, Any], progress: Callable[[str], None]):
    """
    Use Hunter only as a verification/enrichment layer.
    The model must first identify the CEO/co-founder. Then Hunter can confirm
    a matching personal executive address.
    """
    domain = candidate.get("domain", "").strip()
    exec_name = candidate.get("executive_name", "").strip()
    supplied = candidate.get("executive_email", "").strip()

    emails = hunter_domain_search(domain)
    target_tokens = [x.lower() for x in re.findall(r"[A-Za-z]+", exec_name)]

    # Prefer a Hunter personal email whose first/last name aligns with the executive.
    ranked = []
    for item in emails:
        value = (item.get("value") or "").strip()
        if not value or item.get("type") != "personal":
            continue
        name = (item.get("first_name", "") + " " + item.get("last_name", "")).lower().strip()
        confidence = item.get("confidence", 0) or 0
        overlap = sum(1 for token in target_tokens if token in name)
        decision = 1 if item.get("position", "").lower() in ("ceo", "chief executive officer", "co-founder", "cofounder", "founder") else 0
        ranked.append((decision, overlap, confidence, value, item))

    ranked.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    if ranked and (ranked[0][0] or ranked[0][1] >= 1):
        value = ranked[0][3]
        verification = hunter_verify(value)
        if verification.get("status") == "valid":
            return value, verification

    if supplied:
        verification = hunter_verify(supplied)
        if verification.get("status") == "valid":
            return supplied, verification

    return "", {"status": "not_verified", "score": None}

def validate_candidates(candidates: List[Dict[str, Any]], target_count: int, progress: Callable[[str], None]):
    progress("2/4 Validating funding/revenue, platform fit and US presence with fresh web research...")
    valid = []
    rejected = []

    # Validate in batches so the agent reasons across the full candidate set efficiently.
    batch_size = 8
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start+batch_size]
        names = "\n".join(
            f"{i+1}. {c.get('company_name','')} | {c.get('domain','')}"
            for i, c in enumerate(batch)
        )
        instructions = """
You are a strict due-diligence validator for TVB.
Re-research every supplied company on the public web. Do NOT trust the discovery record blindly.

A company qualifies only if ALL hard filters are supported:
- USD 1M to 5M inclusive in funding raised OR revenue;
- operates a technology-related platform;
- minimal to no US presence;
- CEO or co-founder can be identified;
- a direct professional email can be verified later.

Important:
- If funding/revenue is outside the range or cannot be supported, reject.
- If US presence is substantial, reject.
- If a field is uncertain, leave it blank and reject if that field is required.
- Do not use generic addresses such as info@, hello@, contact@, sales@.
- Do not invent or infer an email.
- Return source URLs for company facts and executive/contact facts.
"""
        prompt = f"""
Validate these candidates independently:

{names}

For each, return the strongest evidence you can find. Keep the funding/revenue value numerical in USD.
"""
        data = _json_response(instructions, prompt)
        for c in data["companies"]:
            if c.get("meets_all_hard_filters"):
                valid.append(c)
            else:
                rejected.append({
                    "company_name": c.get("company_name", ""),
                    "reason": c.get("qualification_reason", "Failed one or more hard filters")
                })
        if len(valid) >= target_count:
            break

    return valid, rejected

def run_agent(target_count: int = 15, max_candidates: int = 35, progress: Callable[[str], None] = print):
    candidates = discover_candidates(max_candidates, progress)

    # Deduplicate by domain/name.
    seen = set()
    deduped = []
    for c in candidates:
        key = (c.get("domain", "").lower().strip() or c.get("company_name", "").lower().strip())
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)

    valid, rejected = validate_candidates(deduped, target_count, progress)

    progress("3/4 Verifying CEO/co-founder emails with a dedicated email-verification service...")
    qualified = []

    for idx, candidate in enumerate(valid, start=1):
        if len(qualified) >= target_count:
            break
        progress(f"Checking contact {idx}/{len(valid)}: {candidate.get('company_name','')}")
        email, verification = choose_executive_email(candidate, progress)

        # Only an explicit valid verifier result qualifies as "verified".
        if verification.get("status") != "valid":
            rejected.append({
                "company_name": candidate.get("company_name", ""),
                "reason": "Executive email could not be independently verified."
            })
            continue

        candidate["verified_email"] = email
        candidate["email_verification"] = verification
        candidate.pop("executive_email", None)
        qualified.append(candidate)

    progress("4/4 Final quality gate complete.")
    return {"qualified": qualified, "rejected": rejected}
