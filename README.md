# TVB Company Discovery Agent

An autonomous research agent built for The Venture Build (TVB) internship task.

## What the agent does

The application dynamically discovers company candidates from the public web, validates them against TVB's hard target profile, identifies a CEO/co-founder, and independently verifies the executive email.

### Hard filters

1. Revenue OR funding raised: USD 1M–5M
2. Technology-related platform
3. Minimal to no presence in the United States
4. CEO or co-founder name + verified professional email

Unverified fields are left blank or the candidate is rejected. Generic emails are never treated as CEO/co-founder emails.

## Architecture

```text
Streamlit UI
    |
    v
Discovery Agent
    |-- multiple web-search angles
    v
Candidate pool
    |
    v
Due-diligence Validation Agent
    |-- funding/revenue
    |-- technology/platform fit
    |-- US presence
    |-- executive identity
    v
Executive Contact Enrichment
    |
    v
Hunter Email Verification
    |
    v
Final Quality Gate
    |
    v
15+ qualified leads + evidence
```

## Why this is agentic

The system does not use a hardcoded company list. It generates multiple search angles, discovers candidates, re-researches them, rejects unsupported candidates, and performs a separate contact verification step.

## APIs

- OpenAI Responses API with web search for dynamic research and reasoning.
- Hunter Domain Search + Email Verifier for executive email discovery/verification.
- Streamlit for the live application.

OpenAI's current API supports the Responses API and built-in web search tools. Hunter's API provides Domain Search and Email Verifier endpoints.

## Environment variables

```text
OPENAI_API_KEY=...
HUNTER_API_KEY=...
OPENAI_MODEL=gpt-5.6-luna
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this repository to a public GitHub repository.
2. Open Streamlit Community Cloud.
3. Create app.
4. Select the GitHub repository.
5. Entrypoint: `app.py`.
6. Add secrets:

```toml
OPENAI_API_KEY = "your-openai-key"
HUNTER_API_KEY = "your-hunter-key"
OPENAI_MODEL = "gpt-5.6-luna"
```

7. Deploy.
8. Test the public URL from a clean browser.

## Quality-control checklist

Before submission:

- [ ] Public GitHub repository opens without authentication.
- [ ] Live Streamlit URL opens.
- [ ] Run button works.
- [ ] At least 15 qualifying rows can be produced.
- [ ] No generic email is presented as CEO/co-founder email.
- [ ] Funding/revenue evidence is visible.
- [ ] US-presence reasoning is visible.
- [ ] Source URLs are present.
- [ ] CSV export works.
- [ ] No API keys are committed to GitHub.

## Important limitation

No automated research system can guarantee that every public-web fact is perfect. The app therefore uses a conservative policy: if a required fact or executive email cannot be supported and independently verified, the candidate is excluded rather than filled with a generic or guessed value.
