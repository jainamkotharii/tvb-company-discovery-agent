import os
import pandas as pd
import streamlit as st
from agent import run_agent

st.set_page_config(
    page_title="TVB Company Discovery Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 TVB Company Discovery Agent")
st.caption("Autonomous company discovery, qualification and executive-contact verification")

with st.sidebar:
    st.header("Target profile")
    st.markdown("""
    **Hard filters from the TVB task**
    - Revenue OR funding: **$1M–$5M**
    - Technology-related platform
    - Minimal to no US presence
    - CEO or Co-founder name + verified email
    """)
    target_count = st.slider("Qualified leads to return", 15, 30, 15)
    max_candidates = st.slider("Candidates to research", 20, 60, 35)
    st.divider()
    st.caption("The agent leaves unverified fields blank. It does not substitute generic contact emails for CEO/co-founder emails.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not configured. Add it in Streamlit Secrets before running.")
if not os.getenv("HUNTER_API_KEY"):
    st.warning("HUNTER_API_KEY is not configured. Email verification will be unavailable and the app will not claim an email is verified.")

if st.button("🚀 Run autonomous discovery", type="primary", use_container_width=True):
    with st.status("Running the TVB research agent...", expanded=True) as status:
        try:
            result = run_agent(target_count=target_count, max_candidates=max_candidates, progress=st.write)
            status.update(label="Research complete", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Run failed", state="error", expanded=True)
            st.exception(e)
            st.stop()

    qualified = result["qualified"]
    rejected = result["rejected"]

    st.success(f"Found {len(qualified)} qualified leads. {len(rejected)} candidates were rejected or could not be fully verified.")

    if qualified:
        df = pd.DataFrame(qualified)
        display_cols = [
            "company_name", "description", "industry", "hq_location",
            "funding_or_revenue", "us_presence", "executive_name",
            "executive_title", "verified_email", "qualification_reason",
            "company_source", "contact_source", "email_verification"
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download qualified leads CSV",
            data=csv,
            file_name="tvb_qualified_leads.csv",
            mime="text/csv",
        )

        with st.expander("Evidence and validation details"):
            for row in qualified:
                st.markdown(f"### {row.get('company_name','')}")
                st.write({
                    "qualification": row.get("qualification_reason", ""),
                    "company_source": row.get("company_source", ""),
                    "contact_source": row.get("contact_source", ""),
                    "email_verification": row.get("email_verification", ""),
                })
                st.divider()

    with st.expander("Rejected / incomplete candidates"):
        if rejected:
            rdf = pd.DataFrame(rejected)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        else:
            st.write("No rejected candidates recorded.")

st.divider()
st.caption("Built specifically for the TVB internship task. Research results are evidence-based; uncertain fields are left blank.")
