from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from config import PREPROCESSED_OUTPUT_PATH, TASK2_JSON_PATH, SEMANTIC_STORE_PATH

st.set_page_config(
    page_title="CUAD Pipeline Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading contracts …")
def load_data(path: Path, _mtime: float):
    if not path.exists():
        return [], {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("contracts", []), data.get("metadata", {})


@st.cache_data(show_spinner="Loading Task 2 results …")
def load_task2(path: Path, _mtime: float):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_mtime(path: Path) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def make_summary_df(contracts):
    rows = []
    for c in contracts:
        rows.append({
            "contract_id":        c["contract_id"],
            "category":           c["category"],
            "category_raw":       c.get("category_raw", c["category"]),
            "part":               c["part"],
            "char_count":         c["char_count"],
            "extraction_method":  c["extraction_method"],
            "quality_flagged":    c["quality_flagged"],
            "quality_flag_reason": c["quality_flag_reason"],
        })
    return pd.DataFrame(rows)


def main():
    st.title("📄 CUAD Pipeline Viewer")
    resolved_path = Path(PREPROCESSED_OUTPUT_PATH).resolve()
    st.caption(f"Data source: `{resolved_path}`")

    mtime = get_mtime(PREPROCESSED_OUTPUT_PATH)
    contracts, metadata = load_data(PREPROCESSED_OUTPUT_PATH, mtime)

    if not contracts:
        st.error("No Task 1 data found. Run `python pipeline.py` first.")
        st.stop()

    df = make_summary_df(contracts)
    contract_map = {c["contract_id"]: c for c in contracts}

    t2_mtime = get_mtime(TASK2_JSON_PATH)
    task2_records = load_task2(TASK2_JSON_PATH, t2_mtime)
    task2_map = {r["contract_id"]: r for r in task2_records}

    with st.sidebar:
        st.header("Filters")
        categories = sorted(df["category"].unique())
        selected_cats = st.multiselect("Category", categories, default=categories)
        parts = sorted(df["part"].unique())
        selected_parts = st.multiselect("Part", parts, default=parts)
        flag_filter = st.selectbox("Quality flag", ["All", "Flagged only", "Clean only"])
        st.divider()
        st.caption(f"Task 1 JSON: {pd.Timestamp(mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
        if task2_records:
            st.caption(f"Task 2 JSON: {pd.Timestamp(t2_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")

    mask = df["category"].isin(selected_cats) & df["part"].isin(selected_parts)
    if flag_filter == "Flagged only":
        mask &= df["quality_flagged"]
    elif flag_filter == "Clean only":
        mask &= ~df["quality_flagged"]
    filtered = df[mask]

    tab_summary, tab_explorer, tab_task2, tab_search = st.tabs([
        "📊 Summary", "🔍 Contract Explorer", "📋 Task 2 Results", "🔎 Semantic Search"
    ])

    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total contracts", len(df))
        col2.metric("Avg char count", f"{df['char_count'].mean():,.0f}")
        col3.metric("Min char count", f"{df['char_count'].min():,}")
        col4.metric("Max char count", f"{df['char_count'].max():,}")

        flagged_n = int(df["quality_flagged"].sum())
        if flagged_n > 0:
            st.warning(f"⚠ {flagged_n} contract(s) flagged for quality issues.")
        else:
            st.success("✅ All contracts passed quality checks.")

        st.subheader("Contracts by Category")
        cat_counts = (
            df.groupby("category")["contract_id"].count()
            .reset_index().rename(columns={"contract_id": "count"})
            .sort_values("count", ascending=False)
        )
        st.bar_chart(cat_counts.set_index("category")["count"])

        st.subheader("Sample Breakdown")
        pivot = df.groupby(["category", "part"])["contract_id"].count().unstack(fill_value=0)
        st.dataframe(pivot, use_container_width=True)

        st.subheader("All Contracts")
        display_cols = ["contract_id", "category", "part", "char_count", "extraction_method", "quality_flagged"]
        styled = filtered[display_cols].copy()

        def highlight_flagged(row):
            return ["background-color: #ffcccc"] * len(row) if row["quality_flagged"] else [""] * len(row)

        st.dataframe(styled.style.apply(highlight_flagged, axis=1), use_container_width=True, height=500)

    with tab_explorer:
        contract_ids = filtered["contract_id"].tolist()
        if not contract_ids:
            st.info("No contracts match the current filters.")
            st.stop()

        show_raw_cat = st.toggle("Show original folder name (category_raw)", value=False)

        selected_id = st.selectbox(
            "Select a contract", contract_ids,
            format_func=lambda cid: (
                f"{cid[:70]}  [{contract_map[cid]['category']}  |  "
                f"{contract_map[cid]['part']}  |  "
                f"{contract_map[cid]['char_count']:,} chars"
                + ("  ⚠" if contract_map[cid]["quality_flagged"] else "") + "]"
            ),
        )

        c = contract_map[selected_id]
        n_cols = 6 if show_raw_cat else 5
        meta_cols = st.columns(n_cols)
        meta_cols[0].metric("Category", c["category"])
        if show_raw_cat:
            raw_cat = c.get("category_raw", c["category"])
            meta_cols[1].metric("Folder (raw)", raw_cat,
                delta="normalised ↑" if raw_cat != c["category"] else "unchanged",
                delta_color="normal" if raw_cat != c["category"] else "off")
            offset = 2
        else:
            offset = 1
        meta_cols[offset].metric("Part", c["part"])
        meta_cols[offset+1].metric("Cleaned chars", f"{c['char_count']:,}")
        meta_cols[offset+2].metric("Raw chars", f"{c['raw_char_count']:,}")
        meta_cols[offset+3].metric("Extraction", c["extraction_method"])

        if c["quality_flagged"]:
            st.error(f"⚠ Quality flag: {c['quality_flag_reason']}")

        if c["txt_reference_found"]:
            ratio = c["extraction_ratio"]
            st.info(f"TXT reference found — extraction ratio: **{ratio:.1%}**  "
                    f"(PDF {c['char_count']:,} chars vs TXT {c['txt_char_count']:,} chars)")

        if selected_id in task2_map:
            t2 = task2_map[selected_id]
            with st.expander("📋 Task 2 extraction results", expanded=True):
                st.markdown(f"**Summary:** {t2.get('summary', 'N/A')}")
                st.markdown("---")
                for key, label in [
                    ("termination_clause", "Termination"),
                    ("confidentiality_clause", "Confidentiality"),
                    ("liability_clause", "Liability"),
                ]:
                    val = t2.get(key, "Not found")
                    icon = "✅" if val != "Not found" else "❌"
                    st.markdown(f"**{icon} {label}**")
                    if val != "Not found":
                        st.text_area(label=key, value=val, height=120, label_visibility="collapsed")
                    else:
                        st.caption("Not found in this contract")

        st.subheader("Text Comparison")
        left, right = st.columns(2)
        with left:
            st.markdown("**Raw extracted text**")
            st.text_area(label="raw", value=c["raw_text"] or "(empty)",
                         height=600, label_visibility="collapsed")
        with right:
            st.markdown("**Cleaned text**")
            st.text_area(label="cleaned", value=c["cleaned_text"] or "(empty)",
                         height=600, label_visibility="collapsed")

    with tab_task2:
        if not task2_records:
            st.warning("Task 2 results not found. Run `python pipeline_task2.py` first.")
        else:
            t2_df = pd.DataFrame(task2_records)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Contracts", len(t2_df))
            col2.metric("Termination found", int((t2_df["termination_clause"] != "Not found").sum()))
            col3.metric("Confidentiality found", int((t2_df["confidentiality_clause"] != "Not found").sum()))
            col4.metric("Liability found", int((t2_df["liability_clause"] != "Not found").sum()))

            st.subheader("Coverage by Clause Type")
            coverage = {
                "Termination": int((t2_df["termination_clause"] != "Not found").sum()),
                "Confidentiality": int((t2_df["confidentiality_clause"] != "Not found").sum()),
                "Liability": int((t2_df["liability_clause"] != "Not found").sum()),
            }
            cov_df = pd.DataFrame.from_dict(coverage, orient="index", columns=["Found"])
            cov_df["Not Found"] = 50 - cov_df["Found"]
            st.bar_chart(cov_df)

            st.subheader("All Results")
            display = t2_df[["contract_id", "termination_clause", "confidentiality_clause", "liability_clause"]].copy()
            for col in ["termination_clause", "confidentiality_clause", "liability_clause"]:
                display[col] = display[col].apply(
                    lambda x: x[:80] + "..." if x != "Not found" and len(x) > 80 else x
                )

            def color_not_found(val):
                return "color: #888" if val == "Not found" else ""

            st.dataframe(
                display.style.map(color_not_found,
                    subset=["termination_clause", "confidentiality_clause", "liability_clause"]),
                use_container_width=True,
                height=600,
            )

            st.subheader("Download")
            csv = t2_df.to_csv(index=False)
            st.download_button("⬇ Download task2_results.csv", csv,
                               "task2_results.csv", "text/csv")

    with tab_search:
        st.subheader("🔎 Semantic Clause Search")
        st.caption("Search extracted clauses using natural language. Powered by nomic-embed-text.")

        task2_ready = TASK2_JSON_PATH.exists()
        store_ready = SEMANTIC_STORE_PATH.exists()

        if not task2_ready:
            st.warning("Run `python pipeline_task2.py` then `python semantic_search.py build`.")
        elif not store_ready:
            st.warning("Run: `python semantic_search.py build`")
        else:
            query = st.text_input("Search query",
                placeholder='e.g. "termination without cause" or "liability cap on damages"')
            top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

            if query:
                with st.spinner("Searching …"):
                    try:
                        from semantic_search import search
                        results = search(query, top_k=top_k, store_path=SEMANTIC_STORE_PATH)
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        results = []

                if results:
                    st.markdown(f"**Top {len(results)} results for:** _{query}_")
                    for r in results:
                        with st.expander(
                            f"#{r['rank']}  |  Score: {r['score']:.4f}  |  "
                            f"{r['contract_id'][:55]}  [{r['clause_type']}]"
                        ):
                            st.markdown(f"**Contract:** `{r['contract_id']}`")
                            st.markdown(f"**Clause type:** `{r['clause_type']}`")
                            st.markdown(f"**Score:** `{r['score']:.4f}`")
                            st.text_area(label="clause", value=r["text"],
                                         height=200, label_visibility="collapsed")
                            rec = task2_map.get(r["contract_id"])
                            if rec and rec.get("summary"):
                                st.info(rec["summary"])
                else:
                    st.info("No results found.")

            if store_ready:
                try:
                    import numpy as np
                    store_data = np.load(SEMANTIC_STORE_PATH, allow_pickle=True)
                    st.caption(f"Store: {len(store_data['vectors'])} embedded clauses.")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
