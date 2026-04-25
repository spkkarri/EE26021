import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Semantic Answer Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 3px solid #4f8ef7; padding-bottom: 0.4rem;
    }
    .score-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white; border-radius: 12px; padding: 1.5rem;
        text-align: center; box-shadow: 0 4px 20px rgba(79,142,247,0.3);
    }
    .score-number { font-size: 3.5rem; font-weight: 800; color: #4f8ef7; }
    .concept-card {
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        border-left: 5px solid; font-size: 0.9rem;
    }
    .concept-covered  { background: #f0fff4; border-color: #2ecc71; }
    .concept-partial  { background: #fffbf0; border-color: #f39c12; }
    .concept-missing  { background: #fff5f5; border-color: #e74c3c; }
    .contra-flag      { background: #fff0f0; border: 1px solid #e74c3c;
                        border-radius: 6px; padding: 0.5rem; margin: 0.3rem 0; }
    .contra-ok        { background: #f0fff4; border: 1px solid #2ecc71;
                        border-radius: 6px; padding: 0.5rem; margin: 0.3rem 0; }
    .metric-row       { display: flex; gap: 1rem; }
    .metric-box       { background: #f8f9ff; border-radius: 8px; padding: 1rem;
                        text-align: center; flex: 1; }
</style>
""", unsafe_allow_html=True)


# ─── Lazy imports (avoid loading models on startup) ───────────────────────────
@st.cache_resource(show_spinner="Loading SBERT model (first run only)…")
def load_encoder():
    from src.embedding.encoder import SemanticEncoder
    return SemanticEncoder()


@st.cache_resource(show_spinner=False)
def load_evaluator(enable_nli: bool):
    from src.scoring.scorer import AnswerEvaluator
    return AnswerEvaluator(enable_nli=enable_nli)


@st.cache_resource(show_spinner=False)
def load_explainer():
    from src.explainability.explainer import Explainer
    return Explainer()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
    st.markdown("### ⚙️ Configuration")

    enable_nli = st.toggle(
        "Enable NLI (contradiction detection)",
        value=False,
        help="Loads a DeBERTa NLI model. Slower but detects factual errors.",
    )
    aggregation = st.selectbox(
        "Multi-reference aggregation",
        ["max", "weighted", "softmax"],
        help="How to combine similarity to multiple reference answers.",
    )
    use_hungarian = st.checkbox(
        "Use Hungarian alignment",
        value=True,
        help="Optimal concept-sentence matching. Disable for speed.",
    )
    max_marks = st.slider("Maximum marks", 1, 20, 10)

    st.markdown("---")
    st.markdown("**Score formula:**")
    w_sem  = st.slider("Weight: Semantic",    0.0, 1.0, 0.30, 0.05)
    w_conc = st.slider("Weight: Concept",     0.0, 1.0, 0.50, 0.05)
    w_cont = st.slider("Weight: Contradiction", 0.0, 1.0, 0.15, 0.05)
    w_len  = st.slider("Weight: Length",      0.0, 1.0, 0.05, 0.05)

    st.caption(f"Sum of weights: {w_sem+w_conc+w_cont+w_len:.2f}  "
               f"({'⚠️ not 1.0' if abs(w_sem+w_conc+w_cont+w_len-1.0)>0.01 else '✅'})")

    st.markdown("---")
    st.markdown("**Model info**")
    st.caption("SBERT: all-mpnet-base-v2")
    st.caption("NLI: cross-encoder/nli-deberta-v3-small")
    st.caption("Alignment: Hungarian / Greedy")


# ─── Main layout ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 Semantic Answer Evaluation System</div>',
            unsafe_allow_html=True)
st.caption("Transformer-based evaluation with concept coverage, NLI, and explainability.")

tab_eval, tab_batch, tab_compare, tab_docs = st.tabs(
    ["📝 Single Evaluation", "📦 Batch Evaluation", "📊 Baseline Comparison", "📖 Documentation"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📋 Input")
        question = st.text_area(
            "Question", height=80,
            value="Explain what photosynthesis is and why it is important.",
            placeholder="Enter the exam question…",
        )

        student_answer = st.text_area(
            "Student Answer", height=160,
            value=(
                "Photosynthesis is the process by which plants use sunlight to make glucose "
                "from carbon dioxide and water. Chlorophyll in the leaves captures light energy. "
                "Oxygen is released as a byproduct. This is important because it produces oxygen "
                "for breathing and forms the base of food chains on Earth."
            ),
            placeholder="Enter the student's answer…",
        )

        st.markdown("**Reference Answer(s)**")
        n_refs = st.number_input("Number of reference answers", 1, 5, 1)
        reference_answers = []
        for i in range(int(n_refs)):
            ref = st.text_area(
                f"Reference {i+1}", height=120,
                key=f"ref_{i}",
                value=(
                    "Photosynthesis is a biological process by which plants convert light energy "
                    "into chemical energy stored in glucose. The process uses carbon dioxide and water, "
                    "releasing oxygen as a byproduct. It occurs in chloroplasts using chlorophyll. "
                    "Photosynthesis is vital because it produces oxygen and forms the base of most food chains."
                ) if i == 0 else "",
                placeholder=f"Reference answer {i+1}…",
            )
            if ref.strip():
                reference_answers.append(ref)

        evaluate_btn = st.button("🚀 Evaluate Answer", type="primary", use_container_width=True)

    with col_right:
        st.subheader("📊 Results")

        if evaluate_btn:
            if not student_answer.strip():
                st.error("Please enter a student answer.")
            elif not reference_answers:
                st.error("Please provide at least one reference answer.")
            else:
                with st.spinner("Running semantic evaluation pipeline…"):
                    # Update config weights from sidebar
                    from src.config import DEFAULT_CONFIG
                    DEFAULT_CONFIG.scoring.w_semantic = w_sem
                    DEFAULT_CONFIG.scoring.w_concept  = w_conc
                    DEFAULT_CONFIG.scoring.w_contradiction = w_cont
                    DEFAULT_CONFIG.scoring.w_length   = w_len
                    DEFAULT_CONFIG.scoring.max_marks  = float(max_marks)

                    evaluator = load_evaluator(enable_nli)
                    evaluator.config = DEFAULT_CONFIG.scoring

                    # Patch aggregation
                    evaluator._matcher.aggregation = aggregation

                    result = evaluator.evaluate(
                        question=question,
                        student_answer=student_answer,
                        reference_answers=reference_answers,
                        use_hungarian=use_hungarian,
                    )

                    explainer = load_explainer()
                    report    = explainer.explain(result)

                # ── Score card ──────────────────────────────────────────────
                pct = report.percentage
                colour = "#2ecc71" if pct >= 75 else "#f39c12" if pct >= 50 else "#e74c3c"

                st.markdown(f"""
                <div class="score-card">
                  <div style="font-size:1.1rem;opacity:0.8">Final Score</div>
                  <div class="score-number" style="color:{colour}">
                    {report.final_marks:.1f}<span style="font-size:1.5rem;opacity:0.6">/{int(max_marks)}</span>
                  </div>
                  <div style="font-size:1.2rem;margin-top:0.3rem">{report.summary}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # ── Metric columns ──────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Semantic sim.", f"{result.semantic_score:.3f}")
                m2.metric("Concept score", f"{result.concept_score:.3f}")
                m3.metric("Contradiction", f"{result.contradiction_penalty:.3f}",
                          delta=f"-{result.contradiction_penalty:.3f}" if result.contradiction_penalty > 0 else None,
                          delta_color="inverse")
                m4.metric("Coverage", f"{pct:.0f}%")

                # ── Score breakdown chart ────────────────────────────────────
                st.markdown("##### Score Breakdown")
                try:
                    import plotly.graph_objects as go
                    components = report.score_components
                    labels = [c[0] for c in components]
                    values = [c[1] for c in components]
                    colours_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

                    fig = go.Figure(go.Bar(
                        x=labels, y=values,
                        marker_color=colours_bar,
                        text=[f"{v:+.3f}" for v in values],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=260, margin=dict(t=20, b=40),
                        yaxis=dict(range=[-0.2, 0.7]),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.write(report.score_components)

                # ── Concept coverage ─────────────────────────────────────────
                st.markdown(
                    f"##### Concept Coverage  "
                    f"({report.covered_count}/{report.covered_count+report.missing_count} concepts)"
                )
                for exp in report.concept_explanations:
                    css_class = (
                        "concept-covered" if "✅" in exp.status_label
                        else "concept-partial" if "⚠️" in exp.status_label
                        else "concept-missing"
                    )
                    bar_w = int(exp.similarity * 100)
                    st.markdown(f"""
                    <div class="concept-card {css_class}">
                      <b>{exp.status_label}</b> — {exp.concept[:80]}…<br>
                      <small>Similarity: {exp.similarity:.3f} | Credit: {exp.credit:.2f} | Weight: {exp.weight:.2f}</small><br>
                      <div style="background:#e0e0e0;border-radius:4px;height:6px;margin-top:4px">
                        <div style="background:{'#2ecc71' if exp.covered else '#e74c3c'};
                                    width:{bar_w}%;height:6px;border-radius:4px"></div>
                      </div>
                      <small style="color:#666">Best match: <i>{exp.best_match_sentence[:60]}…</i></small>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Heatmap ──────────────────────────────────────────────────
                if report.heatmap_matrix.size > 1:
                    st.markdown("##### Similarity Heatmap (Concepts × Student Sentences)")
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig2, ax = plt.subplots(figsize=(8, max(3, len(report.heatmap_row_labels) * 0.6)))
                        sns.heatmap(
                            report.heatmap_matrix,
                            xticklabels=report.heatmap_col_labels,
                            yticklabels=report.heatmap_row_labels,
                            annot=True, fmt=".2f",
                            cmap="RdYlGn", vmin=0, vmax=1,
                            ax=ax, cbar_kws={"shrink": 0.6},
                        )
                        ax.set_xlabel("Student sentences")
                        ax.set_ylabel("Reference concepts")
                        plt.tight_layout()
                        st.pyplot(fig2)
                    except Exception as e:
                        st.caption(f"Heatmap unavailable: {e}")

                # ── Contradictions ───────────────────────────────────────────
                if enable_nli and report.contradiction_explanations:
                    st.markdown("##### Logical Consistency (NLI Analysis)")
                    for exp in report.contradiction_explanations:
                        css = "contra-flag" if exp.is_flagged else "contra-ok"
                        st.markdown(f"""
                        <div class="{css}">
                          <b>{exp.label}</b>: {exp.sentence[:100]}…<br>
                          <small>P(contradiction)={exp.contradiction_score:.2f}  
                          P(entailment)={exp.entailment_score:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                elif enable_nli:
                    st.info("No logical consistency issues detected.")
                else:
                    st.caption("💡 Enable NLI in the sidebar for contradiction detection.")

                # ── Feedback ─────────────────────────────────────────────────
                st.markdown("##### 💬 Student Feedback")
                st.info(report.feedback)

                st.markdown("##### 📝 Teacher Notes")
                st.code(report.teacher_notes)

                # ── Download ─────────────────────────────────────────────────
                export = {
                    "question": question,
                    "student_answer": student_answer,
                    "final_marks": report.final_marks,
                    "max_marks": float(max_marks),
                    "percentage": report.percentage,
                    "score_breakdown": result.score_breakdown,
                    "covered_concepts": report.covered_count,
                    "missing_concepts": report.missing_count,
                    "has_contradiction": report.has_contradiction,
                    "feedback": report.feedback,
                }
                st.download_button(
                    "⬇️ Download Report (JSON)",
                    data=json.dumps(export, indent=2),
                    file_name="evaluation_report.json",
                    mime="application/json",
                )

        else:
            st.info("Fill in the question, student answer, and reference answer(s), then click **Evaluate Answer**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📦 Batch Evaluation")
    st.markdown(
        "Upload a JSON file with the format: "
        "`[{question, student_answer, references: [...], human_score}, ...]`"
    )

    uploaded = st.file_uploader("Upload dataset JSON", type=["json"])

    if uploaded:
        try:
            raw = json.load(uploaded)
            # Flatten nested structure if needed
            flat = []
            for item in raw:
                if "student_answers" in item:
                    for sa_obj in item["student_answers"]:
                        flat.append({
                            "question": item["question"],
                            "student_answer": sa_obj["text"],
                            "references": item["references"],
                            "human_score": sa_obj["human_score"],
                        })
                else:
                    flat.append(item)

            st.success(f"Loaded {len(flat)} examples.")

            if st.button("▶️ Run Batch Evaluation", type="primary"):
                from src.evaluation.metrics import run_evaluation
                evaluator = load_evaluator(enable_nli=False)   # NLI disabled for speed

                with st.spinner(f"Evaluating {len(flat)} answers…"):
                    sem_metrics, tfidf_metrics, df = run_evaluation(
                        evaluator, flat, max_marks=float(max_marks)
                    )

                st.markdown("#### Results")
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Semantic System**")
                    st.metric("Pearson r",  f"{sem_metrics.pearson_r:.4f}")
                    st.metric("Spearman r", f"{sem_metrics.spearman_r:.4f}")
                    st.metric("MAE",        f"{sem_metrics.mae:.3f}")
                    st.metric("RMSE",       f"{sem_metrics.rmse:.3f}")
                    st.metric("QWK",        f"{sem_metrics.qwk:.4f}")
                with col2:
                    st.markdown("**TF-IDF Baseline**")
                    st.metric("Pearson r",  f"{tfidf_metrics.pearson_r:.4f}")
                    st.metric("Spearman r", f"{tfidf_metrics.spearman_r:.4f}")
                    st.metric("MAE",        f"{tfidf_metrics.mae:.3f}")
                    st.metric("RMSE",       f"{tfidf_metrics.rmse:.3f}")
                    st.metric("QWK",        f"{tfidf_metrics.qwk:.4f}")

                st.markdown("#### Worst Predictions")
                st.dataframe(sem_metrics.error_cases, use_container_width=True)

                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv, "batch_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        # Show the sample dataset
        sample_path = os.path.join(os.path.dirname(__file__), "../data/sample/sample_dataset.json")
        if os.path.exists(sample_path):
            with open(sample_path) as f:
                sample = json.load(f)
            if st.button("Load sample dataset"):
                st.session_state["sample_loaded"] = True
            st.caption("Or upload your own JSON dataset above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Baseline Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("📊 Semantic vs TF-IDF Baseline Comparison")
    st.markdown("""
    This tab demonstrates the **key limitation of TF-IDF** cosine similarity
    and why transformer embeddings are superior.
    """)

    example_pairs = [
        {
            "label": "Paraphrase (should score HIGH)",
            "reference": "The mitochondria generate ATP through oxidative phosphorylation.",
            "student": "Cells produce energy currency (ATP) in the powerhouse organelles via aerobic respiration.",
            "expected": "High semantic, low TF-IDF",
        },
        {
            "label": "Keyword stuffing (should score LOW)",
            "reference": "Gravity causes objects to accelerate towards Earth at 9.8 m/s².",
            "student": "Gravity acceleration Earth objects fall 9.8 m/s² gravitational force mass.",
            "expected": "Low semantic (incoherent), high TF-IDF",
        },
        {
            "label": "Contradiction (should be PENALISED)",
            "reference": "Photosynthesis releases oxygen as a byproduct.",
            "student": "During photosynthesis, plants absorb oxygen from the atmosphere.",
            "expected": "Low NLI score despite surface similarity",
        },
    ]

    for pair in example_pairs:
        with st.expander(f"**{pair['label']}**"):
            st.markdown(f"**Reference:** {pair['reference']}")
            st.markdown(f"**Student:** {pair['student']}")
            st.markdown(f"*Expected result: {pair['expected']}*")

            if st.button(f"Compare →", key=pair["label"]):
                encoder = load_encoder()

                # Semantic
                sem_sim = encoder.pairwise_similarity(pair["reference"], pair["student"])

                # TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vec = TfidfVectorizer(stop_words="english")
                tfidf = vec.fit_transform([pair["reference"], pair["student"]])
                tfidf_sim = float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])

                c1, c2 = st.columns(2)
                c1.metric("🧠 Semantic (SBERT)", f"{sem_sim:.4f}",
                          help="Transformer cosine similarity")
                c2.metric("📄 TF-IDF baseline", f"{tfidf_sim:.4f}",
                          help="Traditional keyword cosine similarity")

                if sem_sim > tfidf_sim + 0.1:
                    st.success("✅ Semantic correctly gives higher score (paraphrase detected).")
                elif tfidf_sim > sem_sim + 0.1:
                    st.warning("⚠️ TF-IDF gives higher score — likely keyword stuffing case.")
                else:
                    st.info("Scores are similar for this pair.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Documentation
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("""
## 📖 System Documentation

### Architecture
The system is a 5-stage pipeline:
1. **Semantic Embedding Layer** — SBERT (all-mpnet-base-v2) encodes all text into 768-dim vectors
2. **Multi-Reference Matching** — Aggregates similarity across multiple model answers (MAX / Weighted / SoftMax)
3. **Concept-Level Scoring** — Decomposes reference into atomic concepts; scores each independently
4. **NLI Contradiction Detection** — DeBERTa cross-encoder flags factually wrong statements
5. **Final Scoring** — Weighted combination with partial marks, length penalty, and normalisation

### Scoring Formula
```
S_raw = w_sem   × S_semantic
      + w_concept × S_concept
      - w_contra  × P_contradiction
      - w_length  × P_length

Final Mark = clamp(S_raw, 0, 1) × MAX_MARKS
```

**Default weights:** sem=0.30, concept=0.50, contra=0.15, length=0.05

### Partial Credit
```
credit(concept_i) = 0.0                                    if sim < 0.50
                  = (sim - 0.50) / (0.80 - 0.50)          if 0.50 ≤ sim < 0.80
                  = 1.0                                    if sim ≥ 0.80
```

### Concept Alignment
- **Greedy**: Each concept matched to best student sentence independently (O(n·m))
- **Hungarian**: Optimal bipartite assignment — minimises total alignment cost (O(n³))

### Length Penalty
```
ratio = len(student_words) / len(reference_words)
P_length = 1 - ratio/min_ratio   if ratio < min_ratio   (too short)
         = (ratio - max_ratio)/max_ratio   if ratio > max_ratio   (too long)
         = 0   otherwise
```
Default: min=0.20, max=3.00

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Pearson r | Linear correlation with human scores |
| Spearman r | Rank-order correlation (robust to outliers) |
| MAE | Mean absolute error in marks |
| RMSE | Root mean squared error (penalises large errors) |
| QWK | Quadratic Weighted Kappa — standard automated grading metric |

### Limitations & Future Work
- NLI is paragraph-level; sentence-level contradictions may be missed in complex answers
- Concept extraction quality depends on spaCy model quality
- Fine-tuning on domain-specific STS pairs would improve concept-level accuracy
- Multi-language support requires multilingual SBERT models
    """)
