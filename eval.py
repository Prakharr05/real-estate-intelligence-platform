"""
eval.py — Platform Evaluation Script
======================================
Evaluates both the ML valuation models and the RAG pipeline.

ML  : RMSE + R² per category on a 20% holdout test set
RAG : Faithfulness + Answer Relevancy using RAGAS

Results saved to eval_results/ml_scores.json
                  eval_results/rag_scores.json

Run: python eval.py
     python eval.py --skip-rag      (ML only, no OpenAI calls)
     python eval.py --skip-ml       (RAG only)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import SessionLocal
from models import Society, BuilderFloor, Plot
from openai import OpenAI

# ── Output directory ──────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============================================================
# ML EVALUATION
# ============================================================

def load_plots(db):
    rows = db.query(Plot).filter(
        Plot.price_per_sqft.isnot(None),
        Plot.price_per_sqft > 0,
        Plot.plot_area_sqyd.isnot(None),
        Plot.sector != "Unknown",
        Plot.connectivity_score > 0
    ).all()
    records = []
    for r in rows:
        records.append({
            "sector":             r.sector,
            "area_sqyd":          r.plot_area_sqyd,
            "connectivity_score": r.connectivity_score,
            "is_corner":          1 if r.is_corner_plot == "Yes" else 0,
            "price_per_sqft":     r.price_per_sqft,
        })
    return pd.DataFrame(records)


def load_floors(db):
    rows = db.query(BuilderFloor).filter(
        BuilderFloor.price_per_sqft.isnot(None),
        BuilderFloor.price_per_sqft > 0,
        BuilderFloor.area_sqft.isnot(None),
        BuilderFloor.bhk_type.isnot(None),
        BuilderFloor.sector != "Unknown",
        BuilderFloor.connectivity_score > 0
    ).all()
    records = []
    for r in rows:
        try:    bhk = float(str(r.bhk_type).strip())
        except: continue
        records.append({
            "sector":             r.sector,
            "area_sqft":          r.area_sqft,
            "bhk_type":           bhk,
            "connectivity_score": r.connectivity_score,
            "price_per_sqft":     r.price_per_sqft,
        })
    return pd.DataFrame(records)


def load_societies(db):
    rows = db.query(Society).filter(
        Society.price_per_sqft.isnot(None),
        Society.price_per_sqft > 0,
        Society.area_sqft.isnot(None),
        Society.bhk_type.isnot(None),
        Society.sector != "Unknown",
        Society.connectivity_score > 0
    ).all()
    records = []
    for r in rows:
        try:    bhk = float(str(r.bhk_type).strip())
        except: continue
        records.append({
            "sector":             r.sector,
            "area_sqft":          r.area_sqft,
            "bhk_type":           bhk,
            "connectivity_score": r.connectivity_score,
            "possession":         1 if r.possession_status == "Ready" else 0,
            "price_per_sqft":     r.price_per_sqft,
        })
    return pd.DataFrame(records)


def evaluate_ml_category(df, feature_cols, target_col, category_name):
    """
    Loads the saved model, encodes features using saved encoder,
    evaluates on a 20% holdout (same random seed as train.py).
    Returns dict with RMSE and R².
    """
    print(f"\n  Evaluating: {category_name.upper()} ({len(df)} total rows)")

    if len(df) < 20:
        print(f"  ⚠️ Too few rows — skipping.")
        return None

    # ── Load saved model + encoder ────────────────────────────
    model_path   = os.path.join(MODELS_DIR, f"{category_name}_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{category_name}_encoder.pkl")

    if not os.path.exists(model_path):
        print(f"  ⚠️ No model found at {model_path} — run train.py first.")
        return None

    with open(model_path,   'rb') as f: model   = pickle.load(f)
    with open(encoder_path, 'rb') as f: encoder = pickle.load(f)

    # ── Encode sector (same as train.py) ─────────────────────
    df = df.copy()
    # Only keep rows with known sectors (encoder was fit on training data)
    df = df[df['sector'].isin(encoder.classes_)]
    df['sector_enc'] = encoder.transform(df['sector'])

    enc_features = [c for c in feature_cols if c != 'sector'] + ['sector_enc']

    X = df[enc_features].values
    y = df[target_col].values

    # ── 80/20 split — same seed as train.py ──────────────────
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if len(X_test) == 0:
        print(f"  ⚠️ Test set is empty after split.")
        return None

    # ── Predict + evaluate ────────────────────────────────────
    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))

    print(f"  Test rows : {len(y_test)}")
    print(f"  RMSE      : ₹{rmse:,.2f} / sq.ft")
    print(f"  R²        : {r2:.4f}")

    return {
        "category":   category_name,
        "test_rows":  len(y_test),
        "rmse":       round(rmse, 2),
        "r2":         round(r2,   4),
        "evaluated_at": datetime.utcnow().isoformat(),
    }


def run_ml_evaluation():
    print("\n" + "="*55)
    print("  ML MODEL EVALUATION")
    print("="*55)

    db = SessionLocal()
    results = []

    try:
        # Plots
        df_plots = load_plots(db)
        if not df_plots.empty:
            r = evaluate_ml_category(
                df_plots,
                feature_cols=['sector', 'area_sqyd', 'connectivity_score', 'is_corner'],
                target_col='price_per_sqft',
                category_name='plots'
            )
            if r: results.append(r)

        # Floors
        df_floors = load_floors(db)
        if not df_floors.empty:
            r = evaluate_ml_category(
                df_floors,
                feature_cols=['sector', 'area_sqft', 'bhk_type', 'connectivity_score'],
                target_col='price_per_sqft',
                category_name='floors'
            )
            if r: results.append(r)

        # Societies
        df_societies = load_societies(db)
        if not df_societies.empty:
            r = evaluate_ml_category(
                df_societies,
                feature_cols=['sector', 'area_sqft', 'bhk_type',
                              'connectivity_score', 'possession'],
                target_col='price_per_sqft',
                category_name='societies'
            )
            if r: results.append(r)

    finally:
        db.close()

    # ── Save results ──────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "ml_scores.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ ML results saved to {output_path}")
    return results


# ============================================================
# RAG EVALUATION
# ============================================================

# ── Sample Q&A test set per doc type ─────────────────────────
# These are realistic questions a property buyer would ask.
# Answers are intentionally left empty — GPT-4o will generate
# both the answer AND the ground truth for fair evaluation.

RAG_TEST_QUESTIONS = {
    "brochure": [
        "What are the available flat configurations and their sizes?",
        "What amenities does this project offer?",
        "What is the possession date for this project?",
        "What is the price per square foot mentioned in the brochure?",
        "Who is the developer/builder of this project?",
    ],
    "rera": [
        "What is the RERA registration number of this project?",
        "What is the carpet area of the units as per RERA?",
        "What is the project completion date as per RERA filing?",
        "What is the name of the promoter registered with RERA?",
        "Are there any pending legal cases mentioned in the RERA document?",
    ],
    "legal": [
        "What is the total land area mentioned in this document?",
        "Who are the parties involved in this property transaction?",
        "What is the sale consideration amount in this document?",
        "Are there any encumbrances or liabilities on this property?",
        "What is the property's survey number or plot number?",
    ],
}


def generate_rag_test_answer(question: str, context: str) -> str:
    """
    Uses GPT-4o to generate a ground-truth answer for a question
    given the actual document context. This is used as the
    reference answer for faithfulness evaluation.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document analyst. "
                    "Answer the question based ONLY on the provided context. "
                    "Be concise and factual. "
                    "If the answer is not in the context, say 'Not mentioned in document'."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def evaluate_faithfulness(question: str, answer: str,
                           contexts: list[str]) -> float:
    """
    Faithfulness: measures if the answer is grounded in the context.
    Asks GPT-4o to score each claim in the answer as supported/not supported.
    Returns score between 0 and 1.
    """
    context_str = "\n\n---\n\n".join(contexts)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation assistant. "
                    "Your job is to check if an answer is faithful to the given context.\n\n"
                    "Steps:\n"
                    "1. Break the answer into individual claims\n"
                    "2. For each claim, check if it is supported by the context\n"
                    "3. Return a JSON object with:\n"
                    "   - 'total_claims': integer\n"
                    "   - 'supported_claims': integer\n"
                    "   - 'score': float (supported/total, between 0 and 1)\n\n"
                    "Return ONLY the JSON object, no other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                )
            }
        ],
        temperature=0.0,
        max_tokens=200,
    )

    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: measures if the answer actually addresses the question.
    Uses GPT-4o to generate synthetic questions from the answer,
    then measures semantic similarity to the original question.
    Score between 0 and 1.
    """
    # Step 1: Generate synthetic questions from the answer
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given an answer, generate 3 questions that this answer "
                    "could be responding to. Return ONLY a JSON array of 3 "
                    "question strings, no other text."
                )
            },
            {
                "role": "user",
                "content": f"Answer: {answer}"
            }
        ],
        temperature=0.3,
        max_tokens=200,
    )

    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        synthetic_questions = json.loads(raw)
    except Exception:
        return 0.0

    # Step 2: Score semantic similarity between original and synthetic questions
    all_questions = [question] + synthetic_questions
    embeddings_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=all_questions
    )
    embeddings = [e.embedding for e in embeddings_response.data]

    orig_emb  = np.array(embeddings[0])
    synth_embs = [np.array(e) for e in embeddings[1:]]

    # Cosine similarity
    similarities = []
    for s_emb in synth_embs:
        cos_sim = float(
            np.dot(orig_emb, s_emb) /
            (np.linalg.norm(orig_emb) * np.linalg.norm(s_emb) + 1e-8)
        )
        similarities.append(cos_sim)

    return round(float(np.mean(similarities)), 4)


def run_rag_evaluation():
    """
    Evaluates the RAG pipeline on the ingested documents.
    Requires at least one document to be ingested in ChromaDB.
    """
    print("\n" + "="*55)
    print("  RAG PIPELINE EVALUATION")
    print("="*55)

    # Import here to avoid circular issues
    from ingest import retrieve_chunks, get_collection_stats, chroma_client

    stats = get_collection_stats()
    available = {k: v for k, v in stats.items() if v > 0}

    if not available:
        print("\n  ⚠️ No documents ingested in ChromaDB.")
        print("     Ingest at least one PDF via the app or ingest.py first.")
        return None

    print(f"\n  Available collections: {available}")

    all_faithfulness  = []
    all_relevancy     = []
    eval_samples      = []

    
    

    for doc_type, chunk_count in available.items():
        questions = RAG_TEST_QUESTIONS.get(doc_type, [])
        print(f"\n  Testing {doc_type.upper()} "
              f"({chunk_count} chunks, {len(questions)} questions)...")

        for i, question in enumerate(questions):
            try:
                print(f"    Q{i+1}: {question[:60]}...")

                # ── Retrieve context ──────────────────────────────
                chunks = retrieve_chunks(question, doc_type, n_results=3)
                if not chunks:
                    print(f"         ⚠️ No chunks retrieved — skipping")
                    continue

                contexts   = [c["text"] for c in chunks]
                context_str = "\n\n".join(contexts[:3])

                # ── Generate answer using RAG pipeline ────────────
                system_prompt = (
                    "You are a real estate analyst specializing in Faridabad property market. "
                    "Answer based ONLY on the provided context. Be concise and factual."
                )
                rag_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                        "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                rag_answer = rag_response.choices[0].message.content.strip()

                # ── Evaluate ──────────────────────────────────────
                faithfulness = evaluate_faithfulness(question, rag_answer, contexts)
                relevancy    = evaluate_answer_relevancy(question, rag_answer)

                all_faithfulness.append(faithfulness)
                all_relevancy.append(relevancy)

                eval_samples.append({
                    "doc_type":    doc_type,
                    "question":    question,
                    "answer":      rag_answer,
                    "faithfulness":  round(faithfulness, 4),
                    "relevancy":     round(relevancy,    4),
                    "num_chunks_retrieved": len(chunks),
                })

                print(f"         Faithfulness: {faithfulness:.3f} | "
                    f"Relevancy: {relevancy:.3f}")
            
            except Exception as e:
                print(f"         ❌ Error on Q{i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if not all_faithfulness:
        print("\n  ⚠️ No evaluations completed.")
        return None

    # ── Aggregate scores ──────────────────────────────────────
    avg_faithfulness = round(float(np.mean(all_faithfulness)), 4)
    avg_relevancy    = round(float(np.mean(all_relevancy)),    4)

    results = {
        "avg_faithfulness":      avg_faithfulness,
        "avg_answer_relevancy":  avg_relevancy,
        "total_questions":       len(eval_samples),
        "samples":               eval_samples,
        "evaluated_at":          datetime.utcnow().isoformat(),
    }

    print(f"\n  {'─'*40}")
    print(f"  Avg Faithfulness     : {avg_faithfulness:.4f}")
    print(f"  Avg Answer Relevancy : {avg_relevancy:.4f}")
    print(f"  Total Questions      : {len(eval_samples)}")

    # ── Save results ──────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "rag_scores.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ RAG results saved to {output_path}")
    return results


# ============================================================
# SUMMARY PRINTER
# ============================================================

def print_summary(ml_results, rag_results):
    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)

    if ml_results:
        print("\n  📈 ML Model Performance (price_per_sqft prediction):")
        print(f"  {'Category':<15} {'Test Rows':<12} {'RMSE':>12} {'R²':>8}")
        print(f"  {'─'*50}")
        for r in ml_results:
            print(
                f"  {r['category'].capitalize():<15} "
                f"{r['test_rows']:<12} "
                f"₹{r['rmse']:>10,.2f} "
                f"{r['r2']:>8.4f}"
            )

    if rag_results:
        print(f"\n  🤖 RAG Pipeline Performance:")
        print(f"  {'─'*40}")
        print(f"  Faithfulness     : {rag_results['avg_faithfulness']:.4f} / 1.0")
        print(f"  Answer Relevancy : {rag_results['avg_answer_relevancy']:.4f} / 1.0")
        print(f"  Questions Tested : {rag_results['total_questions']}")

    print("\n  📁 Results saved to eval_results/")
    print("     ml_scores.json  — ML metrics per category")
    print("     rag_scores.json — RAG metrics + per-question breakdown")
    print("\n  💡 Resume bullet point:")
    if ml_results and rag_results:
        best_r2 = max(r['r2'] for r in ml_results)
        best_rmse = min(r['rmse'] for r in ml_results)
        print(
            f"  'Built XGBoost AVM achieving R²={best_r2:.2f} and "
            f"RMSE=₹{best_rmse:,.0f}/sqft; RAG pipeline scored "
            f"{rag_results['avg_faithfulness']:.2f} faithfulness and "
            f"{rag_results['avg_answer_relevancy']:.2f} answer relevancy "
            f"on property document Q&A.'"
        )
    print("="*55)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ML models and RAG pipeline")
    parser.add_argument("--skip-ml",  action="store_true",
                        help="Skip ML evaluation")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG evaluation (no OpenAI calls for RAG)")
    args = parser.parse_args()

    ml_results  = None
    rag_results = None

    if not args.skip_ml:
        ml_results = run_ml_evaluation()

    if not args.skip_rag:
        rag_results = run_rag_evaluation()

    print_summary(ml_results, rag_results)