# app.py
from flask import Flask, request, jsonify, render_template, session
import os
import google.generativeai as genai
from rag_model import save_embeddings, answer_query

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK", "key")


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print('✅ Gemini AI enabled (using the latest flash model)')
else:
    gemini_model = None
    print('⚠️  GEMINI_API_KEY not set - Gemini mode will be unavailable')

# Ensure embeddings are prepared at startup
try:
    save_embeddings()
    print("Embeddings saved/updated.")
except Exception as e:
    print("Error preparing embeddings:", e)

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "hurt myself", "harm myself", "i'm going to kill",
    "don't want to live", "better off dead"
]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or "").strip()
    selected_mode = data.get("mode", "hybrid")  # Default to hybrid
    
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # Check for crisis keywords
    lower = user_input.lower()
    if any(keyword in lower for keyword in CRISIS_KEYWORDS):
        crisis_msg = (
            "I'm sorry you're feeling this way. If you're in immediate danger, please call local emergency number 14416 (Tele MANAS) to connect with a counsellor"
            "Would you like resources for coping strategies or to find professional help?"
        )
        return jsonify({"response": crisis_msg, "is_crisis": True})

    # MODE: RAG Only
    if selected_mode == "rag":
        kb_results = answer_query(user_input, top_k=1)
        
        if not kb_results or kb_results[0]['score'] < 0.2:
            return jsonify({
                "response": "Sorry, I couldn't find a relevant answer in the knowledge base. Try rephrasing or ask another question.",
                "mode": "rag"
            })
        
        best = kb_results[0]
        reply = f"{best['answer']}\n\n📚 *Source: {best['question']}*"
        
        return jsonify({
            "response": reply, 
            "source_question": best.get("question"), 
            "score": best.get("score"),
            "mode": "rag"
        })

    # MODE: Gemini Only
    elif selected_mode == "gemini":
        if not gemini_model:
            return jsonify({
                "response": "Gemini mode is not available. Please set GEMINI_API_KEY or switch to RAG/Hybrid mode.",
                "mode": "error"
            })
        
        try:
            prompt = f"""You are MindSpace, a compassionate mental health support assistant. 

Provide a warm, empathetic response to this mental health question. Keep it concise (2-3 paragraphs).

User Question: {user_input}

Response:"""

            response = gemini_model.generate_content(prompt)
            
            return jsonify({
                "response": response.text,
                "mode": "gemini"
            })
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return jsonify({
                "response": f"Error communicating with Gemini AI: {str(e)}. Please try again or switch to RAG mode.",
                "mode": "error"
            })

    # MODE: Hybrid (RAG + Gemini)
    else:  # selected_mode == "hybrid"
        kb_results = answer_query(user_input, top_k=3)
        context = ""
        
        if kb_results and kb_results[0]['score'] > 0.2:
            context = "\n\n".join([
                f"Reference: {r['question']}\n{r['answer']}" 
                for r in kb_results[:2]
            ])

        if gemini_model:
            try:
                if context:
                    prompt = f"""You are MindSpace, a compassionate mental health support assistant. 

Provide a warm, empathetic response to this question. Keep it concise (2-3 paragraphs). Use the reference information below to ground your response.

Here is relevant information from our knowledge base:
{context}

User Question: {user_input}

Response:"""
                else:
                    prompt = f"""You are MindSpace, a compassionate mental health support assistant. 

Provide a warm, empathetic response to this mental health question. Keep it concise (2-3 paragraphs).

User Question: {user_input}

Response:"""

                response = gemini_model.generate_content(prompt)
                
                return jsonify({
                    "response": response.text,
                    "mode": "hybrid",
                    "has_context": bool(context)
                })
                
            except Exception as e:
                print(f"Gemini error in hybrid mode: {e}")

        # Fallback to RAG if Gemini fails or unavailable
        if not kb_results or kb_results[0]['score'] < 0.2:
            return jsonify({
                "response": "Sorry, I couldn't find an answer. Try rephrasing or ask another question.",
                "mode": "rag_fallback"
            })
        
        best = kb_results[0]
        reply = f"{best['answer']}\n\n📚 *Note: Gemini unavailable, using knowledge base*"
        
        return jsonify({
            "response": reply, 
            "source_question": best.get("question"), 
            "score": best.get("score"),
            "mode": "rag_fallback"
        })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
