from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_session import Session
import google.generativeai as genai
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'UV0xs_vaBEHNDPwTEAyMVs9a8GBvveYjhj6ZI5Oj9jw'  # Replace with a strong secret key in production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes
Session(app)

CORS(app, supports_credentials=True)

# Configure Gemini - replace with your actual API key
GENAI_KEY = 'AIzaSyCY5Co0OYXiSzVrLmFV3ulfH1cQoY5VttM'  # Replace with your actual Gemini API key
genai.configure(api_key=GENAI_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load and initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
# Load course and instructor data from JSON files
with open("data/courses.json", "r") as f:
    courses = json.load(f)

with open("data/instructors.json", "r") as f:
    instructors = json.load(f)

# Pre-compute embeddings for courses and instructors
course_embeddings = {course['name']: sentence_model.encode(course['name'] + " " + course['description']) for course in courses}
instructor_embeddings = {instructor['name']: sentence_model.encode(instructor['name'] + " " + instructor['bio']) for instructor in instructors}

def retrieve_relevant_info(query):
    """Retrieve relevant courses and instructors based on the query."""
    query_embedding = sentence_model.encode(query)
    
    # Find relevant courses
    course_scores = []
    for name, embedding in course_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        course_scores.append((similarity, name))
    
    # Find relevant instructors
    instructor_scores = []
    for name, embedding in instructor_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        instructor_scores.append((similarity, name))
    
    # Sort by similarity and get top 3
    relevant_courses = [name for score, name in sorted(course_scores, reverse=True)[:3]]
    relevant_instructors = [name for score, name in sorted(instructor_scores, reverse=True)[:3]]
    
    # Get full course and instructor details
    matched_courses = [course for course in courses if course['name'] in relevant_courses]
    matched_instructors = [instructor for instructor in instructors if instructor['name'] in relevant_instructors]
    
    return matched_courses, matched_instructors

def format_context(courses, instructors):
    """Format the context for the prompt."""
    context = ""
    
    if courses:
        context += "Relevant Courses:\n"
        for course in courses:
            context += f"- {course['name']}: {course['description']} (Instructor: {course['instructor']}, Duration: {course['duration']})\n"
    
    if instructors:
        context += "\nRelevant Instructors:\n"
        for instructor in instructors:
            context += f"- {instructor['name']}: {instructor['bio']}\n"
    
    return context.strip()

def build_prompt_with_context(question, context, conversation_history):
    """Build the prompt with context and conversation history."""
    prompt = f"""
    Context information:
    {context}

    Conversation history:
    """
    
    for msg in conversation_history[1:]:  # Skip system message
        prompt += f"{msg['role']}: {msg['content']}\n"
    
    prompt += f"""
    Current question: {question}
    
    Please provide a helpful response based on the context and conversation history.
    """
    
    return prompt

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.json
        question = data.get("query", "").strip()
        conversation_id = data.get("conversationId")
        
        if not question:
            return jsonify({"response": "Please enter a valid question."})

        # Initialize or retrieve conversation history
        if 'conversations' not in session:
            session['conversations'] = {}
            
        if conversation_id not in session['conversations']:
            session['conversations'][conversation_id] = [
                {"role": "system", "content": "You are a helpful course enquiry assistant."}
            ]
            session.modified = True

        # Add user question to conversation history
        session['conversations'][conversation_id].append({"role": "user", "content": question})
        
        # Retrieve relevant information
        courses, instructors = retrieve_relevant_info(question)
        context = format_context(courses, instructors)

        # Construct prompt
        prompt = build_prompt_with_context(question, context, session['conversations'][conversation_id])

        try:
            gemini_response = model.generate_content(prompt)
            response = gemini_response.text.strip()
            
            # Add assistant response to conversation history
            session['conversations'][conversation_id].append({"role": "assistant", "content": response})
            session.modified = True
            
        except Exception as e:
            response = f"I couldn't process that request. Please try again. (Error: {str(e)})"

        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"})

@app.route("/api/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        text = data.get("text", "").strip()
        target_lang = data.get("target_lang", "es")
        conversation_id = data.get("conversationId")
        
        if not text:
            return jsonify({"translation": "No text to translate"})
        
        # Create a translation prompt
        prompt = f"""
        Translate the following English text to {target_lang}. 
        Keep the meaning accurate but make the translation sound natural in the target language.
        
        Text to translate: {text}
        """
        
        try:
            gemini_response = model.generate_content(prompt)
            translation = gemini_response.text.strip()
            
            # Store translation in session if needed
            if 'translations' not in session:
                session['translations'] = {}
                
            if conversation_id not in session['translations']:
                session['translations'][conversation_id] = []
                
            session['translations'][conversation_id].append({
                "original": text,
                "translation": translation,
                "language": target_lang
            })
            session.modified = True
            
            return jsonify({"translation": translation})
        
        except Exception as e:
            return jsonify({"translation": f"Translation failed: {str(e)}"})
    
    except Exception as e:
        return jsonify({"translation": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
