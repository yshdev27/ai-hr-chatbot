from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import requests
import json

app = FastAPI()

# Model and Data Loading
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
employees = [
    {"id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS"], "experience_years": 5, "projects": ["E-commerce Platform", "Healthcare Dashboard"], "availability": "available"},
    {"id": 2, "name": "Bob Williams", "skills": ["Java", "Spring", "Docker"], "experience_years": 8, "projects": ["Banking System", "Cloud Migration"], "availability": "on-project"},
    {"id": 3, "name": "Charlie Brown", "skills": ["JavaScript", "Node.js", "MongoDB"], "experience_years": 3, "projects": ["Real-time Chat App", "API Development"], "availability": "available"},
    {"id": 4, "name": "Diana Prince", "skills": ["C#", ".NET", "Azure"], "experience_years": 6, "projects": ["CRM Software", "Legacy System Modernization"], "availability": "available"},
    {"id": 5, "name": "Ethan Hunt", "skills": ["React Native", "Firebase", "iOS"], "experience_years": 4, "projects": ["Mobile Banking App", "Social Media App"], "availability": "available"},
    {"id": 6, "name": "Fiona Glenanne", "skills": ["Python", "Flask", "PostgreSQL"], "experience_years": 7, "projects": ["REST API for Fintech", "Data Pipeline"], "availability": "on-project"},
    {"id": 7, "name": "Grace Hopper", "skills": ["COBOL", "Mainframe", "SQL"], "experience_years": 20, "projects": ["Compiler Design", "Database Systems"], "availability": "available"},
    {"id": 8, "name": "Henry Jones Jr.", "skills": ["Archaeology", "History", "Cryptography"], "experience_years": 15, "projects": ["Ancient Artifact Analysis", "Obscure Language Translation"], "availability": "available"},
    {"id": 9, "name": "Ivy Valentine", "skills": ["Unreal Engine", "C++", "3D Modeling"], "experience_years": 5, "projects": ["AAA Game Development", "Virtual Reality Simulation"], "availability": "on-project"},
    {"id": 10, "name": "Jack Sparrow", "skills": ["Navigation", "Negotiation", "Rum Tasting"], "experience_years": 12, "projects": ["Black Pearl Acquisition", "Fountain of Youth Expedition"], "availability": "available"},
    {"id": 11, "name": "Karen Page", "skills": ["Investigative Journalism", "Legal Research", "MS Office"], "experience_years": 4, "projects": ["Union Allied Corruption Report", "Nelson and Murdock Case Files"], "availability": "available"},
    {"id": 12, "name": "Luke Skywalker", "skills": ["Piloting", "Lightsaber Combat", "The Force"], "experience_years": 6, "projects": ["Death Star Destruction", "Jedi Training on Dagobah"], "availability": "on-project"},
    {"id": 13, "name": "Michael Scott", "skills": ["Sales", "Management", "Improv Comedy"], "experience_years": 15, "projects": ["Dunder Mifflin Scranton Branch", "Threat Level Midnight"], "availability": "available"},
    {"id": 14, "name": "Dr. Sarah Chen", "skills": ["Machine Learning", "TensorFlow", "Medical Imaging", "Python"], "experience_years": 6, "projects": ["Medical Diagnosis Platform", "Patient Risk Prediction"], "availability": "available"},
    {"id": 15, "name": "Michael Rodriguez", "skills": ["Machine Learning", "Scikit-learn", "Pandas", "HIPAA"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "Healthcare Data Analysis"], "availability": "on-project"}
]

employee_documents = [f"Name: {e['name']}. Experience: {e['experience_years']} years. Skills: {', '.join(e['skills'])}. Projects: {', '.join(e['projects'])}." for e in employees]
employee_embeddings = retrieval_model.encode(employee_documents)

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
def chat(query: ChatQuery):
    query_embedding = retrieval_model.encode(query.query)
    top_results = util.semantic_search(query_embedding, employee_embeddings, top_k=2)
    response_text = "Based on your requirements, here are my top recommendations:\n\n"
    for result in top_results[0]:
        employee = employees[result['corpus_id']]
        prompt = f"""
        You are an expert HR assistant. A user has the following request: "{query.query}"
        Here is the profile of a top-matched candidate: {json.dumps(employee)}
        Your Task: Write a brief, personalized recommendation for this candidate in 1-2 sentences. Highlight their specific skills and project experience make them a strong fit.
        """
        try:
            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi3:mini", "prompt": prompt, "stream": False},
                timeout=120
            )
            ollama_response.raise_for_status()
            generated_text = ollama_response.json()['response']
            response_text += f"**{employee['name']}**\n{generated_text}\n\n"
        except requests.exceptions.RequestException as e:
            response_text += f"**{employee['name']}**\nCould not generate a detailed summary. Is the Ollama server running? Error: {e}\n\n"
    return {"response": response_text}