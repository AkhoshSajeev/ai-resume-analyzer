import os
import re
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
from pypdf import PdfReader

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

SKILLS = [

    "Python",
    "Java",
    "C",
    "C++",
    "JavaScript",
    "HTML",
    "CSS",
    "Flask",
    "Django",
    "SQL",
    "MySQL",
    "MongoDB",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Data Science",
    "Git",
    "Docker",
    "REST API",
    "Node.js",
    "React",
    "Linux",
    "AWS"

]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    # Get uploaded file
    resume = request.files['resume']
    job_description = request.form['job_description']
    # Save file
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        resume.filename
    )

    resume.save(filepath)

    # Read PDF
    reader = PdfReader(filepath)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    # -----------------------------
    # Basic Information Extraction
    # -----------------------------

    # Extract email
    email_match = re.search(
        r'[\w\.-]+@[\w\.-]+',
        text
    )

    email = (
        email_match.group(0)
        if email_match else "Not Found"
    )

    # Extract phone number
    phone_match = re.search(
        r'\+?\d[\d\s\-]{8,15}',
        text
    )

    phone = (
        phone_match.group(0)
        if phone_match else "Not Found"
    )

    # -----------------------------
    # Skill Extraction
    # -----------------------------

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text.lower():

            found_skills.append(skill)

    # -----------------------------
    # AI Resume Analysis
    # -----------------------------

    prompt = f"""
    Analyze this resume.

    Give:
    1. Candidate summary
    2. Main skills
    3. Strengths
    4. Weaknesses
    5. Suggested improvements
    6. Recommended career roles

    Resume Text:
    {text}
    """

    response = ollama.chat(
        model='mistral',

        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    analysis = response['message']['content']
    resume_embedding = embedding_model.encode([text])

    job_embedding = embedding_model.encode(
    [job_description]
    )
    similarity = cosine_similarity(
    resume_embedding,
    job_embedding
     )[0][0]

    match_score = round(similarity * 100, 2)
    # -----------------------------
    # Resume Score
    # -----------------------------

    score = 0

    if email != "Not Found":
        score += 20

    if phone != "Not Found":
        score += 20

    score += min(len(found_skills) * 5, 40)

    if len(text) > 1000:
        score += 20

    # -----------------------------
    # Return Result
    # -----------------------------

    return f"""
    <h1>Resume Analysis</h1>
    <h2>Job Match Score</h2>
    <p>{match_score}%</p>
    <h2>Resume Score</h2>
    <p>{score}/100</p>

    <h2>Email</h2>
    <p>{email}</p>

    <h2>Phone</h2>
    <p>{phone}</p>

    <h2>Detected Skills</h2>
    <p>{", ".join(found_skills)}</p>

    <h2>AI Analysis</h2>
    <pre>{analysis}</pre>
    """


if __name__ == '__main__':
    app.run(debug=True)