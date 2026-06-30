# 🎤 AI Interview Assistant

An AI-powered desktop application that simulates realistic technical and HR interviews using speech recognition, adaptive questioning, and intelligent performance evaluation. The system combines a Python desktop frontend with a Flask backend powered by Whisper and Groq LLMs to provide a complete mock interview experience.

---

## 📌 Overview

Preparing for technical interviews often requires repeated practice, structured feedback, and realistic interview simulations. This project was built to provide an interactive interview platform that evaluates not only what a candidate says but also how they deliver their answers.

The application acts as an AI interviewer by asking role-specific questions, transcribing spoken responses, evaluating answer quality, providing detailed feedback, and generating a comprehensive interview report.

---

## ✨ Features

### 🎯 Realistic AI Interview

* Role-based interview sessions
* Adaptive follow-up questions
* Multiple interviewer personalities
* Dynamic interview flow
* Configurable interview duration
* Configurable number of questions

---

### 💼 Multiple Interview Roles

Supports interviews for:

* Software Engineer Intern
* Frontend Developer
* Backend Developer
* Data Analyst
* HR Round

Each role uses different interview styles and question categories.

---

### 👤 Interviewer Personas

Choose from different interviewer personalities including:

* Friendly HR
* Strict Technical Lead
* Startup Founder
* Calm Senior Engineer
* Fast-paced Recruiter

Each persona influences the tone and style of questioning.

---

### 🎙 Speech Recognition

The backend uses OpenAI Whisper to:

* Transcribe spoken answers
* Detect spoken content
* Process microphone input
* Support natural conversational interviews

---

### 🤖 AI Evaluation

Each response is evaluated using Groq's Llama model.

The AI analyzes:

* Relevance
* Clarity
* Structure
* Technical depth
* Specificity
* Impact

It also generates:

* Numerical score
* Strengths
* Weaknesses
* Suggestions
* Improved sample answer
* STAR analysis
* Ideal answer hints
* Adaptive next interview question

---

### 📊 Delivery Analysis

The system evaluates speaking performance by estimating:

* Confidence
* Words per minute
* Filler words
* Speaking pace

---

### 📷 Camera Monitoring

The frontend uses MediaPipe Face Mesh to monitor:

* Eye contact
* Attention level

These metrics are displayed live during the interview.

---

### 📈 Final Interview Report

At the end of each interview, the application generates a comprehensive report containing:

* Overall score
* Hire readiness assessment
* Communication feedback
* Technical feedback
* Strengths
* Weaknesses
* Recommended practice plan
* 7-day improvement roadmap

Interview sessions are automatically saved locally for future review.

---

## 🖥 User Interface

The desktop application is built using **CustomTkinter** and provides:

* Modern dark-themed interface
* Live webcam preview
* Microphone activity visualization
* Interview timer
* Question display
* Real-time feedback
* Detailed report tab
* AI voice support using Text-to-Speech

---

## 🏗 Project Architecture

```text
Frontend (CustomTkinter Desktop App)
        │
        │ HTTP Requests
        ▼
Flask Backend
        │
 ├── Whisper Speech-to-Text
 ├── Groq LLM Evaluation
 ├── Speech Analysis
 └── Report Generation
```

---

## 🛠 Technologies Used

### Frontend

* Python
* CustomTkinter
* OpenCV
* MediaPipe
* NumPy
* Pillow
* SoundDevice
* SciPy
* Requests
* pyttsx3

### Backend

* Flask
* OpenAI Whisper
* Groq API
* Llama 3.1 8B Instant
* pyngrok
* JSON

---

## 📂 Project Structure

```text
AI_Interview_Assistant
│
├── main.py                 # Desktop frontend
├── backend.py              # Flask backend
├── backend.ipynb           # Google Colab backend notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How It Works

1. Launch the desktop application.
2. Connect it to the Flask backend using the ngrok URL.
3. Configure:

   * Interview role
   * Interviewer persona
   * Interview mode
   * Duration
   * Question limit
4. Start the interview.
5. The AI asks the first question.
6. Respond using your microphone.
7. Whisper transcribes your response.
8. Groq evaluates your answer.
9. Feedback and scores are displayed instantly.
10. The AI asks the next question.
11. After the interview, a complete report is generated.

---

## ⚙ Installation

### Clone the repository

```bash
git clone https://github.com/saptarshidas578/AI_Interview_Assistant.git
```

```bash
cd AI_Interview_Assistant
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Backend Setup

Run the backend in Google Colab.

You will need:

* Groq API Key
* ngrok Authentication Token

The backend automatically creates a public URL using ngrok.

Copy this URL into the desktop application's **Backend URL** field.

---

### Start the Frontend

Run:

```bash
python main.py
```

Connect to the backend and begin the interview.

---

## 📊 Evaluation Metrics

Each answer is evaluated across multiple dimensions:

* Relevance
* Clarity
* Structure
* Technical Depth
* Specificity
* Impact
* Communication Quality
* Speaking Confidence
* STAR Method Coverage

---

## 🎯 Future Improvements

Potential enhancements include:

* Resume PDF parsing
* OCR-based resume upload
* Company-specific interview modes
* Multi-language interview support
* Video recording
* Emotion detection
* Voice emotion analysis
* Cloud database for interview history
* User authentication
* Web deployment
* Analytics dashboard

---

## 👨‍💻 Author

**Saptarshi Das**

GitHub:
https://github.com/saptarshidas578

---

## 📄 License

This project is intended for educational, learning, and portfolio purposes.

---

## ⭐ If you found this project useful

If you like this project, consider giving it a ⭐ on GitHub. It helps others discover the project and supports future improvements.
