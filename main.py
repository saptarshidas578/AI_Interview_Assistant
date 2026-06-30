import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write
import requests
import os
import json
import threading
import time
import queue
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from enum import Enum, auto
from datetime import datetime
import pyttsx3


# =========================
# CONFIG
# =========================

SERVER_URL = "https://existing-entryway-demise.ngrok-free.dev/"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ROLES = [
    "Software Engineer Intern",
    "Frontend Developer",
    "Backend Developer",
    "Data Analyst",
    "HR Round",
]

PERSONAS = [
    "Friendly HR",
    "Strict Technical Lead",
    "Startup Founder",
    "Calm Senior Engineer",
    "Fast-paced Recruiter",
]

INTERVIEW_MODES = [
    "Quick Practice",
    "Full Interview",
    "Behavioral Round",
    "Technical Round",
    "Resume-Based Interview",
    "Job Description Match",
]

FALLBACK_QUESTIONS = [
    "Tell me about a project you are proud of and your exact contribution.",
    "Describe a challenge you faced and how you solved it.",
    "Can you give me a specific example where you learned something quickly?",
    "Why are you interested in this role?",
]


class InterviewState(Enum):
    IDLE = auto()
    AI_SPEAKING = auto()
    READY_TO_RECORD = auto()
    USER_RECORDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()


class InterviewAssistant(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Interview Assistant")
        self.geometry("1280x820")
        self.minsize(1120, 740)

        self.server_url = SERVER_URL.rstrip("/")
        self.interview_state = InterviewState.IDLE
        self.history = []
        self.current_question = ""
        self.session_started_at = None
        self.interview_start_time = 0
        self.max_duration_seconds = 600
        self.question_limit = 10
        self.turn_count = 0

        self.smooth_attention = 100.0
        self.smooth_eye_contact = 100.0
        self.ema_alpha = 0.15

        self.fs = 16000
        self.is_recording = False
        self.audio_data = []
        self.audio_stream = None
        self.answer_start_time = 0
        self.actual_duration = 0
        self.latest_mic_level = 0.0

        self.camera_running = False
        self.video_ui_every_n_frames = 3
        self.face_process_every_n_frames = 8
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.tts_enabled = ctk.BooleanVar(value=True)
        self.coaching_enabled = ctk.BooleanVar(value=True)
        self.tts_queue = queue.Queue()
        self.tts_stop_event = threading.Event()
        threading.Thread(target=self._tts_worker, daemon=True).start()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._build_ui()
        self._update_mic_visualizer()
        self._set_ready_state()

    # =========================
    # UI
    # =========================

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=310, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.workspace = ctk.CTkFrame(self, fg_color="#101418", corner_radius=0)
        self.workspace.grid(row=0, column=1, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        self._build_sidebar()
        self._build_header()
        self._build_main_area()

    def _build_sidebar(self):
        title = ctk.CTkLabel(self.sidebar, text="Interview Setup", font=("Arial", 24, "bold"))
        title.pack(anchor="w", padx=18, pady=(20, 8))

        self.backend_label = ctk.CTkLabel(self.sidebar, text="Backend URL", anchor="w")
        self.backend_label.pack(fill="x", padx=18, pady=(10, 4))
        self.server_entry = ctk.CTkEntry(self.sidebar)
        self.server_entry.insert(0, self.server_url)
        self.server_entry.pack(fill="x", padx=18)

        self.health_button = ctk.CTkButton(self.sidebar, text="Check Backend", command=self.check_backend)
        self.health_button.pack(fill="x", padx=18, pady=(10, 12))

        self.role_menu = self._labeled_menu("Role", ROLES, "Software Engineer Intern")
        self.persona_menu = self._labeled_menu("Interviewer", PERSONAS, "Friendly HR")
        self.mode_menu = self._labeled_menu("Mode", INTERVIEW_MODES, "Full Interview")

        self.duration_menu = self._labeled_menu("Duration", ["5 minutes", "10 minutes", "15 minutes"], "10 minutes")
        self.limit_menu = self._labeled_menu("Questions", ["3", "5", "8", "10"], "8")

        self.resume_box = self._labeled_textbox("Resume / Project Context", height=82)
        self.jd_box = self._labeled_textbox("Job Description", height=82)

        self.tts_switch = ctk.CTkSwitch(self.sidebar, text="AI voice", variable=self.tts_enabled)
        self.tts_switch.pack(anchor="w", padx=18, pady=(12, 4))

        self.coaching_switch = ctk.CTkSwitch(self.sidebar, text="Show coaching hints", variable=self.coaching_enabled)
        self.coaching_switch.pack(anchor="w", padx=18, pady=(4, 14))

        self.start_button = ctk.CTkButton(self.sidebar, text="Start Interview", height=42, command=self.start_interview)
        self.start_button.pack(fill="x", padx=18, pady=(4, 8))

        self.end_button = ctk.CTkButton(self.sidebar, text="End & Generate Report", fg_color="#34495E", command=self.end_interview, state="disabled")
        self.end_button.pack(fill="x", padx=18)

    def _labeled_menu(self, label, values, default):
        ctk.CTkLabel(self.sidebar, text=label, anchor="w").pack(fill="x", padx=18, pady=(8, 4))
        menu = ctk.CTkComboBox(self.sidebar, values=values)
        menu.set(default)
        menu.pack(fill="x", padx=18)
        return menu

    def _labeled_textbox(self, label, height):
        ctk.CTkLabel(self.sidebar, text=label, anchor="w").pack(fill="x", padx=18, pady=(8, 4))
        box = ctk.CTkTextbox(self.sidebar, height=height, wrap="word")
        box.pack(fill="x", padx=18)
        return box

    def _build_header(self):
        self.header = ctk.CTkFrame(self.workspace, fg_color="#101418", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.header, text="AI Interview Assistant", font=("Arial", 28, "bold"))
        self.title_label.grid(row=0, column=0, sticky="w")

        self.timer_label = ctk.CTkLabel(self.header, text="00:00", font=("Arial", 26, "bold"), text_color="#F1C40F")
        self.timer_label.grid(row=0, column=1, sticky="e", padx=(20, 0))

        self.status_label = ctk.CTkLabel(self.header, text="Ready", font=("Arial", 14), text_color="#2ECC71")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _build_main_area(self):
        self.main = ctk.CTkFrame(self.workspace, fg_color="#101418", corner_radius=0)
        self.main.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.main.grid_columnconfigure(0, weight=0)
        self.main.grid_columnconfigure(1, weight=1)
        self.main.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkScrollableFrame(self.main, fg_color="#171D23", corner_radius=8, width=520)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        

        self.right_panel = ctk.CTkFrame(self.main, fg_color="#171D23", corner_radius=8)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        ctk.CTkLabel(self.left_panel, text="Camera & Delivery", font=("Arial", 18, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
        self.video_label = ctk.CTkLabel(self.left_panel, text="Camera OFF", width=480, height=360, fg_color="#0B0F13", corner_radius=8)
        self.video_label.pack(padx=16, pady=(0, 12))

        self.metric_grid = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.metric_grid.pack(fill="x", padx=16)
        self.metric_grid.grid_columnconfigure((0, 1), weight=1)

        self.eye_value = self._metric_card("Eye Contact", "--", 0, 0)
        self.attention_value = self._metric_card("Attention", "--", 0, 1)
        self.confidence_value = self._metric_card("Confidence", "--", 1, 0)
        self.wpm_value = self._metric_card("WPM", "--", 1, 1)

        ctk.CTkLabel(self.left_panel, text="Microphone", anchor="w").pack(fill="x", padx=16, pady=(16, 4))
        self.mic_progress = ctk.CTkProgressBar(self.left_panel, progress_color="#2ECC71")
        self.mic_progress.pack(fill="x", padx=16)
        self.mic_progress.set(0)

        self.recording_hint = ctk.CTkLabel(
            self.left_panel,
            text="When the AI finishes speaking, click Start Answer. After speaking, click Submit Answer.",
            wraplength=460,
            justify="left",
            text_color="#AAB7C4",
        )
        self.recording_hint.pack(fill="x", padx=16, pady=(8, 0))


        self.controls = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.controls.pack(fill="x", padx=16, pady=16)
        self.controls.grid_columnconfigure((0, 1), weight=1)

        self.record_button = ctk.CTkButton(self.controls, text="Start Answer", height=42, command=self.start_recording, state="disabled", fg_color="#C0392B")
        self.record_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ctk.CTkButton(self.controls, text="Submit Answer", height=42, command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.replay_button = ctk.CTkButton(self.left_panel, text="Replay Current Question", command=lambda: self.speak(self.current_question), state="disabled")
        self.replay_button.pack(fill="x", padx=16, pady=(0, 12))

    def _metric_card(self, label, value, row, column):
        card = ctk.CTkFrame(self.metric_grid, fg_color="#202832", corner_radius=8)
        card.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(card, text=label, font=("Arial", 12), text_color="#AAB7C4").pack(anchor="w", padx=12, pady=(10, 0))
        value_label = ctk.CTkLabel(card, text=value, font=("Arial", 24, "bold"))
        value_label.pack(anchor="w", padx=12, pady=(0, 10))
        return value_label

    def _build_right_panel(self):
        top = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(top, text="Current Question", font=("Arial", 18, "bold")).grid(row=0, column=0, sticky="w")
        self.score_label = ctk.CTkLabel(top, text="Score: --", font=("Arial", 18, "bold"), text_color="#F1C40F")
        self.score_label.grid(row=0, column=1, sticky="e")

        self.question_box = ctk.CTkTextbox(self.right_panel, height=88, wrap="word")
        self.question_box.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.question_box.configure(state="disabled")

        self.tabs = ctk.CTkTabview(self.right_panel)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.right_panel.grid_rowconfigure(2, weight=1)

        self.answer_tab = self.tabs.add("Answer")
        self.feedback_tab = self.tabs.add("Feedback")
        self.report_tab = self.tabs.add("Report")

        self.answer_text = ctk.CTkTextbox(self.answer_tab, wrap="word")
        self.answer_text.pack(fill="both", expand=True, padx=8, pady=8)

        self.feedback_text = ctk.CTkTextbox(self.feedback_tab, wrap="word")
        self.feedback_text.pack(fill="both", expand=True, padx=8, pady=8)

        self.report_text = ctk.CTkTextbox(self.report_tab, wrap="word")
        self.report_text.pack(fill="both", expand=True, padx=8, pady=8)

    # =========================
    # STATE HELPERS
    # =========================

    def set_status(self, text, color="#D6DEE6"):
        self.status_label.configure(text=text, text_color=color)

    def set_question(self, text):
        self.current_question = text
        self.question_box.configure(state="normal")
        self.question_box.delete("1.0", "end")
        self.question_box.insert("1.0", text)
        self.question_box.configure(state="disabled")

    def append_answer(self, text):
        self.answer_text.delete("1.0", "end")
        self.answer_text.insert("1.0", text)

    def append_feedback(self, data):
        breakdown = data.get("breakdown", {})
        speech = data.get("speech_metrics", {})
        star = data.get("star", {})

        lines = [
            f"Score: {data.get('score', '--')}/10",
            f"Difficulty: {data.get('difficulty', '--')}",
            f"Question Type: {data.get('question_type', '--')}",
            "",
            "Feedback:",
            data.get("feedback", ""),
            "",
            "Breakdown:",
            f"Relevance: {breakdown.get('relevance', '--')}/10",
            f"Clarity: {breakdown.get('clarity', '--')}/10",
            f"Specificity: {breakdown.get('specificity', '--')}/10",
            f"Structure: {breakdown.get('structure', '--')}/10",
            f"Technical Depth: {breakdown.get('technical_depth', '--')}/10",
            f"Impact: {breakdown.get('impact', '--')}/10",
            "",
            "Delivery:",
            f"Confidence: {speech.get('confidence', '--')}%",
            f"Filler Words: {speech.get('filler_count', '--')}",
            f"WPM: {speech.get('wpm_estimate', '--')}",
            "",
            "STAR Check:",
            f"Situation: {star.get('situation', '--')}",
            f"Task: {star.get('task', '--')}",
            f"Action: {star.get('action', '--')}",
            f"Result: {star.get('result', '--')}",
            "",
            "Strengths:",
            *[f"- {x}" for x in data.get("strengths", [])],
            "",
            "Weaknesses:",
            *[f"- {x}" for x in data.get("weaknesses", [])],
            "",
            "Suggestions:",
            *[f"- {x}" for x in data.get("suggestions", [])],
        ]

        if self.coaching_enabled.get():
            lines.extend(["", "Improved Sample Answer:", data.get("improved_sample_answer", ""), "", "Hint:", data.get("ideal_answer_hint", "")])

        self.feedback_text.delete("1.0", "end")
        self.feedback_text.insert("1.0", "\n".join(lines))

    def _set_ready_state(self):
        self.set_status("Ready. Check backend, then start interview.", "#2ECC71")
        self.timer_label.configure(text="00:00")
        if hasattr(self, "recording_hint"):
            self.recording_hint.configure(
                text="When the AI finishes speaking, click Start Answer. After speaking, click Submit Answer."
            )

    def lock_setup(self, locked):
        state = "disabled" if locked else "normal"
        for widget in [self.server_entry, self.role_menu, self.persona_menu, self.mode_menu, self.duration_menu, self.limit_menu]:
            widget.configure(state=state)
        self.start_button.configure(state="disabled" if locked else "normal")

    # =========================
    # BACKEND
    # =========================

    def check_backend(self):
        self.server_url = self.server_entry.get().strip().rstrip("/")
        if not self.server_url or "PASTE_YOUR_NGROK_URL_HERE" in self.server_url:
            self.set_status("Paste your ngrok backend URL first.", "#E74C3C")
            return

        def worker():
            try:
                response = requests.get(f"{self.server_url}/health", timeout=8)
                if response.status_code == 200 and response.json().get("ok"):
                    self.after(0, lambda: self.set_status("Backend connected.", "#2ECC71"))
                else:
                    self.after(0, lambda: self.set_status("Backend responded, but health check failed.", "#E74C3C"))
            except Exception as exc:
                self.after(0, lambda: self.set_status(f"Backend offline: {exc}", "#E74C3C"))

        threading.Thread(target=worker, daemon=True).start()

    def fetch_opening_question(self):
        payload = self.base_payload()
        try:
            response = requests.post(f"{self.server_url}/opening_question", json=payload, timeout=20)
            if response.status_code == 200:
                return response.json().get("opening_question", "")
        except Exception:
            pass
        return f"Tell me about yourself and why you are interested in the {self.role_menu.get()} role."

    def base_payload(self):
        return {
            "history": self.history[-8:],
            "persona": self.persona_menu.get(),
            "role": self.role_menu.get(),
            "interview_mode": self.mode_menu.get(),
            "time_remaining_seconds": self.remaining_seconds(),
            "resume_context": self.resume_box.get("1.0", "end").strip(),
            "job_description": self.jd_box.get("1.0", "end").strip(),
        }

    # =========================
    # INTERVIEW FLOW
    # =========================

    def start_interview(self):
        self.server_url = self.server_entry.get().strip().rstrip("/")
        if not self.server_url or "PASTE_YOUR_NGROK_URL_HERE" in self.server_url:
            self.set_status("Paste your ngrok backend URL first.", "#E74C3C")
            return

        duration_text = self.duration_menu.get()
        self.max_duration_seconds = int(duration_text.split()[0]) * 60
        self.question_limit = int(self.limit_menu.get())
        self.turn_count = 0
        self.history = []
        self.session_started_at = datetime.now()
        self.interview_start_time = time.time()
        self.interview_state = InterviewState.AI_SPEAKING

        self.lock_setup(True)
        self.end_button.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.feedback_text.delete("1.0", "end")
        self.answer_text.delete("1.0", "end")
        self.score_label.configure(text="Score: --")
        self.start_camera()
        self._update_timer()

        self.set_status("Preparing opening question...", "#F1C40F")

        def worker():
            question = self.fetch_opening_question()
            self.after(0, lambda: self._show_ai_question(question))

        threading.Thread(target=worker, daemon=True).start()

    def _show_ai_question(self, question):
        if self.interview_state == InterviewState.COMPLETED:
            return
        self.interview_state = InterviewState.AI_SPEAKING
        self.set_question(question)
        self.replay_button.configure(state="normal")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.set_status("AI is speaking. Listen first.", "#F1C40F")
        self.speak(question)

    def _on_tts_finished(self):
        if self.interview_state == InterviewState.AI_SPEAKING:
            self.interview_state = InterviewState.READY_TO_RECORD
            self.record_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.set_status("Your turn. Start answering when ready.", "#2ECC71")
            self.recording_hint.configure(text="Ready: click Start Answer, then speak normally.")


    def start_recording(self):
        if self.interview_state != InterviewState.READY_TO_RECORD:
            return

        self.interview_state = InterviewState.USER_RECORDING
        self.is_recording = True
        self.audio_data = []
        self.latest_mic_level = 0.0
        self.answer_start_time = time.time()
        self.answer_text.delete("1.0", "end")

        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.set_status("Recording answer. Click Submit Answer when you finish speaking.", "#2ECC71")
        self.recording_hint.configure(text="Recording now. When you are done, click Submit Answer below.")

        try:
            self.audio_stream = sd.InputStream(samplerate=self.fs, channels=1, dtype="float32", callback=self._audio_callback)
            self.audio_stream.start()
        except Exception as exc:
            self.is_recording = False
            self.interview_state = InterviewState.READY_TO_RECORD
            self.record_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.set_status(f"Microphone error: {exc}", "#E74C3C")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.actual_duration = time.time() - self.answer_start_time
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        self.interview_state = InterviewState.PROCESSING
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.set_status("Answer submitted. Processing with backend...", "#F1C40F")
        self.recording_hint.configure(text="Submitted. Please wait while the AI transcribes, scores, and prepares the next question.")
        threading.Thread(target=self._submit_turn, daemon=True).start()

    def _submit_turn(self):
        if not self.audio_data:
            self.after(0, lambda: self.set_status("No audio detected. Try again.", "#E74C3C"))
            self.after(0, lambda: self.record_button.configure(state="normal"))
            self.interview_state = InterviewState.READY_TO_RECORD
            return

        os.makedirs("recordings", exist_ok=True)
        file_path = os.path.join("recordings", "answer.wav")
        audio_array = np.concatenate(self.audio_data, axis=0)
        audio_array = np.clip(audio_array, -1.0, 1.0)
        write(file_path, self.fs, np.int16(audio_array * 32767))

        payload = self.base_payload()
        payload["duration_seconds"] = self.actual_duration

        try:
            with open(file_path, "rb") as audio:
                response = requests.post(
                    f"{self.server_url}/interview_turn",
                    files={"audio": audio},
                    data={"payload": json.dumps(payload)},
                    timeout=90,
                )

            if response.status_code == 200:
                data = response.json()
            else:
                data = self.local_fallback_turn(f"Server returned {response.status_code}")
        except Exception as exc:
            data = self.local_fallback_turn(str(exc))

        self.after(0, lambda d=data: self._handle_turn_response(d))

    def _handle_turn_response(self, data):
        transcript = data.get("transcript", "")
        self.append_answer(transcript)
        self.append_feedback(data)
        self.score_label.configure(text=f"Score: {data.get('score', '--')}/10")

        speech = data.get("speech_metrics", {})
        self.confidence_value.configure(text=f"{speech.get('confidence', '--')}%")
        self.wpm_value.configure(text=str(speech.get("wpm_estimate", "--")))

        self.history.append({
            "question": self.current_question,
            "answer": transcript,
            "score": data.get("score"),
            "feedback": data.get("feedback", ""),
            "breakdown": data.get("breakdown", {}),
            "speech_metrics": speech,
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "suggestions": data.get("suggestions", []),
        })

        self.turn_count += 1
        if self.turn_count >= self.question_limit or self.remaining_seconds() <= 0:
            self.end_interview()
            return

        next_question = data.get("next_question") or FALLBACK_QUESTIONS[self.turn_count % len(FALLBACK_QUESTIONS)]
        self._show_ai_question(next_question)

    def local_fallback_turn(self, error):
        return {
            "transcript": "Could not process answer because the backend was unavailable.",
            "score": 5,
            "feedback": f"Backend issue: {error}",
            "breakdown": {"relevance": 5, "clarity": 5, "specificity": 5, "structure": 5, "technical_depth": 5, "impact": 5},
            "strengths": ["The interview flow continued instead of crashing."],
            "weaknesses": ["Backend connection needs to be restored."],
            "suggestions": ["Check Colab runtime, ngrok URL, and internet connection."],
            "speech_metrics": {"confidence": "--", "filler_count": "--", "wpm_estimate": "--", "word_count": "--"},
            "star": {},
            "next_question": FALLBACK_QUESTIONS[self.turn_count % len(FALLBACK_QUESTIONS)],
            "difficulty": "medium",
            "question_type": "fallback",
            "ideal_answer_hint": "",
            "improved_sample_answer": "",
        }

    def end_interview(self):
        if self.interview_state == InterviewState.COMPLETED:
            return
        self.interview_state = InterviewState.COMPLETED
        self.camera_running = False
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.end_button.configure(state="disabled")
        self.lock_setup(False)
        self.set_status("Generating final report...", "#F1C40F")
        threading.Thread(target=self._generate_report, daemon=True).start()

    def _generate_report(self):
        payload = self.base_payload()
        payload["history"] = self.history
        try:
            response = requests.post(f"{self.server_url}/final_report", json=payload, timeout=60)
            report = response.json() if response.status_code == 200 else self.local_report()
        except Exception:
            report = self.local_report()
        self.after(0, lambda: self._show_report(report))

    def _show_report(self, report):
        lines = [
            "FINAL INTERVIEW REPORT",
            "",
            f"Overall Score: {report.get('overall_score', '--')}/10",
            f"Hire Readiness: {report.get('hire_readiness', '--')}",
            "",
            "Summary:",
            report.get("summary", ""),
            "",
            "Top Strengths:",
            *[f"- {x}" for x in report.get("top_strengths", [])],
            "",
            "Top Weaknesses:",
            *[f"- {x}" for x in report.get("top_weaknesses", [])],
            "",
            "Communication Feedback:",
            report.get("communication_feedback", ""),
            "",
            "Technical Feedback:",
            report.get("technical_feedback", ""),
            "",
            "Recommended Practice Plan:",
            *[f"- {x}" for x in report.get("recommended_practice_plan", [])],
            "",
            "Next 7-Day Plan:",
            *[f"- {x}" for x in report.get("next_7_day_plan", [])],
        ]
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", "\n".join(lines))
        self.tabs.set("Report")
        self.save_session(report)
        self.set_status("Interview complete. Session saved locally.", "#2ECC71")

    def local_report(self):
        scores = [x.get("score") for x in self.history if isinstance(x.get("score"), int)]
        avg = int(sum(scores) / len(scores)) if scores else 5
        return {
            "overall_score": avg,
            "hire_readiness": "needs_practice" if avg < 7 else "almost_ready",
            "summary": "Session completed. Use the answer-level feedback to improve structure, specificity, and delivery.",
            "top_strengths": ["Completed the mock interview"],
            "top_weaknesses": ["Needs continued practice"],
            "communication_feedback": "Keep answers concise and structured.",
            "technical_feedback": "Add more role-specific details when relevant.",
            "recommended_practice_plan": ["Practice STAR answers", "Prepare project stories", "Reduce filler words"],
            "next_7_day_plan": ["Practice introduction", "Review projects", "Do another mock interview"],
        }

    def save_session(self, report):
        os.makedirs("sessions", exist_ok=True)
        stamp = self.session_started_at.strftime("%Y%m%d_%H%M%S") if self.session_started_at else datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("sessions", f"interview_session_{stamp}.json")
        data = {
            "started_at": stamp,
            "role": self.role_menu.get(),
            "persona": self.persona_menu.get(),
            "mode": self.mode_menu.get(),
            "history": self.history,
            "report": report,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # =========================
    # TIMER, AUDIO, TTS
    # =========================

    def remaining_seconds(self):
        if not self.interview_start_time:
            return self.max_duration_seconds
        return max(0, int(self.max_duration_seconds - (time.time() - self.interview_start_time)))

    def _update_timer(self):
        if self.interview_state == InterviewState.COMPLETED:
            return
        remaining = self.remaining_seconds()
        mins, secs = divmod(remaining, 60)
        self.timer_label.configure(text=f"{mins:02d}:{secs:02d}")
        if remaining <= 0:
            self.end_interview()
        else:
            self.after(1000, self._update_timer)

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())
            self.latest_mic_level = min(1.0, float(np.linalg.norm(indata) * 10) / 80.0)

    def _update_mic_visualizer(self):
        self.mic_progress.set(self.latest_mic_level if self.is_recording else 0)
        self.after(100, self._update_mic_visualizer)

    def speak(self, text):
        if text and self.tts_enabled.get():
            self.tts_queue.put(text)
        else:
            self.after(300, self._on_tts_finished)

    def _tts_worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception:
            engine = None
        while not self.tts_stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            self.after(0, self._on_tts_finished)

    # =========================
    # CAMERA
    # =========================

    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        threading.Thread(target=self.camera_thread, daemon=True).start()

    def camera_thread(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 24)
        frame_count = 0

        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_count % self.face_process_every_n_frames == 0:
                self._process_face_metrics(rgb)
            if frame_count % self.video_ui_every_n_frames == 0:
                display = cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_AREA)
                self.after(0, self._update_camera_ui, display.copy())
            time.sleep(0.01)

        cap.release()
        self.after(0, lambda: self.video_label.configure(image=None, text="Camera Stopped"))

    def _process_face_metrics(self, rgb):
        results = self.face_mesh.process(rgb)
        target_attention = 0.0
        target_eye_contact = 0.0
        if results.multi_face_landmarks:
            target_eye_contact = 100.0
            landmarks = results.multi_face_landmarks[0]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            nose = landmarks.landmark[1]
            eye_center_x = (left_eye.x + right_eye.x) / 2
            if (eye_center_x - 0.04) <= nose.x <= (eye_center_x + 0.04):
                target_attention = 100.0
        self.smooth_attention = target_attention * self.ema_alpha + self.smooth_attention * (1 - self.ema_alpha)
        self.smooth_eye_contact = target_eye_contact * self.ema_alpha + self.smooth_eye_contact * (1 - self.ema_alpha)

    def _update_camera_ui(self, rgb):
        pil_image = Image.fromarray(rgb)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(480, 360))
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image
        self.eye_value.configure(text=f"{int(self.smooth_eye_contact)}%")
        self.attention_value.configure(text=f"{int(self.smooth_attention)}%")

    # =========================
    # SHUTDOWN
    # =========================

    def on_closing(self):
        self.camera_running = False
        self.tts_stop_event.set()
        if self.is_recording and self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        try:
            self.face_mesh.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = InterviewAssistant()
    app.mainloop()
