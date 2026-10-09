"""
Unit tests for AI Interview Assistant data structures and state machine.
"""

from enum import Enum

import pytest


def test_interview_state_enum():
    from main import InterviewState

    assert hasattr(InterviewState, "IDLE")
    assert hasattr(InterviewState, "CONNECTING")
    assert hasattr(InterviewState, "QUESTION_READY")
    assert hasattr(InterviewState, "SPEAKING_QUESTION")
    assert hasattr(InterviewState, "RECORDING")
    assert hasattr(InterviewState, "EVALUATING")
    assert hasattr(InterviewState, "FEEDBACK_READY")
    assert hasattr(InterviewState, "COMPLETED")


def test_fallback_questions():
    from main import FALLBACK_QUESTIONS, INTERVIEW_MODES, PERSONAS, ROLES

    assert len(FALLBACK_QUESTIONS) >= 4
    assert "Software Engineer Intern" in ROLES
    assert "Strict Technical Lead" in PERSONAS
    assert "Behavioral Round" in INTERVIEW_MODES


if __name__ == "__main__":
    test_interview_state_enum()
    test_fallback_questions()
    print("All unit tests passed successfully!")
