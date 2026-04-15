from video_analyzer.analyzer import VideoAnalyzer


def make_analyzer():
    analyzer = object.__new__(VideoAnalyzer)
    return analyzer


def test_strip_prompt_echo_removes_leaked_template_section():
    analyzer = make_analyzer()
    response = (
        "Frame 12\n"
        "A woman sits at a table with a laptop.\n\n"
        "Your Tasks\n"
        "Step 1: Quick Scan\n"
        "Watch for key changes from previous descriptions."
    )

    cleaned = analyzer._strip_prompt_echo(response)

    assert cleaned == "Frame 12\nA woman sits at a table with a laptop."


def test_strip_prompt_echo_keeps_normal_response_unchanged():
    analyzer = make_analyzer()
    response = "Frame 3\nA title slide shows the text 'What is an AI Agent?'"

    cleaned = analyzer._strip_prompt_echo(response)

    assert cleaned == response


def test_prompt_echo_only_response_is_low_signal():
    analyzer = make_analyzer()

    assert analyzer._is_low_signal_response("Your Tasks") is True


def test_refusal_response_is_low_signal():
    analyzer = make_analyzer()

    assert analyzer._is_low_signal_response("I can’t help you with that.") is True


def test_parse_frame_response_extracts_structured_sections():
    analyzer = make_analyzer()
    response = (
        "SUMMARY:\n"
        "A presentation slide about AI agents is shown.\n\n"
        "NEW_INFORMATION:\n"
        "- Slide title changes to 'AI Agents'\n"
        "- A bullet list appears on the left\n\n"
        "CONTINUITY:\n"
        "- Track whether the bullet list expands\n"
        "- Watch for cursor movement\n\n"
        "VISIBLE_TEXT:\n"
        "- AI Agents\n"
        "- Tool use"
    )

    parsed = analyzer._parse_frame_response(response)

    assert parsed["response"] == response
    assert parsed["summary"] == "A presentation slide about AI agents is shown."
    assert parsed["signals"]["new_information"] == [
        "Slide title changes to 'AI Agents'",
        "A bullet list appears on the left",
    ]
    assert parsed["signals"]["continuity_points"] == [
        "Track whether the bullet list expands",
        "Watch for cursor movement",
    ]
    assert parsed["signals"]["visible_text"] == ["AI Agents", "Tool use"]
    assert parsed["quality"]["accepted_for_context"] is True


def test_parse_frame_response_extracts_bold_mixed_case_sections():
    analyzer = make_analyzer()
    response = (
        "**Summary**: A woman sits at a white table with an open laptop.\n\n"
        "**New Information**:\n"
        "- Her right hand moves onto the keyboard\n\n"
        "**Continuity**: The laptop remains open and centered in front of her.\n\n"
        "**Visible Text**:\n"
        "- None"
    )

    parsed = analyzer._parse_frame_response(response)

    assert parsed["summary"] == "A woman sits at a white table with an open laptop."
    assert parsed["signals"]["new_information"] == [
        "Her right hand moves onto the keyboard"
    ]
    assert parsed["signals"]["continuity_points"] == [
        "The laptop remains open and centered in front of her."
    ]
    assert parsed["signals"]["visible_text"] == []
    assert parsed["quality"]["used_structured_sections"] is True


def test_parse_frame_response_extracts_colonless_bold_headings():
    analyzer = make_analyzer()
    response = (
        "**Summary**\n"
        'A presentation slide titled "Predicting Token Sequences" is visible beside the speaker.\n\n'
        "**New Information**\n"
        "- The slide title is readable\n\n"
        "**Continuity**\n"
        "- The speaker remains seated at the desk\n\n"
        "**Visible Text**\n"
        "- Predicting Token Sequences"
    )

    parsed = analyzer._parse_frame_response(response)

    assert parsed["summary"] == (
        'A presentation slide titled "Predicting Token Sequences" is visible beside the speaker.'
    )
    assert parsed["signals"]["new_information"] == ["The slide title is readable"]
    assert parsed["signals"]["continuity_points"] == [
        "The speaker remains seated at the desk"
    ]
    assert parsed["signals"]["visible_text"] == ["Predicting Token Sequences"]
    assert parsed["quality"]["used_structured_sections"] is True


def test_parse_list_section_ignores_trailing_prose_after_none_bullet():
    analyzer = make_analyzer()
    value = (
        "- None\n\n"
        "This frame shows a woman using a laptop at a table in front of a blue wall."
    )

    parsed = analyzer._parse_list_section(value)

    assert parsed == []


def test_parse_list_section_stops_when_free_prose_starts():
    analyzer = make_analyzer()
    value = (
        '- "LLMs: Key Takeaways" (title)\n'
        '- "LLMs predict the next token based on probabilities" (text)\n\n'
        "This frame should track the slide content in the next shot."
    )

    parsed = analyzer._parse_list_section(value)

    assert parsed == [
        '"LLMs: Key Takeaways" (title)',
        '"LLMs predict the next token based on probabilities" (text)',
    ]


def test_parse_frame_response_falls_back_to_cleaned_summary_without_sections():
    analyzer = make_analyzer()

    parsed = analyzer._parse_frame_response(
        "A static lecture slide remains on screen with a new highlighted bullet point."
    )

    assert parsed["summary"] == (
        "A static lecture slide remains on screen with a new highlighted bullet point."
    )
    assert parsed["quality"]["used_structured_sections"] is False
    assert parsed["quality"]["accepted_for_context"] is True


def test_repetitive_response_is_low_signal():
    analyzer = make_analyzer()
    response = "The slide the slide the slide the slide the slide the slide the slide"

    assert analyzer._is_repetitive_response(response) is True
    assert analyzer._is_low_signal_response(response) is True


def test_format_previous_analyses_uses_summary_and_continuity_only():
    analyzer = make_analyzer()
    analyzer.previous_analyses = [
        {
            "frame_number": 12,
            "summary": "A title slide remains on screen.",
            "signals": {
                "continuity_points": ["Watch for title change", "Track cursor"]
            },
            "quality": {"accepted_for_context": True},
            "response": "raw text should not be used",
        },
        {
            "frame_number": 13,
            "summary": "",
            "signals": {"continuity_points": []},
            "quality": {"accepted_for_context": False},
            "response": "I can't help you with that.",
        },
    ]

    formatted = analyzer._format_previous_analyses()

    assert "Frame 12" in formatted
    assert "Summary: A title slide remains on screen." in formatted
    assert "Track next: Watch for title change; Track cursor" in formatted
    assert "Frame 13" not in formatted
    assert "raw text should not be used" not in formatted


def test_reconstruct_video_prefers_summary_and_structured_signals():
    analyzer = make_analyzer()

    class StubClient:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {"response": "Final description"}

    class FrameStub:
        def __init__(self, timestamp):
            self.timestamp = timestamp

    analyzer.client = StubClient()
    analyzer.model = "test-model"
    analyzer.temperature = 0.2
    analyzer.user_prompt = ""
    analyzer.video_prompt = (
        "NOTES:\n{FRAME_NOTES}\n\nFIRST:\n{FIRST_FRAME}\n\nTRANSCRIPT:\n{TRANSCRIPT}"
    )

    frame_analyses = [
        {
            "summary": "Slide introducing AI agents.",
            "signals": {
                "new_information": ["Title changes to AI Agents"],
                "visible_text": ["AI Agents"],
            },
            "quality": {"accepted_for_context": True},
            "response": "Long raw response that should not be preferred.",
        },
        {
            "summary": "",
            "signals": {"new_information": [], "visible_text": []},
            "quality": {"accepted_for_context": False},
            "response": "I can't help you with that.",
        },
    ]
    frames = [FrameStub(0.0), FrameStub(1.5)]

    result = analyzer.reconstruct_video(frame_analyses, frames, transcript=None)

    assert result == {"response": "Final description"}
    prompt = analyzer.client.calls[0]["prompt"]
    assert "Slide introducing AI agents." in prompt
    assert "New information: Title changes to AI Agents" in prompt
    assert "Visible text: AI Agents" in prompt
    assert "I can't help you with that." not in prompt


def test_strip_summary_prompt_echo_removes_reconstruction_template():
    analyzer = make_analyzer()
    response = (
        "To synthesize a cohesive video summary, follow these steps:\n\n"
        "1. Review Process\n"
        "2. Synthesis Guidelines\n\n"
        "Format Your Summary As:\n"
        "VIDEO SUMMARY"
    )

    cleaned = analyzer._strip_summary_prompt_echo(response)

    assert (
        cleaned
        == "To synthesize a cohesive video summary, follow these steps: 1. Review Process 2. Synthesis Guidelines"
    )


def test_reconstruct_video_cleans_template_echoed_summary_response():
    analyzer = make_analyzer()

    class StubClient:
        def generate(self, **kwargs):
            return {
                "response": (
                    "Video Summary Instructions\n"
                    "Step 1: Review Process\n"
                    "Format Your Summary As:\n"
                    "VIDEO SUMMARY"
                )
            }

    class FrameStub:
        def __init__(self, timestamp):
            self.timestamp = timestamp

    analyzer.client = StubClient()
    analyzer.model = "test-model"
    analyzer.temperature = 0.2
    analyzer.user_prompt = ""
    analyzer.video_prompt = (
        "NOTES:\n{FRAME_NOTES}\n\nFIRST:\n{FIRST_FRAME}\n\nTRANSCRIPT:\n{TRANSCRIPT}"
    )

    frame_analyses = [
        {
            "summary": "A lecturer explains how token probabilities work.",
            "signals": {"new_information": [], "visible_text": []},
            "quality": {"accepted_for_context": True},
            "response": "raw",
        }
    ]
    frames = [FrameStub(0.0)]

    result = analyzer.reconstruct_video(frame_analyses, frames, transcript=None)

    assert result["response"] == ""
