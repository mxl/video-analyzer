from typing import List, Dict, Any, Optional
import logging
import re
from .clients.llm_client import LLMClient
from .prompt import PromptLoader
from .frame import Frame
from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)


LOW_SIGNAL_PREFIXES = (
    "i can't help you with that",
    "i can’t help you with that",
    "error analyzing frame",
)
PROMPT_ECHO_MARKERS = (
    "your tasks",
    "step 1: quick scan",
    "step 2: document your frame",
    "format your notes as:",
    "frame description instructions",
    "writing guidelines",
)
SUMMARY_PROMPT_ECHO_MARKERS = (
    "video summary instructions",
    "your task",
    "step 1: review process",
    "step 2: synthesis guidelines",
    "writing style guidelines",
    "quality check",
    "format your summary as:",
    "write one final video summary in plain text",
)
MAX_PREVIOUS_ANALYSES = 12
MAX_FRAME_NOTES_CHARS = 20000
MAX_TRANSCRIPT_CHARS = 12000
MAX_CONTEXT_SUMMARY_CHARS = 240
FRAME_SECTION_ALIASES = {
    "summary": "SUMMARY",
    "new information": "NEW_INFORMATION",
    "new_information": "NEW_INFORMATION",
    "continuity": "CONTINUITY",
    "visible text": "VISIBLE_TEXT",
    "visible_text": "VISIBLE_TEXT",
}


class VideoAnalyzer:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_loader: PromptLoader,
        temperature: float,
        user_prompt: str = "",
    ):
        """Initialize the VideoAnalyzer.

        Args:
            client: LLM client for making API calls
            model: Name of the model to use
            prompt_loader: Loader for prompt templates
            user_prompt: Optional user question about the video that will be injected into frame analysis
                        and video description prompts using the {prompt} token
        """
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt  # Store user's question about the video
        self._load_prompts()
        self.previous_analyses = []

    def _format_user_prompt(self) -> str:
        """Format the user's prompt by adding prefix if not empty."""
        if self.user_prompt:
            return f"I want to know {self.user_prompt}"
        return ""

    def _compact_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace to keep notes compact and comparable."""
        return " ".join(text.split())

    def _truncate_inline(self, text: str, max_chars: int) -> str:
        """Trim inline text without preserving both ends."""
        compact = self._compact_whitespace(text)
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3].rstrip()}..."

    def _dedupe_items(self, items: List[str]) -> List[str]:
        """Deduplicate short note items while preserving order."""
        unique_items = []
        seen = set()
        for item in items:
            normalized = self._compact_whitespace(item).strip(" -")
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(normalized)
        return unique_items

    def _strip_markdown_fences(self, response: str) -> str:
        """Unwrap a single fenced block when the model returns one."""
        stripped = response.strip()
        match = re.fullmatch(r"```(?:[\w+-]+)?\s*([\s\S]*?)\s*```", stripped)
        if match:
            return match.group(1).strip()
        return stripped

    def _is_repetitive_response(self, response: str) -> bool:
        """Return True when the model drifts into repeated-token or repeated-line noise."""
        normalized = self._compact_whitespace(response.lower())
        if len(normalized) < 60:
            return False

        tokens = re.findall(r"\b[\w']+\b", normalized)
        if len(tokens) >= 12:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.33:
                return True

            max_run = 1
            current_run = 1
            for previous, current in zip(tokens, tokens[1:]):
                if current == previous:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            if max_run >= 4:
                return True

        repeated_lines = {}
        for line in response.splitlines():
            normalized_line = self._compact_whitespace(line.lower())
            if len(normalized_line) < 20:
                continue
            repeated_lines[normalized_line] = repeated_lines.get(normalized_line, 0) + 1
            if repeated_lines[normalized_line] >= 3:
                return True

        return False

    def _is_low_signal_response(self, response: str) -> bool:
        """Return True for refusal/error responses that should not poison later prompts."""
        normalized = self._compact_whitespace(response.strip().lower())
        if not normalized:
            return True
        if any(normalized.startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES):
            return True
        if normalized in PROMPT_ECHO_MARKERS:
            return True
        return self._is_repetitive_response(response)

    def _strip_prompt_echo(self, response: str) -> str:
        """Remove prompt template text accidentally echoed by the model."""
        lowered = response.lower()
        cut_points = [
            lowered.find(marker)
            for marker in PROMPT_ECHO_MARKERS
            if lowered.find(marker) != -1
        ]
        if not cut_points:
            return response.strip()

        cleaned = response[: min(cut_points)].rstrip()
        return cleaned.strip()

    def _strip_summary_prompt_echo(self, response: str) -> str:
        """Remove leaked reconstruction-prompt instructions from a summary response."""
        lowered = response.lower()
        cut_points = [
            lowered.find(marker)
            for marker in SUMMARY_PROMPT_ECHO_MARKERS
            if lowered.find(marker) != -1
        ]
        if cut_points:
            response = response[: min(cut_points)].rstrip()

        fenced = self._strip_markdown_fences(response)
        compact = self._compact_whitespace(fenced)
        if compact.lower().startswith("video summary"):
            compact = re.sub(r"^video summary\s*:?[ \t]*", "", compact, flags=re.I)
        return compact.strip()

    def _is_summary_prompt_echo(self, response: str) -> bool:
        """Return True when the summary response mostly repeats reconstruction instructions."""
        normalized = self._compact_whitespace(response.strip().lower())
        if not normalized:
            return True
        return any(marker in normalized for marker in SUMMARY_PROMPT_ECHO_MARKERS)

    def _extract_sections(self, response: str) -> Dict[str, str]:
        """Extract labeled sections from a structured frame-analysis response."""
        sections = {}
        current_section = None
        current_lines: List[str] = []

        def flush_current_section() -> None:
            nonlocal current_section, current_lines
            if current_section is None:
                return
            body = "\n".join(current_lines).strip()
            if body:
                sections[current_section] = body
            elif current_section not in sections:
                sections[current_section] = ""

        for line in response.splitlines():
            heading = self._match_section_heading(line)
            if heading is not None:
                flush_current_section()
                current_section, inline_content = heading
                current_lines = [inline_content] if inline_content else []
                continue

            if current_section is not None:
                current_lines.append(line.rstrip())

        flush_current_section()
        return sections

    def _match_section_heading(self, line: str) -> Optional[tuple[str, str]]:
        """Return the canonical section name and inline content for one heading line."""
        stripped = line.strip()
        if not stripped:
            return None

        for alias, canonical_name in FRAME_SECTION_ALIASES.items():
            label_variants = {
                alias,
                alias.replace("_", " "),
                alias.replace(" ", "_"),
            }
            for label in label_variants:
                patterns = (
                    rf"^\*\*{re.escape(label)}\*\*:\s*(.*)$",
                    rf"^\*\*{re.escape(label)}:\*\*\s*(.*)$",
                    rf"^\*\*{re.escape(label)}\*\*\s*(.*)$",
                    rf"^{re.escape(label)}:\s*(.*)$",
                    rf"^{re.escape(label)}\s*(.*)$",
                )
                for pattern in patterns:
                    match = re.match(pattern, stripped, re.I)
                    if match:
                        inline_content = match.group(1).strip()
                        return canonical_name, inline_content

        return None

    def _parse_list_section(self, value: str) -> List[str]:
        """Parse bullet-like sections into a normalized list of items."""
        if not value:
            return []

        non_empty_lines = [
            self._compact_whitespace(line)
            for line in value.splitlines()
            if self._compact_whitespace(line)
        ]
        if not non_empty_lines:
            return []

        bullet_pattern = re.compile(r"^[\-*•]+\s*(.*)$")
        bullet_start = next(
            (
                index
                for index, line in enumerate(non_empty_lines)
                if bullet_pattern.match(line)
            ),
            None,
        )

        if bullet_start is None:
            candidate = self._compact_whitespace(" ".join(non_empty_lines))
            if not candidate or candidate.lower() == "none":
                return []
            return self._dedupe_items([candidate])

        items = []
        for line in non_empty_lines[bullet_start:]:
            match = bullet_pattern.match(line)
            if not match:
                break

            candidate = self._compact_whitespace(match.group(1))
            if not candidate:
                continue
            if candidate.lower() == "none":
                return []
            items.append(candidate)

        return self._dedupe_items(items)

    def _build_summary(self, cleaned_response: str, sections: Dict[str, str]) -> str:
        """Create a compact summary that is safe to reuse in later prompts."""
        summary = self._compact_whitespace(sections.get("SUMMARY", ""))
        if summary and summary.lower() != "none":
            return self._truncate_inline(summary, MAX_CONTEXT_SUMMARY_CHARS)

        fallback_parts = []
        new_information = self._parse_list_section(sections.get("NEW_INFORMATION", ""))
        if new_information:
            fallback_parts.append(f"New: {'; '.join(new_information[:2])}")

        visible_text = self._parse_list_section(sections.get("VISIBLE_TEXT", ""))
        if visible_text:
            fallback_parts.append(f"Visible text: {'; '.join(visible_text[:2])}")

        if fallback_parts:
            return self._truncate_inline(
                " ".join(fallback_parts), MAX_CONTEXT_SUMMARY_CHARS
            )

        if self._is_low_signal_response(cleaned_response):
            return ""
        return self._truncate_inline(cleaned_response, MAX_CONTEXT_SUMMARY_CHARS)

    def _parse_frame_response(self, raw_response: str) -> Dict[str, Any]:
        """Convert a raw frame response into cleaned text, summary, and structured signals."""
        cleaned_response = self._strip_markdown_fences(
            self._strip_prompt_echo(raw_response)
        )
        sections = self._extract_sections(cleaned_response)
        signals = {
            "new_information": self._parse_list_section(
                sections.get("NEW_INFORMATION", "")
            ),
            "continuity_points": self._parse_list_section(
                sections.get("CONTINUITY", "")
            ),
            "visible_text": self._parse_list_section(sections.get("VISIBLE_TEXT", "")),
        }
        quality = {
            "is_low_signal": self._is_low_signal_response(cleaned_response),
            "has_prompt_echo": cleaned_response != raw_response.strip(),
            "is_repetitive": self._is_repetitive_response(cleaned_response),
            "used_structured_sections": bool(sections),
        }
        summary = self._build_summary(cleaned_response, sections)
        quality["accepted_for_context"] = bool(summary) and not quality["is_low_signal"]

        return {
            "response": cleaned_response,
            "cleaned_response": cleaned_response,
            "summary": summary,
            "signals": signals,
            "quality": quality,
        }

    def _best_frame_note(self, analysis: Dict[str, Any]) -> str:
        """Return the best compact note available for a frame."""
        summary = self._compact_whitespace(analysis.get("summary", ""))
        if summary:
            return summary

        candidate = analysis.get("cleaned_response") or analysis.get("response", "")
        candidate = self._compact_whitespace(candidate)
        if candidate and not self._is_low_signal_response(candidate):
            return self._truncate_inline(candidate, MAX_CONTEXT_SUMMARY_CHARS)
        return ""

    def _accepted_for_context(self, analysis: Dict[str, Any]) -> bool:
        """Return True when a frame analysis is safe to reuse as context."""
        quality = analysis.get("quality", {})
        if "accepted_for_context" in quality:
            return bool(quality["accepted_for_context"])
        return bool(self._best_frame_note(analysis))

    def _format_context_entry(self, analysis: Dict[str, Any]) -> str:
        """Format one prior frame analysis into a compact context entry."""
        summary = self._best_frame_note(analysis)
        if not summary:
            return ""

        frame_number = analysis.get("frame_number")
        header = f"Frame {frame_number}" if frame_number is not None else "Frame"
        lines = [header, f"Summary: {summary}"]

        continuity_points = (analysis.get("signals") or {}).get(
            "continuity_points"
        ) or []
        if continuity_points:
            lines.append(f"Track next: {'; '.join(continuity_points[:3])}")

        return "\n".join(lines)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Trim large prompt sections while preserving both beginning and end context."""
        if len(text) <= max_chars:
            return text

        marker = "\n\n...[truncated]...\n\n"
        head_chars = max_chars // 3
        tail_chars = max_chars - head_chars - len(marker)
        return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"

    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # Frame Analysis prompt
        self.video_prompt = self.prompt_loader.get_by_index(
            1
        )  # Video Reconstruction prompt

    def _format_previous_analyses(self) -> str:
        """Format previous frame analyses for inclusion in prompt."""
        if not self.previous_analyses:
            return ""

        formatted_analyses = []
        useful_analyses = [
            analysis
            for analysis in self.previous_analyses
            if self._accepted_for_context(analysis)
        ][-MAX_PREVIOUS_ANALYSES:]

        for analysis in useful_analyses:
            formatted_analysis = self._format_context_entry(analysis)
            if formatted_analysis:
                formatted_analyses.append(formatted_analysis)

        return "\n".join(formatted_analyses)

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        # Replace {PREVIOUS_FRAMES} token with formatted previous analyses
        # Replace tokens in the prompt template
        prompt = self.frame_prompt.replace(
            "{PREVIOUS_FRAMES}", self._format_previous_analyses()
        )
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = f"{prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."

        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                temperature=self.temperature,
                num_predict=300,
            )
            logger.debug(f"Successfully analyzed frame {frame.number}")

            # Store the analysis for future frames
            analysis_result = {k: v for k, v in response.items() if k != "context"}
            parsed_response = self._parse_frame_response(
                analysis_result.get("response", "")
            )
            analysis_result.update(parsed_response)
            analysis_result["timestamp"] = frame.timestamp
            analysis_result["frame_number"] = frame.number
            self.previous_analyses.append(analysis_result)

            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            error_result = {
                "response": f"Error analyzing frame {frame.number}: {str(e)}",
                "cleaned_response": f"Error analyzing frame {frame.number}: {str(e)}",
                "summary": "",
                "signals": {
                    "new_information": [],
                    "continuity_points": [],
                    "visible_text": [],
                },
                "quality": {
                    "is_low_signal": True,
                    "has_prompt_echo": False,
                    "is_repetitive": False,
                    "used_structured_sections": False,
                    "accepted_for_context": False,
                },
                "timestamp": frame.timestamp,
                "frame_number": frame.number,
            }
            self.previous_analyses.append(error_result)
            return error_result

    def reconstruct_video(
        self,
        frame_analyses: List[Dict[str, Any]],
        frames: List[Frame],
        transcript: Optional[AudioTranscript] = None,
    ) -> Dict[str, Any]:
        """Reconstruct video description from frame analyses and transcript."""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            if not self._accepted_for_context(analysis):
                continue

            note_lines = [self._best_frame_note(analysis)]
            signals = analysis.get("signals") or {}

            new_information = signals.get("new_information") or []
            if new_information:
                note_lines.append(f"New information: {'; '.join(new_information[:3])}")

            visible_text = signals.get("visible_text") or []
            if visible_text:
                note_lines.append(f"Visible text: {'; '.join(visible_text[:3])}")

            frame_note = f"Frame {i} ({frame.timestamp:.2f}s):\n" + "\n".join(
                line for line in note_lines if line
            )
            frame_notes.append(frame_note)

        analysis_text = self._truncate_text(
            "\n\n".join(frame_notes),
            MAX_FRAME_NOTES_CHARS,
        )

        # Get first frame analysis
        first_frame_text = ""
        for analysis in frame_analyses:
            candidate = self._best_frame_note(analysis)
            if candidate:
                first_frame_text = candidate
                break

        # Include transcript information if available
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = self._truncate_text(
                transcript.text,
                MAX_TRANSCRIPT_CHARS,
            )

        # Replace tokens in the prompt template
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)

        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=1000,
            )
            logger.info("Successfully reconstructed video description")
            result = {k: v for k, v in response.items() if k != "context"}
            cleaned_response = self._strip_summary_prompt_echo(
                result.get("response", "")
            )
            if self._is_summary_prompt_echo(cleaned_response):
                logger.warning(
                    "Summary response appears to echo prompt instructions; returning cleaned fallback text"
                )
            result["response"] = cleaned_response
            return result
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}
