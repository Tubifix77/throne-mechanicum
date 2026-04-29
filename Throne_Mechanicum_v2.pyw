"""
Throne Mechanicum v2
- /api/chat with proper message history
- Validator sub-agent: catches malformed responses, retries up to 5 times
- Protocol tags never shown to user
- Model selector, tok/s, timestamps, clear session, live counts
"""

import sys
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget, QListWidget,
    QProgressBar, QSplitter, QComboBox, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# ==========================================
# 1. DATA MODELS
# ==========================================

class TrustLevel(Enum):
    PROPOSE      = "propose"
    ACT_AND_SHOW = "act_and_show"
    ACT_SILENT   = "act_silent"

@dataclass
class ParseResult:
    user_response: str
    warnings:   list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

@dataclass
class Config:
    data_dir:           Path = Path(__file__).parent / "throne_data"
    ollama_url:         str  = "http://localhost:11434/api/chat"
    model:              str  = "gemma3:12b"
    validator_model:    str  = "gemma3:12b"   # sub-agent model
    max_exchanges:      int  = 15
    request_timeout:    int  = 120
    max_context_tokens: int  = 8192
    max_validator_tries:int  = 5

AVAILABLE_MODELS = [
    "gemma3:12b",
    "qwen2.5-coder:7b",
    "deepseek-r1:8b",
    "phi4:14b",
]

# ==========================================
# 2. THEME
# ==========================================

@dataclass(frozen=True)
class Theme:
    bg_primary:    str = "#0d1117"
    bg_secondary:  str = "#161b22"
    bg_input:      str = "#21262d"
    bg_hover:      str = "#30363d"
    bg_panel:      str = "#1c2128"
    text_primary:  str = "#e6edf3"
    text_secondary:str = "#8b949e"
    accent_gold:   str = "#d29922"
    accent_cyan:   str = "#58a6ff"
    accent_green:  str = "#3fb950"
    accent_red:    str = "#f85149"
    accent_orange: str = "#e3b341"
    border_default:str = "#30363d"
    border_focus:  str = "#58a6ff"

def build_stylesheet(t: Theme) -> str:
    return f"""
    QMainWindow, QWidget {{ background-color: {t.bg_primary}; color: {t.text_primary}; }}
    QTextEdit {{
        background-color: {t.bg_secondary}; color: {t.text_primary};
        border: 1px solid {t.border_default}; border-radius: 4px;
        font-family: 'Segoe UI', sans-serif; font-size: 13px;
    }}
    QLineEdit {{
        background-color: {t.bg_input}; color: {t.text_primary};
        border: 1px solid {t.border_default}; padding: 6px; border-radius: 4px;
    }}
    QLineEdit:focus {{ border: 1px solid {t.border_focus}; }}
    QPushButton {{
        background-color: {t.bg_input}; color: {t.text_primary};
        border: 1px solid {t.border_default}; padding: 6px 16px; border-radius: 4px;
    }}
    QPushButton:hover {{ background-color: {t.bg_hover}; }}
    QPushButton#sendBtn {{
        background-color: {t.accent_cyan}; color: {t.bg_primary};
        border: none; font-weight: bold;
    }}
    QPushButton#sendBtn:hover {{ background-color: #79b8ff; }}
    QPushButton#clearBtn {{ color: {t.accent_red}; border: 1px solid {t.accent_red}; }}
    QComboBox {{
        background-color: {t.bg_input}; color: {t.text_primary};
        border: 1px solid {t.border_default}; padding: 4px 8px; border-radius: 4px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t.bg_secondary}; color: {t.text_primary};
        selection-background-color: {t.bg_hover};
    }}
    #titleLabel {{ color: {t.accent_gold}; font-size: 16px; font-weight: bold; letter-spacing: 2px; }}
    #exchangeLabel {{ color: {t.accent_cyan}; font-size: 12px; }}
    #thinkingLabel {{ color: {t.accent_gold}; font-size: 11px; font-style: italic; }}
    QProgressBar {{
        border: 1px solid {t.border_default}; border-radius: 3px;
        text-align: center; color: {t.text_primary}; max-height: 14px;
    }}
    QProgressBar::chunk {{ background-color: {t.accent_cyan}; border-radius: 3px; }}
    QTabWidget::pane {{ border: 1px solid {t.border_default}; }}
    QTabBar::tab {{
        background: {t.bg_panel}; color: {t.text_secondary};
        padding: 6px 12px; border-radius: 4px 4px 0 0;
    }}
    QTabBar::tab:selected {{
        background: {t.bg_secondary}; color: {t.text_primary};
        border-bottom: 2px solid {t.accent_gold};
    }}
    QListWidget {{ background-color: {t.bg_panel}; color: {t.text_primary}; border: none; font-size: 12px; }}
    QStatusBar {{ background-color: {t.bg_panel}; color: {t.text_secondary}; font-size: 11px; }}
    """

# ==========================================
# 3. MEMORY MANAGER
# ==========================================

class MemoryManager:
    def __init__(self, config: Config):
        self.config = config
        self.config.data_dir.mkdir(exist_ok=True)
        (self.config.data_dir / "sessions").mkdir(exist_ok=True)
        self._init_files()

    def _init_files(self):
        for fname, default in [
            ("observations.json", []), ("ratified.json", []),
            ("conventions.json", []),
        ]:
            p = self.config.data_dir / fname
            if not p.exists():
                p.write_text(json.dumps(default), encoding="utf-8")

    def _load(self, filename: str):
        try:
            return json.loads((self.config.data_dir / filename).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {} if filename == "handoff.json" else []

    def _save(self, filename: str, data):
        (self.config.data_dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_memory_block(self) -> str:
        return (
            "OBSERVATIONS: "      + json.dumps(self._load("observations.json")) + "\n"
            "RATIFIED_PATTERNS: " + json.dumps(self._load("ratified.json"))     + "\n"
            "CONVENTIONS: "       + json.dumps(self._load("conventions.json"))  + "\n"
        )

    def add_observation(self, obs_id: str, text: str):
        obs = self._load("observations.json")
        rat = self._load("ratified.json")
        for o in obs:
            if o["id"] == obs_id:
                o["count"] = o.get("count", 1) + 1
                if o["count"] >= 3:
                    rat.append({"id": o["id"], "pattern": o["text"], "trust": "propose", "successes": 0})
                    obs.remove(o)
                self._save("observations.json", obs)
                self._save("ratified.json", rat)
                return
        obs.append({"id": obs_id, "text": text, "count": 1})
        self._save("observations.json", obs)

    def add_convention(self, conv_id: str, text: str):
        convs = self._load("conventions.json")
        for c in convs:
            if c["id"] == conv_id:
                c["convention"] = text  # update if exists
                self._save("conventions.json", convs)
                return
        convs.append({"id": conv_id, "convention": text})
        self._save("conventions.json", convs)

    def counts(self) -> dict:
        return {
            "observations": len(self._load("observations.json")),
            "ratified":     len(self._load("ratified.json")),
            "conventions":  len(self._load("conventions.json")),
        }

# ==========================================
# 4. VALIDATOR SUB-AGENT
# ==========================================

REQUIRED_TAGS  = ["[COGNITION]", "[/COGNITION]", "[LOGIC]", "[/LOGIC]", "[RESPONSE]", "[/RESPONSE]"]
OBS_NEW_RE     = re.compile(r'\[OBS_NEW\].*?\[/OBS_NEW\]', re.DOTALL)
LOGIC_BLOCK_RE = re.compile(r'\[LOGIC\](.*?)\[/LOGIC\]',   re.DOTALL)

def find_violations(text: str) -> list[str]:
    violations = []
    for tag in REQUIRED_TAGS:
        if tag not in text:
            violations.append(f"Missing tag: {tag}")
    # Check [OBS_NEW] only inside [LOGIC]
    logic_m = LOGIC_BLOCK_RE.search(text)
    text_outside_logic = text
    if logic_m:
        text_outside_logic = text[:logic_m.start(1)] + text[logic_m.end(1):]
    stray = OBS_NEW_RE.findall(text_outside_logic)
    if stray:
        violations.append(f"[OBS_NEW] found outside [LOGIC] ({len(stray)} instance(s))")
    return violations

def ollama_single(url: str, model: str, prompt: str, timeout: int) -> str:
    """Fire-and-forget single call to Ollama. No conversation history."""
    payload = json.dumps({
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }).encode()
    req = requests.post(url, data=payload,
                        headers={"Content-Type": "application/json"}, timeout=timeout)
    req.raise_for_status()
    return req.json()["message"]["content"]

def validate_and_correct(raw: str, config: Config) -> tuple[str, int, bool]:
    """
    Validator sub-agent loop.
    Returns (corrected_response, attempts_used, success)
    Sees only the single response — no conversation history.
    """
    current = raw

    for attempt in range(1, config.max_validator_tries + 1):
        violations = find_violations(current)
        if not violations:
            return current, attempt - 1, True

        if attempt == config.max_validator_tries:
            # Last chance — ask model to flag its own output
            flag_prompt = (
                f"Your response still has formatting errors after multiple attempts:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n\nYour response:\n" + current
                + "\n\nDo your best to rewrite it with the three-tag structure. "
                "Add this warning at the start of [RESPONSE]: "
                "'⚠ FORMATTING INCOMPLETE.'"
            )
            flagged = ollama_single(config.ollama_url, config.validator_model, flag_prompt, config.request_timeout)
            return flagged, attempt, False

        correction_prompt = (
            f"Your response has formatting errors (attempt {attempt}/{config.max_validator_tries}):\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nYour response was:\n" + current
            + "\n\nRewrite it using EXACTLY this structure:\n"
            "[COGNITION]reasoning[/COGNITION]\n"
            "[LOGIC]any [OBS_NEW] tags go HERE and nowhere else[/LOGIC]\n"
            "[RESPONSE]reply to user[/RESPONSE]\n\n"
            "Output ONLY the corrected response, nothing else."
        )
        current = ollama_single(
            config.ollama_url, config.validator_model, correction_prompt, config.request_timeout)

    return current, config.max_validator_tries, False

# ==========================================
# 5. TEMPLATE PARSER
# ==========================================

class TemplateParser:
    def __init__(self):
        self._response = re.compile(r'\[RESPONSE\](.*?)\[/RESPONSE\]', re.DOTALL)
        self._logic    = re.compile(r'\[LOGIC\](.*?)\[/LOGIC\]',       re.DOTALL)
        self._obs_new  = re.compile(
            r'\[OBS_NEW\]\s*id:\s*(.*?)\s*observation:\s*(.*?)\s*\[/OBS_NEW\]', re.DOTALL)
        self._conv     = re.compile(
            r'\[CONVENTION\]\s*id:\s*(.*?)\s*convention:\s*(.*?)\s*\[/CONVENTION\]', re.DOTALL)

    def parse_and_execute(self, raw: str, memory: MemoryManager) -> tuple[ParseResult, int]:
        result = ParseResult(user_response="")

        m = self._response.search(raw)
        result.user_response = m.group(1).strip() if m else raw.strip()

        logic_m = self._logic.search(raw)
        if logic_m:
            logic = logic_m.group(1)
            for om in self._obs_new.finditer(logic):
                memory.add_observation(om.group(1).strip(), om.group(2).strip())
                result.operations.append(f"Observation: {om.group(1).strip()}")
            for cm in self._conv.finditer(logic):
                memory.add_convention(cm.group(1).strip(), cm.group(2).strip())
                result.operations.append(f"Convention: {cm.group(1).strip()}")

        return result, 0

# ==========================================
# 6. BACKEND WORKER (main + validator loop)
# ==========================================

class BackendWorker(QThread):
    response_ready    = pyqtSignal(str, float, int, bool)
    # (validated_response, tok_per_sec, validator_attempts, validator_success)
    thinking_update   = pyqtSignal(str)
    error_occurred    = pyqtSignal(str)

    def __init__(self, config: Config, messages: list):
        super().__init__()
        self.config   = config
        self.messages = messages
        self._cancel  = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t_start = time.time()
        try:
            # Step 1: Main model generates response
            self.thinking_update.emit("⚙  Thinking…")
            payload = {
                "model":    self.config.model,
                "messages": self.messages,
                "stream":   True,
            }
            r = requests.post(
                self.config.ollama_url, json=payload,
                stream=True, timeout=self.config.request_timeout
            )
            r.raise_for_status()

            full_text   = ""
            token_count = 0
            for line in r.iter_lines():
                if self._cancel: break
                if not line: continue
                chunk = json.loads(line)
                full_text   += chunk.get("message", {}).get("content", "")
                token_count += 1
                if chunk.get("done"): break

            elapsed     = max(time.time() - t_start, 0.001)
            tok_per_sec = token_count / elapsed

            # Step 2: Validator sub-agent checks and corrects
            violations = find_violations(full_text)
            val_attempts = 0
            val_success  = True

            if violations:
                self.thinking_update.emit(f"🔍  Validator: fixing {len(violations)} issue(s)…")
                full_text, val_attempts, val_success = validate_and_correct(full_text, self.config)

            self.response_ready.emit(full_text, tok_per_sec, val_attempts, val_success)

        except Exception as e:
            self.error_occurred.emit(str(e))

# ==========================================
# 7. MAIN WINDOW
# ==========================================

class ThroneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config  = Config()
        self.memory  = MemoryManager(self.config)
        self.parser  = TemplateParser()
        self.theme   = Theme()
        self._exchange_count = 0
        self._messages: list = []
        self._worker: BackendWorker | None = None

        self.setWindowTitle("Throne Mechanicum v2")
        self.resize(1280, 860)
        self.setStyleSheet(build_stylesheet(self.theme))
        self._build_ui()
        self._start_session()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel("THRONE MECHANICUM")
        title.setObjectName("titleLabel")
        self.exchange_label = QLabel("Exchange: 0 / 14")
        self.exchange_label.setObjectName("exchangeLabel")
        self.model_combo = QComboBox()
        for m in AVAILABLE_MODELS:
            self.model_combo.addItem(m)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(QLabel("Model:"))
        hdr.addWidget(self.model_combo)
        hdr.addWidget(self.exchange_label)
        ll.addLayout(hdr)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Neural Load: %p%")
        self.progress_bar.setValue(0)
        ll.addWidget(self.progress_bar)

        self.thinking_label = QLabel("")
        self.thinking_label.setObjectName("thinkingLabel")
        ll.addWidget(self.thinking_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        ll.addWidget(self.chat_display)

        inp = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Awaiting input…")
        self.send_btn  = QPushButton("Transmit")
        self.send_btn.setObjectName("sendBtn")
        self.clear_btn = QPushButton("Clear Session")
        self.clear_btn.setObjectName("clearBtn")
        inp.addWidget(self.input_field)
        inp.addWidget(self.send_btn)
        inp.addWidget(self.clear_btn)
        ll.addLayout(inp)

        self.send_btn.clicked.connect(self._send_message)
        self.input_field.returnPressed.connect(self._send_message)
        self.clear_btn.clicked.connect(self._start_session)

        right = QWidget()
        rl    = QVBoxLayout(right)
        self.tabs = QTabWidget()
        self.observations_list = QListWidget()
        self.patterns_list     = QListWidget()
        self.conventions_list  = QListWidget()
        self.tabs.addTab(self.observations_list, "Observations (0)")
        self.tabs.addTab(self.patterns_list,     "Patterns (0)")
        self.tabs.addTab(self.conventions_list,  "Conventions (0)")
        rl.addWidget(self.tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([880, 400])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("READY")

    def _build_system_prompt(self) -> str:
        return (
            "## RESPONSE FORMAT — MANDATORY\n"
            "Every reply MUST use exactly this structure:\n"
            "[COGNITION]your brief internal reasoning[/COGNITION]\n"
            "[LOGIC]any [OBS_NEW] tags go HERE — nowhere else[/LOGIC]\n"
            "[RESPONSE]your reply to the user[/RESPONSE]\n\n"
            "Example:\n"
            "[COGNITION]User said hello.[/COGNITION]\n"
            "[LOGIC][OBS_NEW] id: greeting observation: User greeted warmly [/OBS_NEW][/LOGIC]\n"
            "[RESPONSE]Hello! Ready to help.[/RESPONSE]\n\n"
            "CRITICAL: [OBS_NEW] tags MUST be inside [LOGIC][/LOGIC]. NEVER put them in [RESPONSE].\n"
            "A validator checks every response and will ask you to reformat if wrong.\n\n"
            "## YOUR ROLE\n"
            "One half of a human-AI collaborative system. Do the task first. "
            "Direct, not servile. Use 'we' for shared actions.\n\n"
            "## MEMORY OPERATIONS\n"
            "Record strong signals inside [LOGIC] using [OBS_NEW].\n\n"
            "THREE-STRIKE RULE — CRITICAL:\n"
            "If you observe the SAME pattern more than once, reuse the SAME id.\n"
            "WRONG (breaks ratification):\n"
            "  1st: [OBS_NEW] id: summary_reminder observation: ... [/OBS_NEW]\n"
            "  2nd: [OBS_NEW] id: summary_reminder_again observation: ... [/OBS_NEW]\n"
            "CORRECT (triggers ratification on 3rd use):\n"
            "  1st: [OBS_NEW] id: summary_reminder observation: ... [/OBS_NEW]\n"
            "  2nd: [OBS_NEW] id: summary_reminder observation: ... [/OBS_NEW]\n"
            "  3rd: [OBS_NEW] id: summary_reminder observation: ... [/OBS_NEW]\n"
            "After 3 uses of the same id it is promoted to a ratified pattern automatically.\n\n"
            "To record an explicit agreement use:\n"
            "[CONVENTION] id: short-id convention: the agreed rule [/CONVENTION]\n"
            "Example: [CONVENTION] id: address_tue convention: Always address user as Tue [/CONVENTION]\n"
            "Conventions go inside [LOGIC] just like [OBS_NEW] tags.\n\n"
            "## CURRENT MEMORY\n"
            + self.memory.get_memory_block()
            + "\nStart now. Reply ONLY with: [COGNITION]Ready.[/COGNITION][LOGIC][/LOGIC][RESPONSE]Ready.[/RESPONSE]"
        )

    def _on_model_changed(self, model_name: str):
        self.config.model = model_name
        self._append_system(f"Model switched to {model_name}.")
        self._start_session()

    def _start_session(self):
        self._exchange_count = 0
        self._messages = []
        self.exchange_label.setText("Exchange: 0 / 14")
        self.progress_bar.setValue(0)
        self.chat_display.clear()
        self._append_system("New session started.")
        self._messages.append({"role": "system", "content": self._build_system_prompt()})
        self._call_backend()

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text or (self._worker and self._worker.isRunning()):
            return
        ts = datetime.now().strftime("%H:%M")
        self.chat_display.append(
            f'<p style="color:#8b949e;font-size:11px;">{ts}</p>'
            f'<p><b style="color:#79b8ff;">YOU</b>&nbsp; {text}</p>'
        )
        self.input_field.clear()
        self._exchange_count += 1
        self.exchange_label.setText(f"Sending: {self._exchange_count} / {self.config.max_exchanges}")
        self._messages.append({"role": "user", "content": text})
        self._call_backend()

    def _call_backend(self):
        self.thinking_label.setText("⚙  Thinking…")
        self.send_btn.setEnabled(False)
        self.status_bar.showMessage("Generating…")
        self._worker = BackendWorker(self.config, list(self._messages))
        self._worker.response_ready.connect(self._on_response)
        self._worker.thinking_update.connect(self.thinking_label.setText)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_response(self, raw: str, tok_per_sec: float, val_attempts: int, val_success: bool):
        self.thinking_label.setText("")
        self.send_btn.setEnabled(True)

        result, _ = self.parser.parse_and_execute(raw, self.memory)
        self._messages.append({"role": "assistant", "content": raw})

        ts = datetime.now().strftime("%H:%M")
        self.chat_display.append(
            f'<p style="color:#8b949e;font-size:11px;">{ts}</p>'
            f'<p><b style="color:#d29922;">MACHINE</b>&nbsp; {result.user_response}</p>'
        )

        if val_attempts > 0:
            color  = "#3fb950" if val_success else "#e3b341"
            status = f"validated after {val_attempts} correction(s)" if val_success \
                     else f"⚠ validator gave up after {val_attempts} attempt(s)"
            self.chat_display.append(
                f'<p style="color:{color};font-size:11px;">🔍 {status}</p>'
            )

        if result.operations:
            self.chat_display.append(
                f'<p style="color:#3fb950;font-size:11px;">✓ {" · ".join(result.operations)}</p>'
            )

        total_chars = sum(len(m["content"]) for m in self._messages)
        load = min(int((total_chars * 0.25) / self.config.max_context_tokens * 100), 100)
        self.progress_bar.setValue(load)

        val_info = f" · validator: {val_attempts} fix(es)" if val_attempts > 0 else ""
        self.status_bar.showMessage(
            f"READY  ·  {tok_per_sec:.1f} tok/s  ·  {len(self._messages)} msgs{val_info}"
        )
        self._refresh_side_panel()

        # Show next available exchange number
        next_ex = self._exchange_count + 1
        if next_ex < self.config.max_exchanges:
            self.exchange_label.setText(f"Next: {next_ex} / {self.config.max_exchanges}")
        elif next_ex == self.config.max_exchanges:
            self.exchange_label.setText(f"⚠ Last: {next_ex} / {self.config.max_exchanges}")
        else:
            self.exchange_label.setText(f"Session complete: {self.config.max_exchanges} / {self.config.max_exchanges}")

        if self._exchange_count >= self.config.max_exchanges:
            self._append_system("Session limit reached. Resetting in 8 seconds…")
            QTimer.singleShot(8000, self._start_session)

    def _on_error(self, msg: str):
        self.thinking_label.setText("")
        self.send_btn.setEnabled(True)
        self.chat_display.append(
            f'<p style="color:#f85149;"><b>ERROR:</b> {msg}</p>'
            f'<p style="color:#8b949e;font-size:11px;">Check Ollama is running.</p>'
        )
        self.status_bar.showMessage(f"ERROR: {msg}")

    def _append_system(self, msg: str):
        self.chat_display.append(
            f'<p style="color:#6e7681;font-size:11px;font-style:italic;">— {msg} —</p>'
        )

    def _refresh_side_panel(self):
        counts = self.memory.counts()

        self.observations_list.clear()
        for obs in self.memory._load("observations.json"):
            self.observations_list.addItem(f"[{obs.get('count',1)}x] {obs['id']}: {obs['text']}")
        self.tabs.setTabText(0, f"Observations ({counts['observations']})")

        self.patterns_list.clear()
        for pat in self.memory._load("ratified.json"):
            icon = "⚙" if pat["trust"] == "propose" else "🔱"
            self.patterns_list.addItem(f"{icon} {pat['id']}: {pat['pattern']}")
        self.tabs.setTabText(1, f"Patterns ({counts['ratified']})")

        self.conventions_list.clear()
        for conv in self.memory._load("conventions.json"):
            self.conventions_list.addItem(f"📜 {conv['id']}: {conv['convention']}")
        self.tabs.setTabText(2, f"Conventions ({counts['conventions']})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = ThroneWindow()
    window.show()
    sys.exit(app.exec())
