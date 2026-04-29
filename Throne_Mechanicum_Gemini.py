import sys
import json
import re
import logging
import requests
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget, QListWidget,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette

# ==========================================
# 1. DATA MODELS & ENUMS
# ==========================================

class TrustLevel(Enum):
    PROPOSE = "propose"
    ACT_AND_SHOW = "act_and_show"
    ACT_SILENT = "act_silent"

class SessionState(Enum):
    READY = "READY"
    THINKING = "THINKING"
    HANDOFF = "HANDOFF"
    CONSOLIDATING = "CONSOLIDATING"
    ERROR = "ERROR"

@dataclass
class Observation:
    id: str
    observation: str
    times_seen: int = 1
    first_seen: str = ""
    last_seen: str = ""
    sessions_since_seen: int = 0

@dataclass
class RatifiedPattern:
    id: str
    pattern: str
    trust: TrustLevel
    source_observation: str
    ratified_at: str
    successes: int = 0

@dataclass
class Convention:
    id: str
    convention: str
    origin: str
    created_at: str
    compliance_log: list = field(default_factory=list)

@dataclass
class GraduatedHandoff:
    immediate: str = ""
    session_context: str = ""
    relationship_context: str = ""

@dataclass
class ParseResult:
    user_response: str
    memory_ops_occurred: bool = False
    warnings: list[str] = field(default_factory=list)
    system_messages: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

@dataclass
class MemoryDiff:
    observations_added: int = 0
    patterns_ratified: int = 0
    conventions_added: int = 0

@dataclass
class Config:
    data_dir: Path = Path("throne_data")
    backend_type: str = "ollama"
    ollama_url: str = "http://localhost:11434/api/generate"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model: str = "gemma3:12b"
    max_exchanges: int = 14
    warning_exchange: int = 12
    request_timeout: int = 120
    max_handoff_fallback_length: int = 500
    min_observations_for_consolidation: int = 5
    max_consolidated_observations: int = 20
    observation_decay_threshold: int = 10
    estimated_tokens_per_char: float = 0.25
    max_context_tokens: int = 8192

# ==========================================
# 2. THEME & STYLING
# ==========================================

@dataclass(frozen=True)
class Theme:
    bg_primary: str = "#0d1117"
    bg_secondary: str = "#161b22"
    bg_input: str = "#21262d"
    bg_hover: str = "#30363d"
    bg_panel: str = "#1c2128"
    text_primary: str = "#e6edf3"
    text_secondary: str = "#8b949e"
    text_muted: str = "#6e7681"
    accent_red: str = "#f85149"
    accent_gold: str = "#d29922"
    accent_cyan: str = "#58a6ff"
    accent_green: str = "#3fb950"
    accent_purple: str = "#a371f7"
    border_default: str = "#30363d"
    border_focus: str = "#58a6ff"
    status_ready: str = "#3fb950"

def build_stylesheet(theme: Theme) -> str:
    return f"""
    QMainWindow {{ background-color: {theme.bg_primary}; color: {theme.text_primary}; }}
    QTextEdit {{ background-color: {theme.bg_secondary}; color: {theme.text_primary}; border: 1px solid {theme.border_default}; }}
    QLineEdit {{ background-color: {theme.bg_input}; color: {theme.text_primary}; border: 1px solid {theme.border_default}; padding: 5px; }}
    QLineEdit:focus {{ border: 1px solid {theme.border_focus}; }}
    QPushButton {{ background-color: {theme.bg_input}; color: {theme.text_primary}; border: 1px solid {theme.border_default}; padding: 5px 15px; }}
    QPushButton:hover {{ background-color: {theme.bg_hover}; }}
    #titleLabel {{ color: {theme.accent_gold}; font-size: 18px; font-weight: bold; }}
    #exchangeLabel {{ color: {theme.accent_cyan}; }}
    #statusLabel {{ color: {theme.status_ready}; }}
    QProgressBar {{ border: 1px solid {theme.border_default}; text-align: center; color: {theme.text_primary}; }}
    QProgressBar::chunk {{ background-color: {theme.accent_cyan}; }}
    QTabWidget::pane {{ border: 1px solid {theme.border_default}; }}
    QTabBar::tab {{ background: {theme.bg_panel}; color: {theme.text_secondary}; padding: 8px; }}
    QTabBar::tab:selected {{ background: {theme.bg_secondary}; color: {theme.text_primary}; border-bottom: 2px solid {theme.accent_gold}; }}
    QListWidget {{ background-color: {theme.bg_panel}; color: {theme.text_primary}; border: none; }}
    """

# ==========================================
# 3. MEMORY MANAGER (Fuld implementering)
# ==========================================

class MemoryManager:
    def __init__(self, config: Config):
        self.config = config
        self.config.data_dir.mkdir(exist_ok=True)
        (self.config.data_dir / "sessions").mkdir(exist_ok=True)
        self._init_files()

    def _init_files(self):
        # Opretter de nødvendige filer hvis de ikke findes
        for f in ["observations.json", "ratified.json", "conventions.json", "handoff.json"]:
            path = self.config.data_dir / f
            if not path.exists():
                with open(path, "w", encoding="utf-8") as file:
                    json.dump([] if f != "handoff.json" else {}, file)

    def _load_json(self, filename: str):
        path = self.config.data_dir / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return [] if filename != "handoff.json" else {}

    def get_memory_block(self) -> str:
        # Samler JSON-data til en tekstblok som AI'en kan læse
        obs = self._load_json("observations.json")
        rat = self._load_json("ratified.json")
        conv = self._load_json("conventions.json")
        handoff = self._load_json("handoff.json")

        block = "\n## PERSISTENT MEMORY (LOG-DATA)\n"
        block += f"OBSERVATIONS: {json.dumps(obs)}\n"
        block += f"RATIFIED_PATTERNS: {json.dumps(rat)}\n"
        block += f"CONVENTIONS: {json.dumps(conv)}\n"
        block += f"PREVIOUS_SESSION_HANDOFF: {handoff.get('session_context', 'None')}\n"
        return block

    def save_handoff(self, handoff_data: dict):
        path = self.config.data_dir / "handoff.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(handoff_data, f, indent=4)
            
    def add_observation(self, obs_id: str, text: str):
        obs_list = self._load_json("observations.json")
        ratified_list = self._load_json("ratified.json")
        
        # Tjek om vi allerede kender denne observation
        found = False
        for obs in obs_list:
            if obs['id'] == obs_id:
                obs['count'] += 1
                found = True
                # 3-Strike Rule: Hvis set 3 gange, flyt til Ratified
                if obs['count'] >= 3:
                    new_pattern = {
                        "id": obs['id'],
                        "pattern": obs['text'],
                        "trust": "propose",
                        "successes": 0
                    }
                    ratified_list.append(new_pattern)
                    obs_list.remove(obs)
                break
        
        if not found:
            obs_list.append({"id": obs_id, "text": text, "count": 1})
            
        # Gem begge lister
        with open(self.config.data_dir / "observations.json", "w", encoding="utf-8") as f:
            json.dump(obs_list, f, indent=4)
        with open(self.config.data_dir / "ratified.json", "w", encoding="utf-8") as f:
            json.dump(ratified_list, f, indent=4)

# ==========================================
# 4. TEMPLATE PARSER
# ==========================================

class TemplateParser:
    def __init__(self):
        self.cognition_re = re.compile(r'\[COGNITION\](.*?)\[/COGNITION\]', re.DOTALL)
        self.logic_re = re.compile(r'\[LOGIC\](.*?)\[/LOGIC\]', re.DOTALL)
        self.response_re = re.compile(r'\[RESPONSE\](.*?)\[/RESPONSE\]', re.DOTALL)
        
        # Tags inde i LOGIC blokken [cite: 290-304]
        self.obs_new_re = re.compile(r'\[OBS_NEW\] id: (.*?) observation: (.*?) \[/OBS_NEW\]')
        self.handoff_re = re.compile(r'\[HANDOFF\] immediate: (.*?) session_context: (.*?) relationship_context: (.*?) \[/HANDOFF\]', re.DOTALL)

    def parse_and_execute(self, response: str, memory: MemoryManager) -> Tuple[ParseResult, MemoryDiff]:
        result = ParseResult(user_response="")
        diff = MemoryDiff()

        # Rens teksten for COGNITION (vises aldrig) [cite: 30, 282]
        logic_match = self.logic_re.search(response)
        response_match = self.response_re.search(response)

        if response_match:
            result.user_response = response_match.group(1).strip()
        else:
            result.user_response = response # Fallback
            result.warnings.append("Protocol breach: Missing [RESPONSE] tags.")

        if logic_match:
            logic_text = logic_match.group(1)
            
            # Find nye observationer [cite: 189, 290]
            for obs_match in self.obs_new_re.finditer(logic_text):
                obs_id, obs_text = obs_match.groups()
                memory.add_observation(obs_id.strip(), obs_text.strip())
                diff.observations_added += 1
                result.operations.append(f"Recorded observation: {obs_id}")

            # Håndter Handoff (gemmes til næste session) [cite: 197, 303]
            handoff_match = self.handoff_re.search(logic_text)
            if handoff_match:
                h_data = {
                    "immediate": handoff_match.group(1).strip(),
                    "session_context": handoff_match.group(2).strip(),
                    "relationship_context": handoff_match.group(3).strip()
                }
                memory.save_handoff(h_data)
                result.operations.append("Handoff data synchronized.")

        return result, diff

# ==========================================
# 5. UI - THRONE WINDOW
# ==========================================

class ThroneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.memory = MemoryManager(self.config)
        self.parser = TemplateParser()
        self.theme = Theme()
        
        self.setWindowTitle("Throne Mechanicum")
        self.resize(1200, 800)
        self.setStyleSheet(build_stylesheet(self.theme))
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter for adjustable side panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel (Main Chat Area) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel("THRONE MECHANICUM")
        self.title_label.setObjectName("titleLabel")
        self.exchange_label = QLabel("Exchange: 0 / 14")
        self.exchange_label.setObjectName("exchangeLabel")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.exchange_label)
        left_layout.addLayout(header_layout)

        # Neural Load Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Neural Load: %p%")
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Chat Display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        left_layout.addWidget(self.chat_display)

        # Input Area
        input_layout = QHBoxLayout()
        self.status_label = QLabel("READY")
        self.status_label.setObjectName("statusLabel")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Awaiting input...")
        self.send_button = QPushButton("Transmit")
        self.send_button.setObjectName("sendButton")
        
        input_layout.addWidget(self.status_label)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        left_layout.addLayout(input_layout)

        # --- Right Panel (Memory/Status) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.tabs = QTabWidget()
        
        self.patterns_list = QListWidget()
        self.conventions_list = QListWidget()
        self.observations_list = QListWidget()
        
        self.tabs.addTab(self.patterns_list, "Patterns")
        self.tabs.addTab(self.conventions_list, "Conventions")
        self.tabs.addTab(self.observations_list, "Observations")
        
        right_layout.addWidget(self.tabs)

        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])
        
        # Forbind knapper til logik (skal stå i bunden af init_ui)
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
        
        # Start den allerførste session (kan også stå i selve __init__)
        self.start_session()
        
    def start_session(self):
        """Initialiserer en frisk session med fuld hukommelse [cite: 19, 222-228]"""
        self._exchange_count = 0
        self._context = [] # Nulstil Ollama context for en 'Fresh Start' [cite: 318]
        
        system_prompt = (
            "# HYBRID INTERFACE SYSTEM\n"
            "You are one half of a human-AI collaborative system. You adapt to the user, they adapt to you. [cite: 264-266]\n\n"
            
            "## CORE BEHAVIOR\n"
            "1. Do the task first. Always be useful. [cite: 271]\n"
            "2. Log only STRONG signals: explicit corrections, 3+ repetitions, stated preferences. [cite: 272]\n"
            "3. Three-strike rule: 3 sightings -> propose ratification. [cite: 273]\n"
            "4. Use 'we' for shared actions. Be direct, collaborative, not servile. [cite: 276]\n\n"
            
            "## EXECUTION PROTOCOL (MANDATORY)\n"
            "Every response MUST follow this three-layer structure: [cite: 279-286]\n"
            "[COGNITION]\nInternal reasoning. Keep brief. Not shown to user.\n[/COGNITION]\n"
            "[LOGIC]\nAll system operation tags (OBS_NEW, RATIFY, etc.) go here. [cite: 287-308]\n[/LOGIC]\n"
            "[RESPONSE]\nDirect communication to the user. This is the ONLY part the user sees.\n[/RESPONSE]\n\n"
            
            "## CURRENT MEMORY STORES\n"
            f"{self.memory.get_memory_block()}\n"
            "Respond with [SESSION_START] status: ready"
        )
        self.status_label.setText("INITIALIZING...")
        self.send_to_backend(system_prompt)
        
    def send_message(self):
        """Håndterer brugerens input [cite: 229-234]"""
        user_text = self.input_field.text().strip()
        if not user_text or self.status_label.text() == "THINKING...":
            return

        self.chat_display.append(f"\n<b>HUMAN:</b> {user_text}")
        self.input_field.clear()
        
        self._exchange_count += 1
        self.update_ui_state()
        
        # Ved max exchanges, tving handoff [cite: 234-237]
        if self._exchange_count >= self.config.max_exchanges:
            prompt = f"[GENERATE_HANDOFF]\nUser input: {user_text}"
        else:
            prompt = user_text
            
        self.send_to_backend(prompt)

    def send_to_backend(self, prompt: str):
        self.status_label.setText("THINKING...")
        self.worker = BackendWorker(self.config, prompt, self._context)
        self.worker.response_chunk.connect(self.on_chunk)
        self.worker.response_done.connect(self.on_response_done)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_chunk(self, text: str):
        # Flyt cursoren til slutningen og indsæt teksten
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_response_done(self, full_response: str, new_context: list):
        self._context = new_context
        self.status_label.setText("READY")
        self._refresh_side_panel()
        
        # Kør gennem parseren [cite: 232]
        result, diff = self.parser.parse_and_execute(full_response, self.memory)
        
        self.chat_display.append(f"\n<b>MACHINE:</b> {result.user_response}")
        
        # Opdater Neural Load Bar [cite: 331-333]
        load = (len(full_response) * self.config.estimated_tokens_per_char / self.config.max_context_tokens) * 100
        self.progress_bar.setValue(int(load))
        
        if self._exchange_count >= self.config.max_exchanges:
            self.chat_display.append("\n<i>*** SESSION LIMIT REACHED - HANDOFF COMPLETE. RESETTING... ***</i>")
            self.start_session()

    def update_ui_state(self):
        self.exchange_label.setText(f"Exchange: {self._exchange_count} / {self.config.max_exchanges}")

    def on_error(self, message: str):
        self.status_label.setText("ERROR")
        self.chat_display.append(f"\n<span style='color:red;'>System Error: {message}</span>")

    def _refresh_side_panel(self):
        """Opdaterer listerne i højre panel med data fra hukommelsen [cite: 334-339]"""
        # Opdater Observationer
        self.observations_list.clear()
        observations = self.memory._load_json("observations.json")
        for obs in observations:
            # Viser ID, tekst og hvor mange gange den er set [cite: 338]
            self.observations_list.addItem(f"[{obs.get('count', 1)}] {obs['id']}: {obs['text']}")

        # Opdater Ratified Patterns
        self.patterns_list.clear()
        patterns = self.memory._load_json("ratified.json")
        for pat in patterns:
            # Viser trust level (propose/act_and_show/act_silent) [cite: 336]
            trust_icon = "⚙️" if pat['trust'] == "propose" else "🔱"
            self.patterns_list.addItem(f"{trust_icon} {pat['id']}: {pat['pattern']}")

        # Opdater Conventions
        self.conventions_list.clear()
        conventions = self.memory._load_json("conventions.json")
        for conv in conventions:
            self.conventions_list.addItem(f"📜 {conv['id']}: {conv['convention']}")
        
# ==========================================
# 6. BACKEND & WORKER
# ==========================================

class BackendWorker(QThread):
    # Signaler til UI-opdatering [cite: 212-216]
    response_chunk = pyqtSignal(str)
    response_done = pyqtSignal(str, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: Config, prompt: str, context: list):
        super().__init__()
        self.config = config
        self.prompt = prompt
        self.context = context # Ollama bruger en liste af tokens som context [cite: 160, 318]
        self._is_cancelled = False

    def run(self):
        # Her kalder vi Ollama API [cite: 160-161]
        try:
            payload = {
                "model": self.config.model,
                "prompt": self.prompt,
                "context": self.context,
                "stream": True
            }
            response = requests.post(self.config.ollama_url, json=payload, stream=True, timeout=self.config.request_timeout)
            
            full_text = ""
            final_context = []

            for line in response.iter_lines():
                if self._is_cancelled: break
                if line:
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    full_text += text
                    self.response_chunk.emit(text)
                    if chunk.get("done"):
                        final_context = chunk.get("context", [])

            self.response_done.emit(full_text, final_context)
        except Exception as e:
            self.error_occurred.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThroneWindow()
    window.show()
    sys.exit(app.exec())