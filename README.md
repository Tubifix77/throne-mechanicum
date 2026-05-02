# Throne Mechanicum

A PyQt6 chat interface for local Ollama models with persistent memory and a validator sub-agent that catches malformed LLM responses before they reach the UI.

## What it does

You chat with a local LLM (gemma3:12b by default) through a desktop GUI. The model maintains persistent memory across sessions through a system of observations, ratified patterns, and conventions — so it remembers what it has learned about you and your preferences over time.

A validator sub-agent runs in the background, inspecting every LLM response before it's displayed. Malformed outputs get caught and retried instead of crashing the conversation.

## Features

- **Desktop GUI** built with PyQt6 — double-click to launch (.pyw, no console window)
- **Persistent memory** via observations/ratified patterns/conventions stored in local files
- **Validator sub-agent** catches malformed LLM responses before they reach the UI
- **Local-only** — connects to Ollama at localhost:11434
- **No admin needed, no console** — runs as a .pyw file

## Stack

- **UI:** PyQt6
- **LLM:** gemma3:12b via Ollama
- **Memory:** File-based persistent storage in `throne_data/`

## Usage

Double-click `Throne_Mechanicum_v2.pyw`. Requires Ollama running with gemma3:12b pulled.

## Known quirks

- The model occasionally spells observation IDs inconsistently
- Validator warnings sometimes leak into response text

## Origin

Built by Tue Boas and Claude (Anthropic) in April 2026. The original version (`Throne_Mechanicum_Gemini.py`) was built with Google Gemini; v2 is the Claude-assisted rewrite with persistent memory and the validator sub-agent.

## License

MIT
