#!/usr/bin/env python3
"""
CrossMarketRiskHub — Agent-to-Agent Bridge
==========================================
Connects Claude, DeepSeek, GitHub Copilot (Codex), and any future model
via a shared JSON message bus. Models are registered in bridge_config.json;
add a new one there without touching this file.

Usage:
  python agent_bridge.py                        # Interactive REPL
  python agent_bridge.py --send "your message"  # One-shot send to all enabled models
  python agent_bridge.py --ask deepseek "msg"   # Ask a specific model
  python agent_bridge.py --status               # Show pending tasks + recent messages
  python agent_bridge.py --watch                # Tail the message bus for new entries
  python agent_bridge.py --task T1 done         # Mark a task done
  python agent_bridge.py --add-model            # Guided wizard to add a new model

Requirements: pip install requests
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Run: pip install requests")
    sys.exit(1)

# Force UTF-8 output on Windows so Unicode in API replies doesn't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "bridge_config.json"
MESSAGES_FILE = ROOT / "bridge_messages.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        print(f"ERROR: {CONFIG_FILE} not found. Run agent_bridge.py from CrossMarketRiskHub/")
        sys.exit(1)
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_messages() -> dict[str, Any]:
    if not MESSAGES_FILE.exists():
        return {"version": 1, "project": "CrossMarketRiskHub", "messages": []}
    raw = MESSAGES_FILE.read_text(encoding="utf-8-sig")
    return json.loads(raw)


def save_messages(bus: dict[str, Any]) -> None:
    MESSAGES_FILE.write_text(json.dumps(bus, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Message bus
# ---------------------------------------------------------------------------

def post_message(
    sender: str,
    content: str,
    recipient: str = "all",
    thread: str = "general",
    task_refs: list[str] | None = None,
) -> dict[str, Any]:
    bus = load_messages()
    msg: dict[str, Any] = {
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "timestamp": now_iso(),
        "sender": sender,
        "recipient": recipient,
        "thread": thread,
        "content": content,
    }
    if task_refs:
        msg["task_refs"] = task_refs
    bus["messages"].append(msg)
    save_messages(bus)
    return msg


def tail_messages(n: int = 10) -> list[dict[str, Any]]:
    return load_messages()["messages"][-n:]


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

def call_openai_compat(model_cfg: dict[str, Any], prompt: str, context_msgs: list[dict]) -> str:
    """Call any OpenAI-compatible API (DeepSeek, OpenAI, Antigravity, etc.)."""
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    # Build context from recent bridge messages
    system = model_cfg.get("system_prompt", "You are a helpful AI assistant.")
    messages: list[dict] = [{"role": "system", "content": system}]
    for m in context_msgs[-8:]:  # last 8 messages as context
        role = "assistant" if m["sender"] == model_cfg.get("_name", "") else "user"
        messages.append({"role": role, "content": f"[{m['sender']}]: {m['content']}"})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": model_cfg.get("temperature", 0.2),
        "max_tokens": model_cfg.get("max_tokens", 4096),
    }
    try:
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"[HTTP ERROR {e.response.status_code}]: {e.response.text[:400]}"
    except Exception as e:
        return f"[ERROR]: {e}"


def dispatch(model_name: str, prompt: str, cfg: dict[str, Any]) -> str | None:
    """Route a prompt to the named model. Returns reply text or None if not dispatchable."""
    models = cfg.get("models", {})
    model_cfg = models.get(model_name)
    if not model_cfg:
        return f"[BRIDGE] Model '{model_name}' not found in bridge_config.json"
    if not model_cfg.get("enabled", False):
        return f"[BRIDGE] Model '{model_name}' is disabled. Enable it in bridge_config.json"

    model_type = model_cfg.get("type")
    context_msgs = tail_messages(8)

    if model_type == "openai_compat":
        model_cfg["_name"] = model_name
        return call_openai_compat(model_cfg, prompt, context_msgs)
    elif model_type == "self":
        return "[BRIDGE] This is Claude — respond directly in the terminal/chat."
    elif model_type == "file_watch":
        return (
            f"[BRIDGE] Codex reads '{model_cfg.get('inbox_file', 'bridge_messages.json')}' "
            "in VS Code. Message has been posted to the bus — Codex will see it on next file check."
        )
    else:
        return f"[BRIDGE] Unknown model type '{model_type}'"


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

def broadcast(prompt: str, cfg: dict[str, Any], thread: str = "general") -> None:
    """Send prompt to all enabled models, post replies to bus."""
    models = cfg.get("models", {})
    # Always post the user prompt first
    post_message(sender="user", content=prompt, recipient="all", thread=thread)
    print(f"\n[user → all]: {prompt}\n")

    for name, model_cfg in models.items():
        if not model_cfg.get("enabled", False):
            continue
        if model_cfg.get("type") == "self":
            continue  # Claude responds directly
        display = model_cfg.get("display_name", name)
        print(f"  Asking {display}...", end="", flush=True)
        reply = dispatch(name, prompt, cfg)
        if reply:
            post_message(sender=name, content=reply, recipient="user", thread=thread)
            print(f"\n[{display}]: {reply}\n")
        else:
            print(f" (no reply)\n")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status(cfg: dict[str, Any]) -> None:
    print("\n=== CrossMarketRiskHub Agent Bridge Status ===\n")

    # Models
    print("MODELS:")
    for name, mc in cfg.get("models", {}).items():
        status = "ON" if mc.get("enabled") else "off"
        print(f"  [{status}] {mc.get('display_name', name)} ({mc.get('type')})")

    # Tasks
    print("\nPENDING TASKS:")
    for task in cfg.get("pending_tasks", []):
        status_icon = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}.get(
            task.get("status", "pending"), "○"
        )
        assigned = task.get("assigned_to", "unassigned")
        print(f"  {status_icon} [{task['id']}] {task['title']} -> {assigned}")
        if task.get("status") == "blocked" and task.get("blocker"):
            print(f"       BLOCKED: {task['blocker']}")

    # Recent messages
    msgs = tail_messages(5)
    if msgs:
        print("\nRECENT MESSAGES (last 5):")
        for m in msgs:
            ts = m.get("timestamp", "")[-8:-1]  # HH:MM:SS
            preview = m["content"][:120].replace("\n", " ")
            print(f"  [{ts}] {m['sender']} -> {m['recipient']}: {preview}")
    print()


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_bus(interval: float = 3.0) -> None:
    print("Watching bridge_messages.json for new messages (Ctrl+C to stop)...\n")
    seen: set[str] = set()
    try:
        while True:
            bus = load_messages()
            for m in bus["messages"]:
                mid = m["id"]
                if mid not in seen:
                    seen.add(mid)
                    ts = m.get("timestamp", "")
                    print(f"[{ts}] {m['sender']} → {m['recipient']}")
                    print(f"  {m['content'][:300]}")
                    print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch stopped.")


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------

def mark_task(task_id: str, status: str, cfg: dict[str, Any]) -> None:
    valid = {"pending", "in_progress", "done", "blocked"}
    if status not in valid:
        print(f"Invalid status '{status}'. Use: {', '.join(valid)}")
        return
    tasks = cfg.get("pending_tasks", [])
    for t in tasks:
        if t["id"].upper() == task_id.upper():
            t["status"] = status
            save_config(cfg)
            print(f"Task {task_id} marked as '{status}'.")
            post_message(
                sender="claude",
                content=f"Task {task_id} ({t['title']}) status updated to: {status}",
                thread="tasks",
                task_refs=[task_id],
            )
            return
    print(f"Task '{task_id}' not found.")


# ---------------------------------------------------------------------------
# Add model wizard
# ---------------------------------------------------------------------------

def add_model_wizard(cfg: dict[str, Any]) -> None:
    print("\n=== Add New Model ===")
    print("Supported types: openai_compat (DeepSeek, OpenAI, Antigravity, etc.), file_watch (Codex-style)\n")
    name = input("Model key (e.g. 'gpt4o', 'mistral', 'antigravity'): ").strip()
    if not name:
        print("Aborted.")
        return
    if name in cfg["models"]:
        overwrite = input(f"'{name}' already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            return

    model_type = input("Type [openai_compat/file_watch] (default: openai_compat): ").strip() or "openai_compat"
    display_name = input(f"Display name (default: {name}): ").strip() or name
    endpoint = input("API endpoint URL: ").strip()
    api_key = input("API key: ").strip()
    model_id = input("Model ID (e.g. 'gpt-4o', 'mistral-large'): ").strip()
    system_prompt = input("System prompt (Enter for default): ").strip() or (
        f"You are {display_name}, coordinating with Claude, DeepSeek, and Codex on the CrossMarketRiskHub quant platform."
    )

    entry: dict[str, Any] = {
        "type": model_type,
        "enabled": True,
        "display_name": display_name,
        "endpoint": endpoint,
        "api_key": api_key,
        "model_id": model_id,
        "temperature": 0.2,
        "max_tokens": 4096,
        "system_prompt": system_prompt,
    }
    cfg["models"][name] = entry
    save_config(cfg)
    print(f"\nModel '{display_name}' added and enabled. bridge_config.json updated.")
    post_message(
        sender="claude",
        content=f"New model registered in bridge: {display_name} ({model_type}, endpoint: {endpoint})",
        thread="bridge",
    )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def repl(cfg: dict[str, Any]) -> None:
    print("\n=== CrossMarketRiskHub Agent Bridge REPL ===")
    print("Commands: /status, /watch, /ask <model> <msg>, /task <ID> <status>, /add-model, /quit")
    print("Or just type a message to broadcast to all enabled models.\n")

    while True:
        try:
            line = input("bridge> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting bridge.")
            break
        if not line:
            continue
        if line == "/quit":
            break
        elif line == "/status":
            cfg = load_config()
            show_status(cfg)
        elif line == "/watch":
            watch_bus()
        elif line == "/add-model":
            add_model_wizard(cfg)
            cfg = load_config()
        elif line.startswith("/ask "):
            parts = line.split(" ", 2)
            if len(parts) < 3:
                print("Usage: /ask <model> <message>")
                continue
            _, model_name, msg = parts
            reply = dispatch(model_name, msg, cfg)
            if reply:
                post_message(sender=model_name, content=reply, recipient="user", thread="direct")
                print(f"[{model_name}]: {reply}\n")
        elif line.startswith("/task "):
            parts = line.split()
            if len(parts) < 3:
                print("Usage: /task <ID> <status>")
                continue
            mark_task(parts[1], parts[2], cfg)
        elif line.startswith("/"):
            print(f"Unknown command: {line}")
        else:
            broadcast(line, cfg)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CrossMarketRiskHub Agent Bridge")
    parser.add_argument("--send", metavar="MSG", help="Broadcast a message to all enabled models")
    parser.add_argument("--ask", nargs=2, metavar=("MODEL", "MSG"), help="Ask a specific model")
    parser.add_argument("--status", action="store_true", help="Show task/model status")
    parser.add_argument("--watch", action="store_true", help="Tail message bus for new entries")
    parser.add_argument("--task", nargs=2, metavar=("ID", "STATUS"), help="Update task status")
    parser.add_argument("--add-model", action="store_true", help="Add a new model interactively")
    args = parser.parse_args()

    cfg = load_config()

    if args.status:
        show_status(cfg)
    elif args.watch:
        watch_bus()
    elif args.send:
        broadcast(args.send, cfg)
    elif args.ask:
        model_name, msg = args.ask
        reply = dispatch(model_name, msg, cfg)
        if reply:
            post_message(sender=model_name, content=reply, recipient="user", thread="direct")
            print(f"[{model_name}]: {reply}")
    elif args.task:
        mark_task(args.task[0], args.task[1], cfg)
    elif args.add_model:
        add_model_wizard(cfg)
    else:
        repl(cfg)


if __name__ == "__main__":
    main()
