"""
Transcendence CLI Chat — Interactive REPL for chatting with fine-tuned models.

Usage:
    # API mode (connect to a deployed model):
    python scripts/chat_cli.py --endpoint http://localhost:8001

    # Direct mode (load model in-process):
    python scripts/chat_cli.py --model-path Qwen/Qwen3-1.7B
    python scripts/chat_cli.py --model-path Qwen/Qwen3-1.7B --adapter-path ./output/lora

    # With options:
    python scripts/chat_cli.py --endpoint http://localhost:8001 --system-prompt "You are a pirate."
    python scripts/chat_cli.py --model-path ./model --no-stream --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import asyncio
import sys

# Ensure stderr uses UTF-8 on Windows
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcendence-chat",
        description="Interactive CLI chat with a fine-tuned model.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="URL of a running model API (e.g., http://localhost:8001)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path for direct loading",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to conversations",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    return parser


def _print_banner(info: dict) -> None:
    """Print the welcome banner with session info."""
    sys.stdout.write("\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("  Transcendence Chat\n")
    sys.stdout.write("=" * 50 + "\n")

    mode = info.get("mode", "unknown")
    if mode == "api":
        sys.stdout.write(f"  Mode:     API ({info.get('endpoint', '')})\n")
        model = info.get("model", "unknown")
        sys.stdout.write(f"  Model:    {model}\n")
    else:
        sys.stdout.write("  Mode:     Direct (in-process)\n")
        sys.stdout.write(f"  Model:    {info.get('model_path', 'unknown')}\n")
        if info.get("adapter_path"):
            sys.stdout.write(f"  Adapter:  {info['adapter_path']}\n")

    sys.stdout.write("\n")
    sys.stdout.write("  Commands: /help /clear /info /quit\n")
    sys.stdout.write("=" * 50 + "\n\n")
    sys.stdout.flush()


def _print_help() -> None:
    """Print available REPL commands."""
    sys.stdout.write("\nAvailable commands:\n")
    sys.stdout.write("  /help   - Show this help message\n")
    sys.stdout.write("  /clear  - Clear conversation history\n")
    sys.stdout.write("  /info   - Show current session info\n")
    sys.stdout.write("  /quit   - Exit the chat\n")
    sys.stdout.write("  /exit   - Exit the chat\n\n")
    sys.stdout.flush()


def _print_info(info: dict) -> None:
    """Print session info."""
    sys.stdout.write("\nSession Info:\n")
    for key, value in info.items():
        sys.stdout.write(f"  {key}: {value}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


async def _run_repl(session) -> None:
    """Main REPL loop."""
    use_streaming = session._config.streaming

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                sys.stdout.write("Goodbye!\n")
                sys.stdout.flush()
                break
            elif cmd == "/clear":
                session.clear_history()
                sys.stdout.write("Conversation history cleared.\n\n")
                sys.stdout.flush()
                continue
            elif cmd == "/info":
                _print_info(session.get_info())
                continue
            elif cmd == "/help":
                _print_help()
                continue
            else:
                sys.stdout.write(f"Unknown command: {user_input}\n")
                sys.stdout.write("Type /help for available commands.\n\n")
                sys.stdout.flush()
                continue

        # Generate response
        sys.stdout.write("\nAssistant: ")
        sys.stdout.flush()

        if use_streaming:
            async for token in session.stream_message(user_input):
                sys.stdout.write(token)
                sys.stdout.flush()
        else:
            response = await session.send_message(user_input)
            sys.stdout.write(response)
            sys.stdout.flush()

        sys.stdout.write("\n\n")
        sys.stdout.flush()


def main() -> None:
    args = _build_parser().parse_args()

    if not args.endpoint and not args.model_path:
        sys.stderr.write(
            "ERROR: provide at least one of --endpoint or --model-path\n"
        )
        raise SystemExit(1)

    from shared.config import ChatConfig
    from hosting_pipeline.services.chat_service import ChatSession

    config = ChatConfig(
        endpoint=args.endpoint,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        streaming=not args.no_stream,
    )

    session = ChatSession(config)

    async def _run() -> None:
        try:
            info = await session.initialize()
            _print_banner(info)
            await _run_repl(session)
        except KeyboardInterrupt:
            sys.stdout.write("\n\nInterrupted. Shutting down...\n")
            sys.stdout.flush()
        finally:
            await session.shutdown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
