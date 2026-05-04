from __future__ import annotations

import argparse
import sys

import httpx


def ask(url: str, message: str, session_id: str, language: str, self_improve: bool) -> str:
    response = httpx.post(
        f"{url.rstrip('/')}/chat",
        json={
            "message": message,
            "session_id": session_id,
            "language": language,
            "self_improve": self_improve,
        },
        timeout=180,
    )
    if response.status_code in {500, 503}:
        try:
            data = response.json()
            detail = data.get("detail", response.text)
            hint = data.get("hint", "")
            return f"Сервер запущен, но модель не настроена.\n\nОшибка: {detail}\n{hint}".strip()
        except Exception:
            return f"Сервер вернул ошибку {response.status_code}: {response.text}"
    response.raise_for_status()
    data = response.json()
    answer = data.get("answer", "")
    model = data.get("model", "")
    score = data.get("evaluation_score")

    meta = f"\n\n[model: {model}"
    if score is not None:
        meta += f", score: {score}"
    meta += "]"
    return answer + meta


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Terminal chat client for Napi AI.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--session", default="terminal")
    parser.add_argument("--language", choices=["auto", "ru", "en", "pl"], default="auto")
    parser.add_argument("--no-improve", action="store_true")
    args = parser.parse_args()

    print("Napi terminal chat")
    print("Type /exit to quit, /health to check server.")
    print()

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not message:
            continue
        if message.lower() in {"/exit", "exit", "quit", "q"}:
            return 0
        if message.lower() == "/health":
            try:
                health = httpx.get(f"{args.url.rstrip('/')}/health", timeout=10).json()
                print(f"napi> {health}")
            except Exception as exc:
                print(f"napi> health error: {exc}")
            continue

        try:
            print("napi> thinking...")
            answer = ask(
                url=args.url,
                message=message,
                session_id=args.session,
                language=args.language,
                self_improve=not args.no_improve,
            )
            print(f"napi> {answer}")
        except Exception as exc:
            print(f"napi> error: {exc}")
            print("napi> Check config.yaml model/provider settings and napi_server.err.log.")


if __name__ == "__main__":
    raise SystemExit(main())
