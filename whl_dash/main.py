import argparse
import sys
import threading
import webbrowser

from dashboard import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apollo Universal Protobuf Formula Dashboard"
    )
    parser.add_argument(
        "record_path",
        nargs="?",
        default="../2026-03-20-09-26-20",
        help="Path to a .record file or directory with .record files",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    try:
        app = create_app(args.record_path)
    except Exception as e:
        print(f"Failed to initialize dashboard: {e}")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print(f"Starting dashboard at {url}")

    # Automatically open local browser if running on 127.0.0.1
    if args.host == "127.0.0.1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
