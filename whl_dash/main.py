#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2026-03-23
# Author: daohu527


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
        "--record_path",
        nargs="?",
        default="",
        help="Path to a .record file or directory with .record files",
    )
    parser.add_argument("--host", default="0.0.0.0")
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
