"""Local development server for the Stain Normalisation dashboard."""

from __future__ import annotations
import argparse
import http.server
from pathlib import Path
import socketserver
import sys

_REPO = Path(__file__).resolve().parents[1]


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(_REPO), **kwargs)

    def do_GET(self):
        # Redirect root to /docs/
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/docs/")
            self.end_headers()
            return
        # Normalize /docs to /docs/
        elif self.path == "/docs":
            self.send_response(301)
            self.send_header("Location", "/docs/")
            self.end_headers()
            return
        elif self.path == "/docs/":
            self.path = "/docs/index.html"

        return super().do_GET()

    def log_message(self, format, *args):
        # Concise logging to standard output
        sys.stdout.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} {args[2]}\n")
        sys.stdout.flush()


def run(port: int = 8000, host: str = "127.0.0.1") -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), DashboardHandler) as httpd:
        print(f"Stain Normalisation Dashboard running at:", flush=True)
        print(f"  -> http://localhost:{port}/", flush=True)
        print(f"  -> http://{host}:{port}/docs/", flush=True)
        print("Serving files from project root and docs directory.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dashboard on localhost.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()
    run(port=args.port, host=args.host)
