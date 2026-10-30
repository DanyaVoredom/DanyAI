import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("LORA_TOKEN", "danyai-up")
OUT = "/home/lipton/lora_in"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        self.send_response(200)
        body = b"ok\n"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "upload" or parts[1] != TOKEN:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        name = os.path.basename(parts[2]) or "snap.zip"
        n = int(self.headers.get("Content-Length", 0))
        os.makedirs(OUT, exist_ok=True)
        tmp = os.path.join(OUT, name + ".part")
        got = 0
        with open(tmp, "wb") as f:
            while got < n:
                chunk = self.rfile.read(min(1 << 20, n - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        if got != n:
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        os.replace(tmp, os.path.join(OUT, name))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
