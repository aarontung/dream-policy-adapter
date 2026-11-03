"""Loopback-only worker; run with python -m dream_policy_adapter.worker."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from .dream_policy import DreamPolicy, load_sdk


class SessionManager:
    def __init__(self, factory=DreamPolicy):
        self.factory = factory
        self.lock = threading.Lock()
        self.policy = self.session_id = None

    def close(self):
        policy, self.policy, self.session_id = self.policy, None, None
        if policy is not None:
            policy.close()

    def dispatch(self, path, payload):
        sid = payload.get('session_id')
        if not isinstance(sid, str) or len(sid) != 32 or any(c not in '0123456789abcdef' for c in sid):
            raise ValueError("Invalid session_id")
        # If an SDK call hangs, fail new operations instead of queuing stale actions.
        if not self.lock.acquire(timeout=1):
            raise RuntimeError("Worker busy; restart worker if SDK request does not terminate")
        try:
            if path == '/session/start':
                checkpoint = payload.get('checkpoint')
                if not isinstance(checkpoint, str) or not checkpoint.strip() or payload.get('max_horizon') != 48 or type(payload.get('invert_grippers')) is not bool:
                    raise ValueError("Invalid session configuration")
                self.close()
                self.policy = self.factory(checkpoint, 48, payload['invert_grippers'])
                self.session_id = sid
                return {'session_id': sid}
            if path == '/session/end':
                if sid == self.session_id:
                    self.close()
                return {'session_id': sid}
            if path != '/infer' or sid != self.session_id or self.policy is None:
                raise ValueError("Unknown route or stale session")
            try:
                actions = self.policy.infer(payload)
                return {'session_id': sid, 'actions': actions.tolist()}
            except BaseException:
                self.close()
                raise
        finally:
            self.lock.release()


def make_server(port, manager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log instructions, frames, SDK errors or credentials.

        def reply(self, code, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'protocol': 'dream-policy-adapter-v1', 'sdk_compatible': True})
            else:
                self.reply(404, {'error': 'unknown route'})

        def do_POST(self):
            self.connection.settimeout(10)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected object')
                result = manager.dispatch(self.path, payload)
            except (ValueError, KeyError, TypeError):
                self.reply(400, {'error': 'invalid request or action data'})
            except Exception:
                self.reply(503, {'error': 'Dream worker failed; session may require restart'})
            else:
                self.reply(200, result)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9120)
    args = parser.parse_args()
    load_sdk()  # Fail before opening the port if the SDK API is incompatible.
    manager = SessionManager()
    server = make_server(args.port, manager)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with manager.lock:
            manager.close()


if __name__ == '__main__':
    main()
