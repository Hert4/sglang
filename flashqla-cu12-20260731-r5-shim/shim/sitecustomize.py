# Strip prefix ingress của run.ai để sglang (bản mới) hứng đúng path.
# Đặt file này vào PYTHONPATH của image -> chạy lúc Python khởi động, tự vá FastAPI
# TRƯỚC khi sglang tạo app. Không sửa source sglang. Bật bằng env RUNAI_PATH_PREFIX.
import os

_PREFIX = os.environ.get("RUNAI_PATH_PREFIX", "").rstrip("/")

if _PREFIX:
    import fastapi

    class _StripPrefix:
        """Pure-ASGI middleware: bỏ prefix nếu request có, để nguyên bare-path (probe run.ai)."""
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("type") in ("http", "websocket"):
                path = scope.get("path", "")
                if path.startswith(_PREFIX):
                    scope = dict(scope)
                    scope["path"] = path[len(_PREFIX):] or "/"
                    raw = scope.get("raw_path")
                    pb = _PREFIX.encode()
                    if isinstance(raw, (bytes, bytearray)) and raw.startswith(pb):
                        scope["raw_path"] = raw[len(pb):] or b"/"
            await self.app(scope, receive, send)

    _orig_init = fastapi.FastAPI.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.add_middleware(_StripPrefix)

    fastapi.FastAPI.__init__ = _patched_init
