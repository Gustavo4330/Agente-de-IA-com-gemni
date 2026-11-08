"""Limite de requisições por IP, guardado em memória.

Funciona bem com uma instância só. Com várias réplicas, o contador teria
que ir para um lugar compartilhado (ex.: Redis).
"""

import time


class RateLimiter:
    """Permite até `max_requests` por IP a cada `window_seconds`. Com 0, não limita nada."""

    def __init__(self, max_requests: int, window_seconds: float, clock=time.monotonic):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self.windows: dict[str, tuple[float, int]] = {}  # ip -> (início da janela, contagem)

    def allow(self, ip: str) -> bool:
        if self.max_requests == 0:
            return True

        now = self.clock()
        start, count = self.windows.get(ip, (now, 0))
        if now - start >= self.window_seconds:
            start, count = now, 0  # janela expirou, recomeça
        if count >= self.max_requests:
            return False

        self.windows[ip] = (start, count + 1)
        if len(self.windows) > 10_000:
            self._forget_old_ips(now)
        return True

    def _forget_old_ips(self, now: float) -> None:
        """Evita que o dicionário cresça para sempre com IPs que não voltaram."""
        self.windows = {ip: w for ip, w in self.windows.items() if now - w[0] < self.window_seconds}
