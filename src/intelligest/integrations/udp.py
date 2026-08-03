from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UDPActionConfig:
    host: str
    port: int
    broadcast: bool
    minimum_stable_seconds: float
    minimum_confidence: float
    class_payloads: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> UDPActionConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        payloads = {str(key): str(value) for key, value in data["class_payloads"].items()}
        if any(not value for value in payloads.values()):
            raise ValueError("Los payloads UDP no pueden estar vacíos")
        return cls(
            host=str(data["host"]),
            port=int(data["port"]),
            broadcast=bool(data.get("broadcast", False)),
            minimum_stable_seconds=float(data.get("minimum_stable_seconds", 0)),
            minimum_confidence=float(data.get("minimum_confidence", 0)),
            class_payloads=payloads,
        )

    def payload_for(self, class_name: str) -> bytes:
        try:
            return self.class_payloads[class_name].encode("utf-8")
        except KeyError as exc:
            raise ValueError(f"No hay acción UDP para la clase {class_name}") from exc


def send_action(config: UDPActionConfig, class_name: str, timeout: float = 1.0) -> None:
    payload = config.payload_for(class_name)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        if config.broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(payload, (config.host, config.port))
