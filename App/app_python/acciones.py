#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
acciones.py — Emisor UDP (broadcast por defecto).
"""

import socket

def send_udp(letter: str, ip: str = "255.255.255.255", port: int = 1097, timeout: float = 1.0) -> None:
    """
    Envía 'letter' (str de 1 carácter) como paquete UDP a ip:port.
    Por defecto usa broadcast (255.255.255.255:1097).
    """
    if not isinstance(letter, str) or len(letter) != 1:
        raise ValueError("letter debe ser un string de 1 carácter (ej. 'a').")

    data = letter.encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        # habilitar broadcast si es dirección de broadcast
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        # no es necesario bind local explícito
        sock.sendto(data, (ip, port))
    finally:
        sock.close()
