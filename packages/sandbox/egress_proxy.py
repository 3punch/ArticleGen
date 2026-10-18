"""Small allowlist-only HTTP CONNECT proxy used for opt-in sandbox egress.

The sandbox lives on an internal Docker network. This sidecar is the only member
with an additional outbound bridge; it permits only configured HTTP/HTTPS hosts and
only public destination IP addresses, preventing direct egress and common SSRF paths.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
from urllib.parse import urlsplit


DEFAULT_PORTS = {80, 443}
MAX_HEADER_BYTES = 64 * 1024
MAX_HOST_BYTES = 253
MAX_TLS_HANDSHAKE_BYTES = 256 * 1024


def normalize_allowlist(entries: object) -> tuple[str, ...]:
    """Validate and canonicalize exact/wildcard host rules.

    Rules without a port allow only 80 and 443. ``host:port`` rules permit exactly
    that port. IP literals are accepted syntactically but the proxy still refuses
    non-public/reserved addresses at connection time.
    """
    if entries is None:
        return ()
    if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple, set, frozenset)):
        raise ValueError("network allowlist must be a sequence of host rules")
    normalized: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("network allowlist entries must be non-empty strings")
        raw = entry.strip()
        if any(char in raw for char in "\r\n\x00") or "/" in raw or "@" in raw or "?" in raw or "#" in raw:
            raise ValueError(f"invalid network allowlist entry: {entry!r}")
        wildcard = raw.startswith("*.")
        if "*" in raw[2 if wildcard else 0:]:
            raise ValueError("wildcards are only supported as a leading '*.'")
        authority = raw[2:] if wildcard else raw
        try:
            parsed = urlsplit("//" + authority)
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid network allowlist entry: {entry!r}") from exc
        if not host or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError(f"invalid network allowlist entry: {entry!r}")
        host = _normalize_host(host)
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("allowlist ports must be between 1 and 65535")
        if wildcard and _is_ip_literal(host):
            raise ValueError("wildcards cannot be used with IP addresses")
        rendered_host = f"[{host}]" if ":" in host else host
        rule = ("*." if wildcard else "") + rendered_host
        if port is not None:
            rule += f":{port}"
        normalized.add(rule)
    return tuple(sorted(normalized))


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip(".").casefold()
    if not host or len(host) > MAX_HOST_BYTES:
        raise ValueError("invalid network hostname")
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        pass
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("invalid internationalized hostname") from exc
    if len(ascii_host) > MAX_HOST_BYTES:
        raise ValueError("invalid network hostname")
    labels = ascii_host.split(".")
    if any(
        not label
        or len(label) > 63
        or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
        for label in labels
    ):
        raise ValueError("network allowlist entries must be valid hostnames")
    return ascii_host


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _allowed(host: str, port: int, rules: tuple[str, ...]) -> bool:
    try:
        host = _normalize_host(host)
    except ValueError:
        return False
    for rendered in rules:
        wildcard = rendered.startswith("*.")
        authority = rendered[2:] if wildcard else rendered
        try:
            parsed = urlsplit("//" + authority)
            rule_host = _normalize_host(parsed.hostname or "")
            rule_port = parsed.port
        except ValueError:
            continue
        if rule_port is not None:
            if port != rule_port:
                continue
        elif port not in DEFAULT_PORTS:
            continue
        if wildcard:
            if host != rule_host and host.endswith("." + rule_host):
                return True
        elif host == rule_host:
            return True
    return False


def _public_ip(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return parsed.is_global


async def _resolve_public(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses: list[str] = []
        for family, _, _, _, sockaddr in infos:
            address = sockaddr[0]
            if family in (socket.AF_INET, socket.AF_INET6) and _public_ip(address) and address not in addresses:
                addresses.append(address)
        return addresses
    return [literal.compressed] if literal.is_global else []


def _parse_authority(value: str, default_port: int) -> tuple[str, int]:
    parsed = urlsplit("//" + value)
    if parsed.username is not None or parsed.password is not None or parsed.path not in ("", "/"):
        raise ValueError("malformed host authority")
    host = parsed.hostname
    if not host:
        raise ValueError("host is required")
    explicit_port = parsed.port
    port = default_port if explicit_port is None else explicit_port
    if not 1 <= port <= 65535:
        raise ValueError("invalid port")
    return _normalize_host(host), port


def _parse_client_hello_sni(handshake: bytes) -> str | None:
    """Return a TLS ClientHello SNI, or raise for a malformed handshake."""
    if len(handshake) < 4 or handshake[0] != 1:
        raise ValueError("TLS record does not contain a ClientHello")
    message_length = int.from_bytes(handshake[1:4], "big")
    end = 4 + message_length
    if end > len(handshake) or message_length < 34:
        raise ValueError("truncated TLS ClientHello")
    body = memoryview(handshake)[4:end]
    offset = 34  # legacy version + 32-byte random
    if offset >= len(body):
        raise ValueError("truncated TLS session ID")
    session_length = body[offset]
    offset += 1 + session_length
    if offset + 2 > len(body):
        raise ValueError("truncated TLS cipher suites")
    cipher_length = int.from_bytes(body[offset:offset + 2], "big")
    offset += 2 + cipher_length
    if offset >= len(body):
        raise ValueError("truncated TLS compression methods")
    compression_length = body[offset]
    offset += 1 + compression_length
    if offset == len(body):
        return None
    if offset + 2 > len(body):
        raise ValueError("truncated TLS extensions")
    extensions_length = int.from_bytes(body[offset:offset + 2], "big")
    offset += 2
    extensions_end = offset + extensions_length
    if extensions_end > len(body):
        raise ValueError("truncated TLS extension block")
    while offset + 4 <= extensions_end:
        extension_type = int.from_bytes(body[offset:offset + 2], "big")
        extension_length = int.from_bytes(body[offset + 2:offset + 4], "big")
        offset += 4
        extension_end = offset + extension_length
        if extension_end > extensions_end:
            raise ValueError("truncated TLS extension")
        if extension_type == 0xFE0D:
            raise ValueError("Encrypted ClientHello is not supported by the egress allowlist")
        if extension_type == 0:
            if extension_length < 2:
                raise ValueError("invalid TLS server-name extension")
            names_length = int.from_bytes(body[offset:offset + 2], "big")
            name_offset = offset + 2
            names_end = name_offset + names_length
            if names_end > extension_end:
                raise ValueError("truncated TLS server-name list")
            while name_offset + 3 <= names_end:
                name_type = body[name_offset]
                name_length = int.from_bytes(body[name_offset + 1:name_offset + 3], "big")
                name_offset += 3
                name_end = name_offset + name_length
                if name_end > names_end:
                    raise ValueError("truncated TLS server name")
                if name_type == 0:
                    return _normalize_host(bytes(body[name_offset:name_end]).decode("ascii"))
                name_offset = name_end
            return None
        offset = extension_end
    return None


async def _read_client_hello(reader: asyncio.StreamReader) -> tuple[str | None, bytes]:
    """Read enough of the client's first TLS handshake message for SNI checking."""
    first_byte = await reader.readexactly(1)
    if first_byte != b"\x16":
        # Non-TLS CONNECT protocols are still constrained by their CONNECT host/IP.
        return None, first_byte
    prefix = bytearray(first_byte)
    handshake = bytearray()
    record_type = first_byte
    while len(handshake) <= MAX_TLS_HANDSHAKE_BYTES:
        header_tail = await reader.readexactly(4)
        record_header = record_type + header_tail
        prefix.extend(header_tail)
        if record_header[0] != 0x16:
            raise ValueError("unexpected TLS record before ClientHello")
        record_length = int.from_bytes(record_header[3:5], "big")
        if not 1 <= record_length <= MAX_TLS_HANDSHAKE_BYTES:
            raise ValueError("invalid TLS record length")
        payload = await reader.readexactly(record_length)
        prefix.extend(payload)
        handshake.extend(payload)
        if len(handshake) > MAX_TLS_HANDSHAKE_BYTES:
            raise ValueError("TLS ClientHello exceeds the size limit")
        if len(handshake) >= 4:
            if handshake[0] != 1:
                raise ValueError("TLS handshake is not a ClientHello")
            expected_length = 4 + int.from_bytes(handshake[1:4], "big")
            if expected_length > MAX_TLS_HANDSHAKE_BYTES:
                raise ValueError("TLS ClientHello exceeds the size limit")
            if len(handshake) >= expected_length:
                return _parse_client_hello_sni(bytes(handshake[:expected_length])), bytes(prefix)
        record_type = await reader.readexactly(1)
        prefix.extend(record_type)
        if record_type != b"\x16":
            raise ValueError("unexpected TLS record before ClientHello")
    raise ValueError("TLS ClientHello exceeds the size limit")


async def _send_error(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
    body = reason.encode("utf-8", errors="replace")[:256]
    response = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Connection: close\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("ascii", errors="replace") + body
    writer.write(response)
    try:
        await writer.drain()
    except OSError:
        pass


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while True:
        chunk = await reader.read(64 * 1024)
        if not chunk:
            break
        writer.write(chunk)
        await writer.drain()
    try:
        writer.write_eof()
    except (AttributeError, OSError, RuntimeError):
        pass


async def _relay(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter,
                 upstream_reader: asyncio.StreamReader, upstream_writer: asyncio.StreamWriter) -> None:
    tasks = {
        asyncio.create_task(_pipe(client_reader, upstream_writer)),
        asyncio.create_task(_pipe(upstream_reader, client_writer)),
    }
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                         rules: tuple[str, ...]) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        raw_header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
        if len(raw_header) > MAX_HEADER_BYTES:
            await _send_error(writer, 431, "Request Header Fields Too Large")
            return
        lines = raw_header[:-4].split(b"\r\n")
        try:
            method, target, version = lines[0].decode("ascii").split(" ", 2)
            headers = [line.decode("latin-1") for line in lines[1:] if line]
        except (UnicodeDecodeError, ValueError):
            await _send_error(writer, 400, "Bad Request")
            return
        if not version.startswith("HTTP/"):
            await _send_error(writer, 400, "Bad Request")
            return

        is_connect = method.upper() == "CONNECT"
        if is_connect:
            host, port = _parse_authority(target, 443)
            request_target = ""
        else:
            parsed_url = urlsplit(target)
            if parsed_url.scheme:
                if (parsed_url.scheme.casefold() != "http" or not parsed_url.hostname
                        or parsed_url.username is not None or parsed_url.password is not None):
                    await _send_error(writer, 400, "Only HTTP requests or HTTPS CONNECT are supported")
                    return
                host = _normalize_host(parsed_url.hostname)
                explicit_port = parsed_url.port
                port = 80 if explicit_port is None else explicit_port
                if not 1 <= port <= 65535:
                    await _send_error(writer, 400, "Invalid port")
                    return
                path = parsed_url.path or "/"
                request_target = path + (("?" + parsed_url.query) if parsed_url.query else "")
            else:
                host_headers = [value.split(":", 1)[1].strip() for value in headers
                                if value.casefold().startswith("host:")]
                if not host_headers:
                    await _send_error(writer, 400, "Host header is required")
                    return
                host, port = _parse_authority(host_headers[0], 80)
                request_target = target
                if not request_target.startswith("/"):
                    await _send_error(writer, 400, "Invalid request target")
                    return
        if not _allowed(host, port, rules):
            await _send_error(writer, 403, "Destination is not allowlisted")
            return

        addresses = await asyncio.wait_for(_resolve_public(host, port), timeout=5)
        if not addresses:
            await _send_error(writer, 403, "Destination does not resolve to a public address")
            return
        upstream_reader: asyncio.StreamReader | None = None
        for address in addresses:
            try:
                upstream_reader, upstream_writer = await asyncio.wait_for(
                    asyncio.open_connection(address, port), timeout=10,
                )
                break
            except OSError:
                pass
        if upstream_reader is None or upstream_writer is None:
            await _send_error(writer, 502, "Upstream connection failed")
            return

        if is_connect:
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            sni, client_hello = await asyncio.wait_for(_read_client_hello(reader), timeout=10)
            if sni is not None and not _allowed(sni, port, rules):
                # The CONNECT host may be an allowed CDN hostname while the TLS SNI
                # attempts to reach a different virtual host on the same IP.
                return
            upstream_writer.write(client_hello)
            await upstream_writer.drain()
        else:
            forwarded_headers = []
            for header in headers:
                if ":" not in header:
                    await _send_error(writer, 400, "Malformed header")
                    return
                name = header.split(":", 1)[0].strip().casefold()
                if name in {"proxy-authorization", "proxy-connection", "connection", "keep-alive", "host"}:
                    continue
                forwarded_headers.append(header)
            authority_host = f"[{host}]" if ":" in host else host
            host_header = authority_host if port == 80 else f"{authority_host}:{port}"
            outgoing = [f"{method} {request_target} {version}", f"Host: {host_header}",
                        *forwarded_headers, "Connection: close", "", ""]
            upstream_writer.write("\r\n".join(outgoing).encode("latin-1"))
            await upstream_writer.drain()

        await _relay(reader, writer, upstream_reader, upstream_writer)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
        await _send_error(writer, 400, "Incomplete or timed out request")
    except (ValueError, UnicodeError):
        await _send_error(writer, 400, "Bad Request")
    except (OSError, ConnectionError):
        await _send_error(writer, 502, "Upstream connection failed")
    finally:
        if upstream_writer is not None:
            upstream_writer.close()
            try:
                await upstream_writer.wait_closed()
            except OSError:
                pass
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def serve(host: str = "0.0.0.0", port: int = 3128) -> None:
    raw_rules = os.environ.get("PROJECTFORGE_EGRESS_ALLOWLIST", "[]")
    try:
        decoded = json.loads(raw_rules)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid PROJECTFORGE_EGRESS_ALLOWLIST JSON") from exc
    rules = normalize_allowlist(decoded)
    if not rules:
        raise RuntimeError("egress proxy requires a non-empty allowlist")
    server = await asyncio.start_server(
        lambda reader, writer: _handle_client(reader, writer, rules), host, port,
        limit=MAX_HEADER_BYTES + 1,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    port_value = int(os.environ.get("PROJECTFORGE_EGRESS_PORT", "3128"))
    if not 1 <= port_value <= 65535:
        raise SystemExit("PROJECTFORGE_EGRESS_PORT must be between 1 and 65535")
    asyncio.run(serve(port=port_value))


if __name__ == "__main__":
    main()
