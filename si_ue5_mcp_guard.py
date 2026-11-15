"""SI-UE5-MCP-Guard: Pre-flight validator and JSON-RPC safety proxy for Unreal Engine 5 MCP.

Solves three common failures when connecting Claude Code or local AI models to UE5:
1. Validates or generates `.mcp.json` / client config for UE 5.8 native MCP or UE 5.3-5.7 TCP bridge.
2. Verifies that the UE5 Editor MCP listener port is reachable before AI tool execution.
3. Filters JSON-RPC `tools/call` payloads in proxy mode to prevent destructive editor commands,
   protected actor deletion, or runaway actor spawn loops, while logging an audit trail.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 55557
DEFAULT_NATIVE_PORT = 30010
DEFAULT_MAX_SPAWN_BATCH = 25

PROTECTED_ACTOR_TOKENS = (
    "playerstart",
    "gamemode",
    "directionallight",
    "skylight",
    "worldsettings",
    "landscape",
)

DANGEROUS_CONSOLE_PATTERNS = (
    re.compile(r"\bquit\b", re.IGNORECASE),
    re.compile(r"\bexit\b", re.IGNORECASE),
    re.compile(r"\bdebug\s+crash\b", re.IGNORECASE),
    re.compile(r"\bforcecrash\b", re.IGNORECASE),
    re.compile(r"\bopen\s+/", re.IGNORECASE),
)

DESTRUCTIVE_TOOL_TOKENS = (
    "delete_all",
    "destroy_all",
    "clear_level",
    "wipe_scene",
    "format_",
)


@dataclass(frozen=True)
class GuardDecision:
    """Result of evaluating a single MCP tool call payload."""

    allowed: bool
    reason: str
    tool_name: str
    severity: str = "allow"


def check_tcp_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP listener is accepting connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_mcp_server_entry(
    mode: str,
    bridge_script: str | None = None,
    python_executable: str = "python",
    host: str = DEFAULT_BRIDGE_HOST,
    port: int | None = None,
) -> dict[str, Any]:
    """Build a `.mcp.json` `mcpServers` entry for UE 5.8 native or UE 5.3-5.7 bridge."""
    normalized = mode.strip().lower()
    if normalized == "native-5.8":
        target_port = port or DEFAULT_NATIVE_PORT
        return {
            "command": python_executable,
            "args": [
                str(Path(__file__).resolve()),
                "proxy",
                "--host",
                host,
                "--port",
                str(target_port),
                "--skip-port-check",
                "--",
                python_executable,
                "-m",
                "unreal_mcp_native_bridge",
            ],
            "env": {
                "UE_MCP_MODE": "native-5.8",
                "UE_MCP_HOST": host,
                "UE_MCP_PORT": str(target_port),
            },
        }

    if normalized == "bridge-5.7":
        target_port = port or DEFAULT_BRIDGE_PORT
        script_path = bridge_script or "unreal_mcp_server.py"
        return {
            "command": python_executable,
            "args": [
                str(Path(__file__).resolve()),
                "proxy",
                "--host",
                host,
                "--port",
                str(target_port),
                "--",
                python_executable,
                script_path,
            ],
            "env": {
                "UE_MCP_MODE": "bridge-5.7",
                "UE_MCP_HOST": host,
                "UE_MCP_PORT": str(target_port),
            },
        }

    raise ValueError(f"Unsupported UE5 MCP mode: {mode!r}. Use 'native-5.8' or 'bridge-5.7'.")


def merge_mcp_config(
    config_path: Path,
    server_name: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Insert or update `server_name` inside a `.mcp.json` file without dropping other servers."""
    data: dict[str, Any] = {}
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8").strip()
        if raw:
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected JSON object in {config_path}")
            data = loaded

    servers = data.get("mcpServers")
    if servers is None:
        servers = {}
        data["mcpServers"] = servers
    elif not isinstance(servers, dict):
        raise ValueError(f"Expected 'mcpServers' object in {config_path}")

    servers[server_name] = entry
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def _extract_numeric_count(arguments: dict[str, Any]) -> int | None:
    """Return integer spawn/batch count if provided in common UE5 MCP argument keys."""
    for key in ("count", "num_actors", "batch_size", "iterations"):
        val = arguments.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    actors = arguments.get("actors")
    if isinstance(actors, list):
        return len(actors)
    return None


def evaluate_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    allow_destructive: bool = False,
    max_spawn_batch: int = DEFAULT_MAX_SPAWN_BATCH,
) -> GuardDecision:
    """Inspect an MCP `tools/call` target and decide whether it is safe for Unreal Editor."""
    args = arguments or {}
    lower_name = tool_name.strip().lower()

    if not allow_destructive and any(tok in lower_name for tok in DESTRUCTIVE_TOOL_TOKENS):
        return GuardDecision(
            allowed=False,
            reason=f"Blocked bulk destructive UE5 tool '{tool_name}'. Pass --allow-destructive to override.",
            tool_name=tool_name,
            severity="block",
        )

    if "delete" in lower_name or "destroy" in lower_name or "remove_actor" in lower_name:
        target = str(
            args.get("actor_name")
            or args.get("name")
            or args.get("target")
            or ""
        ).lower()
        for protected in PROTECTED_ACTOR_TOKENS:
            if protected in target and not allow_destructive:
                return GuardDecision(
                    allowed=False,
                    reason=(
                        f"Blocked deletion of protected UE5 actor '{target}' "
                        f"(matched '{protected}')."
                    ),
                    tool_name=tool_name,
                    severity="block",
                )

    if "console" in lower_name or "exec" in lower_name:
        command_text = str(args.get("command") or args.get("cmd") or "")
        for pattern in DANGEROUS_CONSOLE_PATTERNS:
            if pattern.search(command_text) and not allow_destructive:
                return GuardDecision(
                    allowed=False,
                    reason=f"Blocked unsafe UE5 console command: {command_text!r}",
                    tool_name=tool_name,
                    severity="block",
                )

    if "spawn" in lower_name or "create_actor" in lower_name or "batch" in lower_name:
        count = _extract_numeric_count(args)
        if count is not None and count > max_spawn_batch:
            return GuardDecision(
                allowed=False,
                reason=(
                    f"Blocked runaway spawn count ({count} > max_spawn_batch={max_spawn_batch}) "
                    f"in tool '{tool_name}'."
                ),
                tool_name=tool_name,
                severity="block",
            )

    return GuardDecision(
        allowed=True,
        reason="Passed SI-UE5-MCP-Guard policy.",
        tool_name=tool_name,
        severity="allow",
    )


def append_audit_log(
    audit_path: Path | None,
    decision: GuardDecision,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Append a structured JSONL audit event if audit_path is configured."""
    if audit_path is None:
        return
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tool": decision.tool_name,
        "allowed": decision.allowed,
        "severity": decision.severity,
        "reason": decision.reason,
        "arg_keys": sorted((arguments or {}).keys()),
    }
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def inspect_jsonrpc_message(
    raw_line: str,
    *,
    allow_destructive: bool = False,
    max_spawn_batch: int = DEFAULT_MAX_SPAWN_BATCH,
    audit_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Inspect one JSON-RPC line. Return (forward_to_upstream, error_response_line)."""
    stripped = raw_line.strip()
    if not stripped:
        return False, None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return True, None

    if not isinstance(payload, dict) or payload.get("method") != "tools/call":
        return True, None

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return True, None

    tool_name = str(params.get("name") or "")
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    decision = evaluate_tool_call(
        tool_name,
        arguments,
        allow_destructive=allow_destructive,
        max_spawn_batch=max_spawn_batch,
    )
    append_audit_log(audit_path, decision, arguments)

    if decision.allowed:
        return True, None

    req_id = payload.get("id")
    if req_id is None:
        return False, None

    error_obj = {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32001,
            "message": decision.reason,
        },
    }
    return False, json.dumps(error_obj) + "\n"


def run_proxy(
    target_cmd: list[str],
    *,
    host: str = DEFAULT_BRIDGE_HOST,
    port: int = DEFAULT_BRIDGE_PORT,
    skip_port_check: bool = False,
    allow_destructive: bool = False,
    max_spawn_batch: int = DEFAULT_MAX_SPAWN_BATCH,
    audit_path: Path | None = Path("si_ue5_mcp_audit.jsonl"),
) -> int:
    """Spawn the upstream MCP server and filter stdin JSON-RPC requests in real time."""
    if not target_cmd:
        raise ValueError("Missing upstream command after `--` for proxy mode.")

    if not skip_port_check and not check_tcp_port(host, port, timeout=1.0):
        sys.stderr.write(
            f"[SI-UE5-MCP-Guard] Warning: Unreal Editor MCP listener not detected on {host}:{port}. "
            "Start Unreal Editor (or run ModelContextProtocol.StartServer in UE 5.8) if tool calls fail.\n"
        )
        sys.stderr.flush()

    proc = subprocess.Popen(
        target_cmd,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    def _pump_stdin() -> None:
        assert proc.stdin is not None
        try:
            for line in sys.stdin:
                forward, err_line = inspect_jsonrpc_message(
                    line,
                    allow_destructive=allow_destructive,
                    max_spawn_batch=max_spawn_batch,
                    audit_path=audit_path,
                )
                if err_line is not None:
                    sys.stdout.write(err_line)
                    sys.stdout.flush()
                elif forward:
                    proc.stdin.write(line)
                    proc.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    reader = threading.Thread(target=_pump_stdin, daemon=True)
    reader.start()
    return proc.wait()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for doctor, init-config, inspect-call, and proxy."""
    parser = argparse.ArgumentParser(
        prog="si_ue5_mcp_guard",
        description="Safety firewall, port doctor, and .mcp.json generator for Unreal Engine 5 MCP.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check whether UE5 MCP TCP listener is reachable.")
    p_doc.add_argument("--host", default=DEFAULT_BRIDGE_HOST)
    p_doc.add_argument("--port", type=int, default=DEFAULT_BRIDGE_PORT)
    p_doc.add_argument("--timeout", type=float, default=1.0)

    p_init = sub.add_parser("init-config", help="Create or update guarded `.mcp.json` for UE5.")
    p_init.add_argument("--mode", choices=("native-5.8", "bridge-5.7"), default="native-5.8")
    p_init.add_argument("--config", type=Path, default=Path(".mcp.json"))
    p_init.add_argument("--server-name", default="unreal-engine-guarded")
    p_init.add_argument("--bridge-script", default=None)
    p_init.add_argument("--python", default=sys.executable)
    p_init.add_argument("--host", default=DEFAULT_BRIDGE_HOST)
    p_init.add_argument("--port", type=int, default=None)

    p_check = sub.add_parser("inspect-call", help="Evaluate one JSON-RPC line or payload.")
    p_check.add_argument("--json", required=True, help="Raw JSON-RPC string or JSON object.")
    p_check.add_argument("--max-spawn-batch", type=int, default=DEFAULT_MAX_SPAWN_BATCH)
    p_check.add_argument("--allow-destructive", action="store_true")

    p_proxy = sub.add_parser("proxy", help="Wrap upstream UE5 MCP stdio server with guardrails.")
    p_proxy.add_argument("--host", default=DEFAULT_BRIDGE_HOST)
    p_proxy.add_argument("--port", type=int, default=DEFAULT_BRIDGE_PORT)
    p_proxy.add_argument("--skip-port-check", action="store_true")
    p_proxy.add_argument("--allow-destructive", action="store_true")
    p_proxy.add_argument("--max-spawn-batch", type=int, default=DEFAULT_MAX_SPAWN_BATCH)
    p_proxy.add_argument("--audit-log", type=Path, default=Path("si_ue5_mcp_audit.jsonl"))
    p_proxy.add_argument("target", nargs=argparse.REMAINDER)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "doctor":
        ok = check_tcp_port(args.host, args.port, timeout=args.timeout)
        status = {
            "host": args.host,
            "port": args.port,
            "reachable": ok,
            "status": "READY" if ok else "OFFLINE",
        }
        print(json.dumps(status, indent=2))
        return 0 if ok else 1

    if args.command == "init-config":
        entry = build_mcp_server_entry(
            mode=args.mode,
            bridge_script=args.bridge_script,
            python_executable=args.python,
            host=args.host,
            port=args.port,
        )
        updated = merge_mcp_config(args.config, args.server_name, entry)
        print(json.dumps({"written": str(args.config), "servers": list(updated["mcpServers"].keys())}, indent=2))
        return 0

    if args.command == "inspect-call":
        forward, err_line = inspect_jsonrpc_message(
            args.json,
            allow_destructive=args.allow_destructive,
            max_spawn_batch=args.max_spawn_batch,
        )
        print(json.dumps({"allowed": forward, "error_response": json.loads(err_line) if err_line else None}, indent=2))
        return 0 if forward else 2

    if args.command == "proxy":
        cmd = list(args.target)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        return run_proxy(
            cmd,
            host=args.host,
            port=args.port,
            skip_port_check=args.skip_port_check,
            allow_destructive=args.allow_destructive,
            max_spawn_batch=args.max_spawn_batch,
            audit_path=args.audit_log,
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
