"""Unit tests for SI-UE5-MCP-Guard."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si_ue5_mcp_guard as guard  # noqa: E402


class TestSIUE5MCPGuard(unittest.TestCase):
    def test_port_doctor_live_socket(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            self.assertTrue(guard.check_tcp_port("127.0.0.1", port, timeout=0.5))

        self.assertFalse(guard.check_tcp_port("127.0.0.1", port, timeout=0.2))

    def test_merge_mcp_config_preserves_existing_servers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / ".mcp.json"
            cfg.write_text(
                json.dumps({"mcpServers": {"aseprite": {"command": "uv"}}}),
                encoding="utf-8",
            )
            entry = guard.build_mcp_server_entry("native-5.8")
            result = guard.merge_mcp_config(cfg, "unreal-guarded", entry)
            self.assertIn("aseprite", result["mcpServers"])
            self.assertIn("unreal-guarded", result["mcpServers"])
            self.assertEqual(
                result["mcpServers"]["unreal-guarded"]["env"]["UE_MCP_MODE"],
                "native-5.8",
            )

    def test_blocks_protected_actor_deletion(self) -> None:
        decision = guard.evaluate_tool_call(
            "delete_actor",
            {"actor_name": "PlayerStart_0"},
        )
        self.assertFalse(decision.allowed)
        self.assertIn("playerstart", decision.reason.lower())

    def test_blocks_dangerous_console_command(self) -> None:
        decision = guard.evaluate_tool_call(
            "execute_console_command",
            {"command": "quit"},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.severity, "block")

    def test_blocks_runaway_spawn_count(self) -> None:
        decision = guard.evaluate_tool_call(
            "spawn_actor",
            {"type": "StaticMeshActor", "count": 150},
            max_spawn_batch=25,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("150", decision.reason)

    def test_allows_safe_blueprint_call_and_logs_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            rpc_line = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {
                        "name": "add_blueprint_node",
                        "arguments": {"blueprint_name": "BP_Hero", "node_type": "Branch"},
                    },
                }
            )
            forward, err = guard.inspect_jsonrpc_message(
                rpc_line,
                audit_path=audit_path,
            )
            self.assertTrue(forward)
            self.assertIsNone(err)
            lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            logged = json.loads(lines[0])
            self.assertTrue(logged["allowed"])
            self.assertEqual(logged["tool"], "add_blueprint_node")

    def test_blocked_jsonrpc_returns_valid_error_packet(self) -> None:
        rpc_line = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "delete_all_actors",
                    "arguments": {},
                },
            }
        )
        forward, err = guard.inspect_jsonrpc_message(rpc_line)
        self.assertFalse(forward)
        self.assertIsNotNone(err)
        parsed = json.loads(err or "{}")
        self.assertEqual(parsed["id"], 42)
        self.assertEqual(parsed["error"]["code"], -32001)


if __name__ == "__main__":
    unittest.main()
