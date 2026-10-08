from servemind.mcp.tool_manager import Tool, ToolManager


def test_tool_scope_is_enforced_before_handler():
    calls = []
    manager = ToolManager()
    manager.register(Tool("read_order", "read", lambda params, context: calls.append(1)))
    denied = manager.call("read_order", {}, {"allowed_tools": {"other_tool"}})
    assert denied.success is False
    assert denied.error == "forbidden:tool_scope"
    assert calls == []
    allowed = manager.call("read_order", {}, {"allowed_tools": {"read_order"}})
    assert allowed.success is True
    assert calls == [1]
    assert "params" not in manager.traces[-1]
