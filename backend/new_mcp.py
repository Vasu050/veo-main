import asyncio
import base64
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

log = logging.getLogger(__name__)

MCP_URL = os.getenv(
    "ATLASSIAN_MCP_URL",
    "https://mcp.atlassian.com/v2/mcp?tools=all"
).strip('"').strip("'")

DEFAULT_CLOUD_ID = os.getenv(
    "ATLASSIAN_CLOUD_ID",
    "ef6bdfdf-7960-46f9-9a35-47450de6247c"
).strip('"').strip("'")

_TOOLS_CACHE: Optional[List[Any]] = None
_TOOLS_CACHE_TIME: float = 0.0
_TOOLS_CACHE_TTL_SECONDS: float = 3600.0  # 1 hour
_RESOLVED_CLOUD_ID: Optional[str] = None


def _auth_header() -> str:
    """Generate Basic Auth header using email and API token from .env."""
    username = (
        os.getenv("ATLASSIAN_USERNAME")
        or os.getenv("JIRA_EMAIL")
        or "vasu.j@prodapt.com"
    ).strip('"').strip("'")

    token = (
        os.getenv("ATLASSIAN_MCP_TOKEN")
        or os.getenv("JIRA_API_TOKEN")
        or ""
    ).strip('"').strip("'")

    if not token:
        raise RuntimeError("ATLASSIAN_MCP_TOKEN or JIRA_API_TOKEN must be set in .env")

    raw = f"{username}:{token}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _authorization_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Return authorization headers. Uses Bearer token if valid, otherwise falls back to permanent Basic Auth."""
    if token and isinstance(token, str) and token.strip() and not token.strip().lower().startswith("bearer none"):
        clean = token.strip()
        if clean.lower().startswith("bearer "):
            clean = clean[7:].strip()
        if clean:
            return {"Authorization": f"Bearer {clean}"}
    return {"Authorization": _auth_header()}



def _extract_text(tool_result: Any) -> str:
    """Extract plain text from MCP ToolResult."""
    parts = []
    for item in getattr(tool_result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


@asynccontextmanager
async def _session(headers: Optional[Dict[str, str]] = None):
    """Context manager for MCP ClientSession with safe stream unpacking."""
    auth_headers = headers or _authorization_headers()
    async with httpx.AsyncClient(headers=auth_headers, timeout=60.0) as http_client:
        # streamable_http_client returns a 3-element tuple; take streams[0] and streams[1]
        async with streamable_http_client(MCP_URL, http_client=http_client) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                log.info("MCP session initialized")
                yield session


async def _resolve_cloud_id(session: ClientSession, default_cloud_id: Optional[str] = None) -> str:
    """Resolve Jira cloud ID from environment or getAccessibleAtlassianResources."""
    global _RESOLVED_CLOUD_ID
    if default_cloud_id:
        return default_cloud_id
    if DEFAULT_CLOUD_ID:
        return DEFAULT_CLOUD_ID
    if _RESOLVED_CLOUD_ID:
        return _RESOLVED_CLOUD_ID
    try:
        res = await session.call_tool("getAccessibleAtlassianResources", {})
        data = json.loads(_extract_text(res))
        if isinstance(data, dict) and "data" in data and "resources" in data["data"]:
            resources = data["data"]["resources"]
            if resources:
                _RESOLVED_CLOUD_ID = resources[0].get("cloudId", "") or resources[0].get("id", "")
                return _RESOLVED_CLOUD_ID
        elif isinstance(data, list) and data:
            _RESOLVED_CLOUD_ID = data[0].get("id", "") or data[0].get("cloudId", "")
            return _RESOLVED_CLOUD_ID
    except Exception as e:
        log.warning("Could not parse cloud id: %s", e)
    return DEFAULT_CLOUD_ID


async def _list_all_tools(session: ClientSession) -> List[Any]:
    result = await session.list_tools()
    tools = list(result.tools)
    cursor = getattr(result, "nextCursor", getattr(result, "next_cursor", None))
    seen_cursors = set()

    while cursor:
        if cursor in seen_cursors:
            raise RuntimeError("MCP tool listing returned a repeated pagination cursor.")
        seen_cursors.add(cursor)
        try:
            result = await session.list_tools(cursor=cursor)
        except TypeError:
            result = await session.list_tools(
                params=PaginatedRequestParams(cursor=cursor)
            )
        tools.extend(result.tools)
        cursor = getattr(result, "nextCursor", getattr(result, "next_cursor", None))

    return tools


async def _get_all_tools(session: ClientSession, force_refresh: bool = False) -> List[Any]:
    """Retrieve all tools with in-memory TTL caching to avoid repeated MCP streaming scans."""
    global _TOOLS_CACHE, _TOOLS_CACHE_TIME
    now = time.monotonic()
    if not force_refresh and _TOOLS_CACHE is not None and (now - _TOOLS_CACHE_TIME) < _TOOLS_CACHE_TTL_SECONDS:
        return _TOOLS_CACHE
    tools = await _list_all_tools(session)
    _TOOLS_CACHE = tools
    _TOOLS_CACHE_TIME = now
    return tools


async def list_available_mcp_tools(
    access_token: Optional[str] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Return the tools currently exposed by the configured MCP server, using cache if fresh."""
    global _TOOLS_CACHE, _TOOLS_CACHE_TIME
    now = time.monotonic()
    if not force_refresh and _TOOLS_CACHE is not None and (now - _TOOLS_CACHE_TIME) < _TOOLS_CACHE_TTL_SECONDS:
        tools = _TOOLS_CACHE
    else:
        async with _session(headers=_authorization_headers(access_token)) as session:
            tools = await _get_all_tools(session, force_refresh=force_refresh)

    return [
        {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "inputSchema": getattr(mcp_tool, "inputSchema", getattr(mcp_tool, "input_schema", {})),
            "annotations": {
                "readOnlyHint": (
                    getattr(mcp_tool.annotations, "readOnlyHint", getattr(mcp_tool.annotations, "read_only_hint", None))
                    if getattr(mcp_tool, "annotations", None)
                    else None
                ),
                "destructiveHint": (
                    getattr(mcp_tool.annotations, "destructiveHint", getattr(mcp_tool.annotations, "destructive_hint", None))
                    if getattr(mcp_tool, "annotations", None)
                    else None
                ),
            },
        }
        for mcp_tool in tools
    ]


@dataclass(frozen=True)
class MCPToolCallResult:
    name: str
    content: Any = None
    error: Optional[str] = None


async def call_mcp_tools(
    tool_calls: List[Dict[str, Any]],
    access_token: Optional[str] = None,
    cloud_id: Optional[str] = None,
) -> List[MCPToolCallResult]:
    """Execute selected MCP tools concurrently and preserve the call order."""
    if not tool_calls:
        return []

    async with _session(headers=_authorization_headers(access_token)) as session:
        tools_list = await _get_all_tools(session)
        available_tools = {
            item.name: item for item in tools_list
        }
        validated_calls = []

        for tool_call in tool_calls:
            name = tool_call.get("name")
            arguments = tool_call.get("args", {})
            if name not in available_tools:
                raise ValueError(f"LLM selected unavailable MCP tool: {name!r}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments for MCP tool {name!r} must be an object.")
            validated_calls.append((name, dict(arguments), available_tools[name]))

        resolved_cloud_id = await _resolve_cloud_id(
            session,
            cloud_id or DEFAULT_CLOUD_ID,
        )

        async def invoke(
            name: str,
            arguments: Dict[str, Any],
            mcp_tool: Any,
        ) -> MCPToolCallResult:
            annotations = getattr(mcp_tool, "annotations", None)
            read_only = getattr(annotations, "readOnlyHint", getattr(annotations, "read_only_hint", None)) if annotations else None
            destructive = getattr(annotations, "destructiveHint", getattr(annotations, "destructive_hint", None)) if annotations else None
            if (
                annotations is None
                or read_only is not True
                or destructive is True
            ):
                return MCPToolCallResult(
                    name=name,
                    error=(
                        "Blocked: this video workflow only executes MCP tools "
                        "explicitly annotated as read-only."
                    ),
                )
            input_schema = getattr(mcp_tool, "inputSchema", getattr(mcp_tool, "input_schema", {}))
            properties = (input_schema or {}).get("properties", {})
            if "cloudId" in properties:
                arguments["cloudId"] = resolved_cloud_id
            try:
                result = await session.call_tool(name, arguments)
                text = _extract_text(result)
                is_error = getattr(result, "isError", getattr(result, "is_error", False))
                if is_error:
                    raise RuntimeError(text or f"MCP tool {name!r} returned an error.")
                structured_content = getattr(result, "structuredContent", getattr(result, "structured_content", None))
                if structured_content is not None:
                    content = structured_content
                else:
                    try:
                        content = json.loads(text)
                    except json.JSONDecodeError:
                        content = text
                return MCPToolCallResult(name=name, content=content)
            except Exception as exc:
                error = format_mcp_error(exc)
                log.warning("MCP tool %s failed: %s", name, error)
                return MCPToolCallResult(name=name, error=error)

        return await asyncio.gather(
            *(invoke(name, arguments, mcp_tool) for name, arguments, mcp_tool in validated_calls)
        )


async def fetch_issues(keys: List[str], access_token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch Jira issues by key with complete details (view='full')."""
    if not keys:
        return {}

    headers = _authorization_headers(access_token)
    out: Dict[str, Any] = {}
    async with _session(headers=headers) as session:
        cloud_id = await _resolve_cloud_id(session)
        for key in keys:
            try:
                res = await session.call_tool(
                    "getJiraIssue",
                    {"cloudId": cloud_id, "issueIdOrKey": key, "view": "full"},
                )
                text = _extract_text(res)
                try:
                    out[key] = json.loads(text)
                except json.JSONDecodeError:
                    out[key] = {"raw": text}
            except Exception as e:
                log.exception("Failed to fetch Jira issue %s: %s", key, e)
                out[key] = {"error": str(e)}
    return out


def _jira_description_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_jira_description_text(item) for item in value)
    if not isinstance(value, dict):
        return ""
    if value.get("type") == "text":
        return value.get("text", "")
    if value.get("type") == "hardBreak":
        return "\n"

    text = _jira_description_text(value.get("content", []))
    if value.get("type") in {"paragraph", "heading", "codeBlock", "listItem"}:
        return f"{text}\n"
    return text


def format_single_issue_markdown(issue_data: Dict[str, Any]) -> str:
    """Format a single Jira issue into clear Markdown with acceptance criteria, linked items, and attachments."""
    fields = issue_data.get("fields", {})
    key = issue_data.get("key", "Unknown")
    summary = fields.get("summary", "No summary")
    issue_type = fields.get("issuetype", {}).get("name", "Issue")
    status = fields.get("status", {}).get("name", "Unknown")
    priority = fields.get("priority", {}).get("name", "Normal")

    lines = [
        f"### Jira Issue: {key} [{issue_type}] - {summary}",
        f"- **Status**: {status} | **Priority**: {priority}",
    ]

    desc = fields.get("description")
    if isinstance(desc, (dict, list)):
        description_text = _jira_description_text(desc).strip()
        if not description_text and desc:
            description_text = json.dumps(desc, ensure_ascii=False, indent=2)
    else:
        description_text = str(desc).strip() if desc else ""
    if description_text:
        lines.append(
            f"- **Description & Acceptance Criteria**:\n{description_text}"
        )

    links = fields.get("issuelinks", [])
    if links:
        link_lines = []
        for link in links:
            in_issue = link.get("inwardIssue")
            out_issue = link.get("outwardIssue")
            link_type = link.get("type", {})
            if in_issue:
                rel = link_type.get("inward", "linked to")
                ikey = in_issue.get("key", "")
                isum = in_issue.get("fields", {}).get("summary", "")
                istat = in_issue.get("fields", {}).get("status", {}).get("name", "")
                link_lines.append(f"  - [{ikey}] ({rel}): {isum} [{istat}]")
            if out_issue:
                rel = link_type.get("outward", "links to")
                okey = out_issue.get("key", "")
                osum = out_issue.get("fields", {}).get("summary", "")
                ostat = out_issue.get("fields", {}).get("status", {}).get("name", "")
                link_lines.append(f"  - [{okey}] ({rel}): {osum} [{ostat}]")
        if link_lines:
            lines.append("- **Linked Work Items**:\n" + "\n".join(link_lines))

    attachments = fields.get("attachment", [])
    if attachments:
        att_names = [a.get("filename") for a in attachments if isinstance(a, dict) and a.get("filename")]
        if att_names:
            lines.append(f"- **Attachments**: {', '.join(att_names)}")

    return "\n".join(lines)


def format_jira_context_markdown(raw_result: Any) -> str:
    """Format Jira MCP output (single issue, JQL search results, or projects) into rich, clean Markdown."""
    if isinstance(raw_result, str):
        try:
            raw_result = json.loads(raw_result)
        except Exception:
            return raw_result

    if not isinstance(raw_result, dict):
        return str(raw_result)

    # 1. Single issue
    data = raw_result.get("data", raw_result)
    if isinstance(data, dict) and "fields" in data and "key" in data:
        return format_single_issue_markdown(data)

    # 2. List of issues from searchJiraIssuesUsingJql
    issues = []
    if isinstance(data, dict) and "issues" in data and isinstance(data["issues"], list):
        issues = data["issues"]
    elif isinstance(raw_result.get("issues"), list):
        issues = raw_result["issues"]

    if issues:
        # Prioritize core stories, epics, tasks, and bugs over raw test cases/subtasks
        primary_issues = []
        secondary_issues = []
        for iss in issues:
            itype = iss.get("fields", {}).get("issuetype", {}).get("name", "").lower()
            if itype in ("story", "epic", "task", "bug"):
                primary_issues.append(iss)
            else:
                secondary_issues.append(iss)

        ordered_issues = primary_issues + secondary_issues
        parts = [format_single_issue_markdown(iss) for iss in ordered_issues]
        return "\n\n" + ("\n\n" + "=" * 40 + "\n\n").join(parts)

    # 3. Project list
    projects = []
    if isinstance(data, dict) and "projects" in data and isinstance(data["projects"], list):
        projects = data["projects"]
    elif isinstance(data, list):
        projects = data

    if projects:
        lines = ["### Jira Projects:"]
        for p in projects:
            if isinstance(p, dict):
                pkey = p.get("key") or p.get("id", "")
                pname = p.get("name", "")
                lines.append(f"- **{pkey}**: {pname}")
        return "\n".join(lines)

    return json.dumps(raw_result, indent=2)


def format_mcp_error(exc: Exception) -> str:
    """Extract human-readable explanation from MCP and network exceptions (unwrapping TaskGroups)."""
    cur = exc
    while hasattr(cur, "exceptions") and cur.exceptions:
        cur = cur.exceptions[0]
    error = getattr(cur, "error", None)
    if error is not None:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None) or str(cur)
        err_str = f"{type(cur).__name__} (code {code}): {message}"
    else:
        err_str = f"{type(cur).__name__}: {cur}"
    
    if "ConnectError" in err_str or "All connection attempts failed" in err_str or "timed out" in err_str.lower():
        return (
            "Network Connection Timeout: Unable to connect to Atlassian MCP (https://mcp.atlassian.com). "
            "Please ensure your internet connection is active and verify if your corporate VPN (e.g. Prodapt VPN) or network proxy is connected."
        )
    if "401" in err_str or "Unauthorized" in err_str:
        return "Authentication Failed (401 Unauthorized): Please check your JIRA_EMAIL and JIRA_API_TOKEN in .env."
    if "403" in err_str or "Forbidden" in err_str:
        return "Permission Denied (403 Forbidden): Your account does not have access to this Jira Cloud resource."
    return err_str


async def search_jira_context(
    query: str = "",
    access_token: Optional[str] = None,
    cloud_id: Optional[str] = None,
) -> Any:
    """Intelligent Jira MCP search: detects issue keys, executes JQL queries with full view, or lists projects."""
    target_cloud_id = cloud_id or DEFAULT_CLOUD_ID
    headers = _authorization_headers(access_token)

    async def _run(active_headers: Dict[str, str]) -> Any:
        async with _session(headers=active_headers) as session:
            cid = await _resolve_cloud_id(session, target_cloud_id)
            query_str = (query or "").strip()
            issue_match = re.search(r"\b([A-Z][A-Z0-9]+-\d+)\b", query_str)

            # Check if this is a direct single issue key e.g. "ST-1" or "SCRUM-5"
            if issue_match and len(query_str.split()) == 1:
                issue_key = issue_match.group(1)
                log.info("Fetching single Jira issue: %s with view='full'", issue_key)
                res = await session.call_tool(
                    "getJiraIssue",
                    {"cloudId": cid, "issueIdOrKey": issue_key, "view": "full"},
                )
            elif query_str and query_str.lower() != "projects":
                # Check if query_str is already JQL syntax
                jql_indicators = ("~", " AND ", " OR ", "order by", "ORDER BY", "issuetype", "summary ~", "text ~", "status =", "project =")
                is_jql = any(ind in query_str for ind in jql_indicators)

                if is_jql:
                    jql = query_str
                else:
                    # Multi-keyword query: match keywords broadly across summary or text
                    clean_words = [
                        w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", query_str).split()
                        if len(w) > 2 and w.lower() not in {"and", "the", "for", "with", "via", "from", "into", "taking", "account", "items"}
                    ]
                    if clean_words:
                        clauses = [f'(summary ~ "{w}" OR text ~ "{w}")' for w in clean_words[:6]]
                        jql = " OR ".join(clauses)
                    else:
                        jql = f'text ~ "{query_str}"'

                log.info("Searching Jira issues with JQL: %s (view='full', maxResults=25)", jql)
                try:
                    res = await session.call_tool(
                        "searchJiraIssuesUsingJql",
                        {"cloudId": cid, "jql": jql, "maxResults": 25, "view": "full"},
                    )
                except Exception as e:
                    log.warning("JQL search failed (%s), retrying with simple text query", e)
                    safe_terms = " ".join([w for w in query_str.split() if len(w) > 2][:3])
                    res = await session.call_tool(
                        "searchJiraIssuesUsingJql",
                        {"cloudId": cid, "jql": f'text ~ "{safe_terms}"', "maxResults": 25, "view": "full"},
                    )
            else:
                log.info("Listing Jira projects")
                res = await session.call_tool(
                    "listJiraProjects",
                    {"cloudId": cid},
                )

            extracted = _extract_text(res)
            try:
                return json.loads(extracted)
            except Exception:
                return extracted

    try:
        return await _run(headers)
    except Exception as exc:
        # If Bearer token fails (e.g. 401 expired), fall back to Basic Auth automatically
        if "Bearer" in headers.get("Authorization", ""):
            log.warning("Bearer token failed (%s). Retrying with Basic Auth...", exc)
            return await _run({"Authorization": _auth_header()})
        raise


async def list_cloud_resources() -> List[Dict[str, Any]]:
    """List accessible cloud resources via getAccessibleAtlassianResources."""
    try:
        async with _session() as session:
            res = await session.call_tool("getAccessibleAtlassianResources", {})
            text = _extract_text(res)
            data = json.loads(text)
            if isinstance(data, dict) and "data" in data and "resources" in data["data"]:
                return data["data"]["resources"]
            elif isinstance(data, list):
                return data
            return [{"id": DEFAULT_CLOUD_ID, "name": "zenith-rca", "url": "https://zenith-rca.atlassian.net"}]
    except Exception as e:
        log.warning("Could not list cloud resources: %s", e)
        return [{"id": DEFAULT_CLOUD_ID, "name": "zenith-rca", "url": "https://zenith-rca.atlassian.net"}]


if __name__ == "__main__":
    print("Testing new_mcp with Basic Auth...")
    test_res = asyncio.run(search_jira_context(query="SCRUM-5"))
    print("\nResult for SCRUM-5:\n", json.dumps(test_res, indent=2) if isinstance(test_res, dict) else test_res)
