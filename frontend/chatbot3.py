import asyncio
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.jira_agent import decide_and_fetch_jira_context
from backend.new_mcp import format_mcp_error, search_jira_context
from backend.chat_service import generate_video_storyboard
from backend.video_service import generate_storyboard_video

st.set_page_config(
    page_title="Video Studio (Agentic MCP)",
    page_icon="🎥",
    layout="wide",
)

HAS_V2 = hasattr(st, "components") and hasattr(st.components, "v2")

COLOR_PRESETS = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Red": "#FF0000",
    "Blue": "#0000FF",
    "Green": "#008000",
    "Yellow": "#FFFF00",
    "Orange": "#FFA500",
    "Purple": "#800080",
    "Pink": "#FFC0CB",
    "Gray": "#808080",
    "Navy": "#000080",
    "Teal": "#008080",
}

SUBTITLE_STYLES = [
    "classic",
    "classic-progressive",
    "classic-one-word",
    "boxed-line",
    "boxed-word",
]

SUBTITLE_POSITIONS = [
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center-center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
    "mid-bottom-center",
    "mid-top-center",
    "custom",
]

EDITABLE_SUBTITLE_SETTINGS = {
    "style",
    "position",
    "max-words-per-line",
    "font-size",
    "font-weight",
    "line-color",
    "word-color",
    "outline-color",
    "outline-width",
}

EDITABLE_TEXT_SETTINGS = {
    "font-family",
    "font-size",
    "font-weight",
    "color",
    "background-color",
    "text-align",
    "vertical-align",
    "align",
}

EDITABLE_ELEMENT_LAYOUT = {"x", "y", "width", "height"}


def _movie_editor_key(job_id, path, suffix="value"):
    return f"movie-editor-{job_id}-{'-'.join(map(str, path))}-{suffix}"


def _edit_movie_value(
    value,
    label,
    job_id,
    path,
    disabled=False,
    editable=False,
    subtitle_settings=False,
    text_settings=False,
):
    if isinstance(value, dict):
        edited = {}
        element_type = value.get("type")
        fields = dict(value)
        if subtitle_settings and fields.get("position") == "custom":
            fields.setdefault("x", 0)
            fields.setdefault("y", 0)
        with st.expander(label, expanded=len(path) <= 2):
            for field_name, field_value in fields.items():
                field_label = (
                    str(field_name)
                    .replace("_", " ")
                    .replace("-", " ")
                    .title()
                )
                edited[field_name] = _edit_movie_value(
                    field_value,
                    field_label,
                    job_id,
                    (*path, field_name),
                    disabled,
                    editable=(
                        (element_type in {"voice", "text"} and field_name == "text")
                        or field_name == "background-color"
                        or (
                            element_type in {"text", "html", "image", "video"}
                            and field_name in EDITABLE_ELEMENT_LAYOUT
                        )
                        or (
                            text_settings
                            and field_name in EDITABLE_TEXT_SETTINGS
                        )
                        or (
                            subtitle_settings
                            and field_name in EDITABLE_SUBTITLE_SETTINGS
                        )
                        or (
                            subtitle_settings
                            and field_name in {"x", "y"}
                            and value.get("position") == "custom"
                        )
                    ),
                    subtitle_settings=(
                        subtitle_settings
                        or (
                            element_type == "subtitles"
                            and field_name == "settings"
                        )
                    ),
                    text_settings=(
                        text_settings
                        or (
                            element_type == "text"
                            and field_name == "settings"
                        )
                    ),
                )
        return edited

    if isinstance(value, list):
        edited = []
        with st.expander(label, expanded=len(path) <= 2):
            item_label = {
                "scenes": "Scene",
                "elements": "Element",
            }.get(str(path[-1]) if path else "", "Item")
            for index, item in enumerate(value):
                detail = (
                    f" ({item['type'].title()})"
                    if isinstance(item, dict) and isinstance(item.get("type"), str)
                    else ""
                )
                edited.append(
                    _edit_movie_value(
                        item,
                        f"{item_label} {index + 1}{detail}",
                        job_id,
                        (*path, index),
                        disabled,
                        editable=editable,
                        subtitle_settings=subtitle_settings,
                        text_settings=text_settings,
                    )
                )
        return edited

    widget_key = _movie_editor_key(job_id, path)
    field_name = str(path[-1]).lower() if path else ""

    if not editable:
        st.caption(f"{label} (fixed): {value}")
        return value

    if isinstance(value, bool):
        st.session_state.setdefault(widget_key, value)
        return st.checkbox(label, key=widget_key, disabled=disabled)

    if isinstance(value, (int, float)):
        st.session_state.setdefault(widget_key, value)
        return st.number_input(
            label,
            key=widget_key,
            step=1 if isinstance(value, int) else 0.1,
            disabled=disabled,
        )

    if subtitle_settings and field_name in {"style", "position"}:
        options = (
            SUBTITLE_STYLES if field_name == "style" else SUBTITLE_POSITIONS
        )
        option_key = _movie_editor_key(job_id, path, "subtitle-option")
        current_value = st.session_state.get(option_key, value)
        if current_value not in options:
            options = [current_value, *options]
        st.session_state.setdefault(option_key, value)
        return st.selectbox(
            label,
            options,
            key=option_key,
            disabled=disabled,
        )

    if isinstance(value, str) and ("color" in field_name or "colour" in field_name):
        normalized_value = value.strip().lower()
        preset = next(
            (
                name
                for name, color_value in COLOR_PRESETS.items()
                if name.lower() == normalized_value
                or color_value.lower() == normalized_value
            ),
            "Custom value",
        )
        options = [*COLOR_PRESETS, "Custom value"]
        choice_key = _movie_editor_key(job_id, path, "color-choice")
        st.session_state.setdefault(choice_key, preset)
        selected = st.selectbox(
            label,
            options,
            key=choice_key,
            disabled=disabled,
        )
        if selected == "Custom value":
            custom_key = _movie_editor_key(job_id, path, "custom-color")
            picker_key = _movie_editor_key(job_id, path, "color-picker")
            st.session_state.setdefault(custom_key, value)
            initial_color = st.session_state[custom_key]
            if not (
                isinstance(initial_color, str)
                and len(initial_color) == 7
                and initial_color.startswith("#")
                and all(
                    character in "0123456789abcdefABCDEF"
                    for character in initial_color[1:]
                )
            ):
                initial_color = next(
                    (
                        color_value
                        for color_name, color_value in COLOR_PRESETS.items()
                        if color_name.lower() == str(initial_color).strip().lower()
                        or color_value.lower() == str(initial_color).strip().lower()
                    ),
                    "#000000",
                )
            st.session_state.setdefault(picker_key, initial_color)

            def sync_custom_color():
                st.session_state[custom_key] = st.session_state[picker_key]

            def sync_color_picker():
                custom_color = str(st.session_state[custom_key]).strip().lower()
                if (
                    len(custom_color) == 7
                    and custom_color.startswith("#")
                    and all(
                        character in "0123456789abcdef"
                        for character in custom_color[1:]
                    )
                ):
                    st.session_state[picker_key] = custom_color
                    return

                for color_name, color_value in COLOR_PRESETS.items():
                    if custom_color in {color_name.lower(), color_value.lower()}:
                        st.session_state[picker_key] = color_value
                        return

            st.color_picker(
                f"Pick {label}",
                key=picker_key,
                on_change=sync_custom_color,
                disabled=disabled,
            )
            return st.text_input(
                f"Custom value for {label}",
                key=custom_key,
                on_change=sync_color_picker,
                disabled=disabled,
            )
        return COLOR_PRESETS[selected]

    if isinstance(value, str) or value is None:
        initial_value = value if isinstance(value, str) else ""
        st.session_state.setdefault(widget_key, initial_value)
        if (
            field_name in {"html", "text", "comment"}
            or "\n" in initial_value
            or len(initial_value) > 100
        ):
            return st.text_area(
                label,
                key=widget_key,
                height=220 if field_name == "html" else 120,
                disabled=disabled,
            )
        edited_value = st.text_input(label, key=widget_key, disabled=disabled)
        return None if value is None and not edited_value else edited_value

    st.caption(f"{label}: {value}")
    return value


if HAS_V2:
    VIDEO_PROMPT = st.components.v2.component(
        "ai_video_studio.video_prompt",
        html="""
        <div class="video-composer">
          <label for="video-description">Describe your video</label>
          <div class="composer-row">
            <textarea
              id="video-description"
              placeholder="Describe the video you want. You can mention Jira tickets (e.g. SCRUM-5) or general ideas..."
              rows="3"
            ></textarea>
          </div>
          <button class="generate-button" type="button" disabled>
            Generate video
          </button>
        </div>
        """,
        css="""
        .video-composer {
          display: grid;
          gap: 0.5rem;
          font-family: var(--st-font, sans-serif);
          color: var(--st-text-color, #262730);
        }
        .composer-row {
          border: 1px solid var(--st-widget-border-color, #c8c8c8);
          border-radius: 0.75rem;
          background: var(--st-background-color, #fff);
        }
        .composer-row:focus-within {
          border-color: var(--st-primary-color, #ff4b4b);
          box-shadow: 0 0 0 1px var(--st-primary-color, #ff4b4b);
        }
        .composer-row textarea {
          display: block;
          width: 100%;
          box-sizing: border-box;
          resize: vertical;
          padding: 0.75rem;
          border: 0;
          outline: 0;
          border-radius: inherit;
          background: transparent;
          color: var(--st-text-color, #262730);
          font: inherit;
        }
        .generate-button {
          justify-self: center;
          flex: 0 0 auto;
          padding: 0.65rem 1rem;
          border: 0;
          border-radius: 1rem;
          background: var(--st-primary-color, #ff4b4b);
          color: white;
          font: inherit;
          cursor: pointer;
        }
        .generate-button:disabled {
          opacity: 0.45;
          cursor: default;
        }
        """,
        js="""
        export default function (component) {
          const { data, parentElement, setStateValue, setTriggerValue } = component
          const prompt = parentElement.querySelector("#video-description")
          const button = parentElement.querySelector(".generate-button")
          if (!prompt || !button) return

          let submissionPending = Boolean(data?.busy)
          let submittedRequestId = null

          if (prompt.value !== (data?.prompt ?? "")) {
            prompt.value = data?.prompt ?? ""
          }

          const updateButton = () => {
            if (
              submissionPending &&
              submittedRequestId &&
              !data?.busy &&
              data?.lastRequestId === submittedRequestId
            ) {
              submissionPending = false
            }
            const isBusy = submissionPending || Boolean(data?.busy)
            prompt.disabled = isBusy
            button.disabled = isBusy || !prompt.value.trim()
          }

          prompt.oninput = () => {
            updateButton()
            setStateValue("prompt", prompt.value)
          }
          prompt.onkeydown = (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault()
              button.click()
            }
          }
          button.onclick = () => {
            const text = prompt.value.trim()
            if (!text || button.disabled) return

            const requestId = globalThis.crypto?.randomUUID?.()
              ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`
            submittedRequestId = requestId
            submissionPending = true
            button.disabled = true
            prompt.disabled = true
            prompt.value = ""
            setStateValue("prompt", "")
            setStateValue("busy", true)
            setTriggerValue("generate", { prompt: text, request_id: requestId })
          }
          updateButton()
        }
        """,
    )
else:
    VIDEO_PROMPT = None

with st.sidebar:
    st.header("Atlassian MCP")
    jira_email = os.getenv("ATLASSIAN_USERNAME") or os.getenv("JIRA_EMAIL") or "vasu.j@prodapt.com"
    jira_site = os.getenv("ATLASSIAN_SITE") or "zenith-rca.atlassian.net"

    st.success("✅ Atlassian Connected (API Token)")
    st.caption(f"**Site:** `{jira_site}`\n\n**User:** `{jira_email}`")

    with st.expander("MCP Settings", expanded=False):
        st.write("**Auth Mode:** Basic (API Token - Permanent)")
        st.write(f"**Cloud ID:** `{os.getenv('ATLASSIAN_CLOUD_ID', 'ef6bdfdf-7960-46f9-9a35-47450de6247c')}`")
        st.write(f"**MCP Endpoint:** `{os.getenv('ATLASSIAN_MCP_URL', 'https://mcp.atlassian.com/v2/mcp?tools=all')}`")

    test_query = st.text_input("Jira MCP Query / Issue Key", value="ST-1")
    if st.button("Test Jira MCP (new_mcp)", use_container_width=True):
        with st.spinner(f"Executing Jira MCP test for '{test_query}'..."):
            try:
                result = asyncio.run(
                    search_jira_context(
                        query=test_query,
                    )
                )
                st.success(f"✅ Jira MCP Succeeded for '{test_query}'!")
                if isinstance(result, (dict, list)):
                    st.json(result)
                else:
                    st.markdown(result)
            except Exception as e:
                st.error(f"⚠️ Jira MCP Error: {format_mcp_error(e)}")

    st.divider()
    st.header("Jira agent settings")

    model_name = st.selectbox(
        "Gemini model",
        [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.1-pro-preview",
        ],
    )
    if st.button("Clear chat", width="stretch"):
        component_state = st.session_state.get("video_prompt_component")
        if component_state is not None:
            component_state["prompt"] = ""
        st.session_state.video_jobs = []
        st.rerun()

st.title("🎥 Video Studio (Dynamic Atlassian MCP)")
st.caption(
    "🤖 **Agentic Workflow:** You simply type what video you want. Gemini autonomously evaluates if your request "
    "needs Atlassian data, discovers the tools currently available from MCP, and selects one or more tools to retrieve "
    "relevant evidence. It then builds a detailed storyboard and asks Veo to generate each full-screen scene with native audio. "
    "Subtitles are added only when requested."
)

if "video_jobs" not in st.session_state:
    st.session_state.video_jobs = []
if "video_generation_in_progress" not in st.session_state:
    st.session_state.video_generation_in_progress = False

submission = None
is_new_submission = False

if HAS_V2 and VIDEO_PROMPT is not None:
    with st.bottom:
        video_submission = VIDEO_PROMPT(
            key="video_prompt_component",
            data={
                "prompt": st.session_state.get(
                    "video_prompt_component", {}
                ).get("prompt", ""),
                "busy": st.session_state.video_generation_in_progress,
                "lastRequestId": st.session_state.get("last_video_request_id"),
            },
            on_prompt_change=lambda: None,
            on_generate_change=lambda: None,
        )

    if video_submission is not None:
        if isinstance(video_submission, dict):
            submission = video_submission.get("generate")
        else:
            try:
                submission = getattr(video_submission, "generate", None)
            except Exception:
                try:
                    submission = video_submission.get("generate")
                except Exception:
                    submission = None

    request_id = submission.get("request_id") if isinstance(submission, dict) else None
    is_new_submission = (
        isinstance(submission, dict)
        and isinstance(request_id, str)
        and request_id != st.session_state.get("last_video_request_id")
    )
else:
    chat_prompt = st.chat_input(
        "Describe the video you want (mention a Jira issue like ST-1 or ST Sprint 1, or a general topic)...",
        disabled=st.session_state.video_generation_in_progress,
    )
    if chat_prompt and chat_prompt.strip():
        req_id = uuid.uuid4().hex
        submission = {"prompt": chat_prompt.strip(), "request_id": req_id}
        is_new_submission = True

if is_new_submission:
    st.session_state.last_video_request_id = request_id
    st.session_state.video_generation_in_progress = True

    prompt = submission.get("prompt", "").strip()
    if prompt:
        job = {
            "id": uuid.uuid4().hex,
            "prompt": prompt,
            "agent_decision": None,
            "jira_context": None,
            "raw_jira_context": None,
            "storyboard": None,
            "video_bytes": None,
            "error": None,
        }
        st.session_state.video_jobs.append(job)

        with st.spinner(
            "Evaluating request and preparing a complete video storyboard..."
        ):
            try:
                # --- AGENTIC STEP ---
                # Gemini selects from the tools currently exposed by Atlassian MCP.
                is_jira_required, jira_ctx, query_used, mcp_error = asyncio.run(
                    decide_and_fetch_jira_context(
                        prompt=prompt,
                        model_name=model_name,
                    )
                )

                if is_jira_required:
                    if jira_ctx:
                        job["jira_context"] = jira_ctx

                    if mcp_error:
                        job["agent_decision"] = (
                            "⚠️ Jira evidence retrieval failed or was incomplete. "
                            "Video generation was stopped."
                        )
                        job["error"] = (
                            "Could not retrieve all required Jira evidence. "
                            f"Details: {mcp_error}"
                        )
                    elif not jira_ctx:
                        job["agent_decision"] = (
                            "⚠️ Jira evidence was required, but no matching "
                            "evidence was returned. Video generation was stopped."
                        )
                        job["error"] = (
                            "No Jira evidence was returned for this request, "
                            "so video generation was stopped."
                        )
                    else:
                        job["agent_decision"] = (
                            f"✅ Atlassian MCP tools selected: {query_used}. "
                            "Retrieved context for the video."
                        )
                elif mcp_error:
                    job["agent_decision"] = (
                        "⚠️ Could not determine whether Jira evidence was needed: "
                        f"{mcp_error}. Video generation was stopped."
                    )
                    job["error"] = (
                        "Could not determine whether Jira evidence is required, "
                        f"so video generation was stopped. Details: {mcp_error}"
                    )
                else:
                    job["agent_decision"] = (
                        "ℹ️ Jira evidence was not requested or needed. "
                        "Generating from your prompt only."
                    )

                if not job["error"]:
                    job["storyboard"] = generate_video_storyboard(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=0.3,
                        jira_context=job.get("jira_context") or jira_ctx,
                    )
            except Exception as exc:
                job["error"] = f"Could not prepare the video storyboard: {exc}"

    st.session_state.video_generation_in_progress = False

for index, job in enumerate(st.session_state.video_jobs):
    is_latest_job = index == len(st.session_state.video_jobs) - 1
    if is_new_submission and is_latest_job:
        st.html(
            """
            <div id="latest-video-output"></div>
            <script>
              requestAnimationFrame(() => {
                document.getElementById("latest-video-output")
                  ?.scrollIntoView({ behavior: "smooth", block: "start" })
              })
            </script>
            """,
            unsafe_allow_javascript=True,
        )

    with st.chat_message("user"):
        st.markdown(job["prompt"])

    with st.chat_message("assistant"):
        if job.get("agent_decision"):
            st.info(f"🤖 **Agent Decision:** {job['agent_decision']}")

        if job.get("jira_context"):
            with st.expander("📋 Review retrieved Jira evidence", expanded=False):
                st.markdown(job["jira_context"])

        storyboard = job.get("storyboard")
        if storyboard:
            scenes = storyboard["scenes"]
            with st.expander(
                f"Review video storyboard ({len(scenes)} scenes)",
                expanded=False,
            ):
                st.caption(
                    "Each scene uses Veo-generated audio. Subtitles are "
                    f"{'included' if storyboard.get('include_subtitles') else 'not included'}. "
                    "Veo generates full-screen scenes and integrates animated "
                    "workflow or chart shapes into the visual direction when "
                    "they help explain the scene."
                )
                for scene_number, scene in enumerate(scenes, start=1):
                    st.markdown(f"**Scene {scene_number}: {scene['title']}**")
                    st.markdown(f"**Planned voiceover:** {scene['narration']}")
                    st.caption(f"Visual: {scene['visual_prompt']}")

        if job["error"] and not storyboard:
            st.error(job["error"])
            continue

        if job["error"]:
            st.error(job["error"])

        if job.get("video_bytes"):
            st.video(
                job["video_bytes"],
                format="video/mp4",
                alt="Generated video preview",
            )
            st.download_button(
                "Download MP4",
                data=job["video_bytes"],
                file_name=f"video_{job['id']}.mp4",
                mime="video/mp4",
                key=f"download-mp4-{job['id']}",
                width="stretch",
                icon=":material/download:",
            )

    if job.get("storyboard") and not job.get("video_bytes"):
        scene_count = len(job["storyboard"]["scenes"])
        if st.button(
            f"Generate {scene_count}-scene video with Veo audio",
            type="primary",
            key=f"generate-storyboard-{job['id']}",
            width="stretch",
        ):
            status = st.status(
                "Generating the storyboard scenes...",
                expanded=True,
            )
            try:
                job["video_bytes"] = generate_storyboard_video(
                    job["storyboard"],
                    status_callback=lambda message: status.update(
                        label=message
                    ),
                )
                job["error"] = None
                status.update(
                    label="Video generated and ready to download.",
                    state="complete",
                )
                st.rerun()
            except Exception as exc:
                job["error"] = f"Video generation failed: {exc}"
                status.update(
                    label="Video rendering failed.",
                    state="error",
                )
                st.error(job["error"])
