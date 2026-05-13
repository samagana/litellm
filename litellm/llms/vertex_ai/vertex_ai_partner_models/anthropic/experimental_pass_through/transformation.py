from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_MIN_THINKING_BUDGET_TOKENS
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase
from ..output_params_utils import sanitize_vertex_anthropic_output_params

CLEAR_THINKING_EDIT_TYPE = "clear_thinking_20251015"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


def _has_clear_thinking_edit(context_management: Optional[dict]) -> bool:
    if not isinstance(context_management, dict):
        return False
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return False
    return any(
        isinstance(e, dict) and e.get("type") == CLEAR_THINKING_EDIT_TYPE
        for e in edits
    )


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        vertex_ai_project = VertexBase.safe_get_vertex_ai_project(litellm_params)
        vertex_ai_location = VertexBase.safe_get_vertex_ai_location(litellm_params)

        project_id: Optional[str] = None
        if "Authorization" not in headers:
            vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(
                litellm_params
            )

            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_ai_project,
                custom_llm_provider="vertex_ai",
            )

            headers["Authorization"] = f"Bearer {access_token}"
        else:
            # Authorization already in headers, but we still need project_id
            project_id = vertex_ai_project

        # Always calculate api_base if not provided, regardless of Authorization header
        if api_base is None:
            api_base = self.get_complete_vertex_url(
                custom_api_base=api_base,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                project_id=project_id or "",
                partner=VertexPartnerProvider.claude,
                stream=optional_params.get("stream", False),
                model=model,
            )

        headers["content-type"] = "application/json"

        # Add beta headers for Vertex AI
        tools = optional_params.get("tools", [])
        beta_values: set[str] = set()

        # Get existing beta headers if any
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))

        # Check for context management
        context_management_param = optional_params.get("context_management")
        if context_management_param is not None:
            # Check edits array for compact_20260112 type
            edits = context_management_param.get("edits", [])
            has_compact = False
            has_other = False

            for edit in edits:
                edit_type = edit.get("type", "")
                if edit_type == "compact_20260112":
                    has_compact = True
                else:
                    has_other = True

            # Add compact header if any compact edits exist
            if has_compact:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value)

            # Add context management header if any other edits exist
            if has_other:
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value
                )

        # If clear_thinking_20251015 is present without enabled/adaptive thinking,
        # the request body transformation will inject minimal enabled thinking.
        # Vertex requires the interleaved-thinking beta header for that to be valid.
        if _has_clear_thinking_edit(context_management_param):
            thinking = optional_params.get("thinking")
            thinking_type = (
                thinking.get("type") if isinstance(thinking, dict) else None
            )
            if thinking_type not in ("enabled", "adaptive"):
                beta_values.add(INTERLEAVED_THINKING_BETA)

        # Check for web search tool
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type", "").startswith(
                ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value
            ):
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value
                )
                break

        # Check for tool search tools - Vertex AI uses different beta header
        anthropic_model_info = AnthropicModelInfo()
        if anthropic_model_info.is_tool_search_used(tools):
            beta_values.add(get_tool_search_beta_header("vertex_ai"))

        if beta_values:
            headers["anthropic-beta"] = ",".join(beta_values)

        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "api_base is required. Unable to determine the correct api_base for the request."
            )
        return api_base  # no transformation is needed - handled in validate_environment

    def _ensure_thinking_for_clear_thinking_context_management(
        self,
        anthropic_messages_request: Dict,
    ) -> bool:
        """
        Vertex AI rejects ``clear_thinking_20251015`` context-management edits
        unless extended thinking is ``enabled`` or ``adaptive``. Claude Code sends
        the edit with ``thinking: {type: adaptive}``, but the ``adaptive`` type is
        not preserved on the outbound Vertex request, producing 400:

            ``clear_thinking_20251015`` strategy requires ``thinking`` to be
            enabled or adaptive

        When we detect that edit type without an active enabled/adaptive thinking
        config, inject a minimal ``thinking`` config so the request succeeds.
        Mirrors the Bedrock fix in
        ``AmazonAnthropicClaudeMessagesConfig._ensure_thinking_for_clear_thinking_context_management``.

        Returns:
            True if ``thinking`` was added or upgraded.
        """
        if not _has_clear_thinking_edit(
            anthropic_messages_request.get("context_management")
        ):
            return False

        thinking = anthropic_messages_request.get("thinking")
        if isinstance(thinking, dict):
            t = thinking.get("type")
            if t in ("enabled", "adaptive"):
                return False
            verbose_logger.debug(
                "Vertex %s: replacing thinking=%s with minimal enabled thinking",
                CLEAR_THINKING_EDIT_TYPE,
                thinking,
            )

        max_tokens = anthropic_messages_request.get("max_tokens")
        budget = ANTHROPIC_MIN_THINKING_BUDGET_TOKENS
        if isinstance(max_tokens, int) and max_tokens <= budget:
            verbose_logger.warning(
                "Vertex %s: max_tokens=%s is not greater than minimum thinking "
                "budget (%s); cannot inject thinking safely",
                CLEAR_THINKING_EDIT_TYPE,
                max_tokens,
                budget,
            )
            return False

        anthropic_messages_request["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
        verbose_logger.debug(
            "Vertex %s: injected thinking with budget_tokens=%s",
            CLEAR_THINKING_EDIT_TYPE,
            budget,
        )
        return True

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._remove_scope_from_cache_control(anthropic_messages_request)

        self._ensure_thinking_for_clear_thinking_context_management(
            anthropic_messages_request=anthropic_messages_request,
        )

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop(
            "model", None
        )  # do not pass model in request body to vertex ai

        sanitize_vertex_anthropic_output_params(anthropic_messages_request)

        return anthropic_messages_request
