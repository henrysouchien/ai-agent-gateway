"""Gateway injection point for the transport-owned standard runtime."""
from hank_model_transport.host_runtime import STANDARD_SEND_PROMPT_RUNTIME
from hank_model_transport.send_prompt import SendPromptRuntime


GATEWAY_SEND_PROMPT_RUNTIME: SendPromptRuntime = STANDARD_SEND_PROMPT_RUNTIME
