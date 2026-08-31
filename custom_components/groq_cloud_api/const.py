"""Constants for the Groq Cloud API integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "groq_cloud_api"

LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Groq Cloud API"
DEFAULT_CONVERSATION_NAME = "Groq Conversation"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CONF_SUPPORTS_REASONING = "supports_reasoning"
CONF_REASONING_EFFORT = "reasoning_effort"

# Known models with predefined reasoning effort options
QWEN_REASONING_OPTIONS = ["default", "none"]
GPT_OSS_REASONING_OPTIONS = ["low", "medium", "high"]
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 300

# Presupuesto de tokens para el historial que se reenvía en cada petición.
# El límite de Groq es de tokens por minuto contando ENTRADA + SALIDA, así que
# esto tiene que dejar sitio para max_tokens (y, con razonamiento activo, para
# los tokens de pensamiento, que salen de la misma bolsa).
CONF_HISTORY_BUDGET = "history_budget"
RECOMMENDED_HISTORY_BUDGET = 5000

# Cadena de modelos de respaldo, en orden de preferencia. Los límites de Groq
# son POR MODELO, así que cada uno de la cadena aporta su propia ventana de
# tokens por minuto: cuatro modelos ≈ cuatro veces el cupo.
CONF_MODEL_CHAIN = "model_chain"
RECOMMENDED_MODEL_CHAIN: list[str] = []
# Segundos que se evita reutilizar un modelo tras haberlo usado. Rota ANTES de
# que Groq rechace, para no pagar el viaje de red de un 429.
CONF_MODEL_COOLDOWN = "model_cooldown"
RECOMMENDED_MODEL_COOLDOWN = 60

# Respaldo en Cloudflare Workers AI. La gracia es que ahí corre EL MISMO
# qwen3.8-27b y no hay límite de tokens por minuto, así que cuando Groq se
# queda sin cupo se sigue con la misma calidad, solo más lento (mediana ~3,4 s
# contra ~1 s, con picos ocasionales de casi 20 s).
CONF_CF_ACCOUNT = "cf_account_id"
CONF_CF_TOKEN = "cf_api_token"
CONF_CF_MODEL = "cf_model"
RECOMMENDED_CF_MODEL = "@cf/qwen/qwen3.8-27b"
# Marca interna para distinguir a qué proveedor va cada candidato de la cadena.
PREFIJO_CF = "cf:"
# Estimación conservadora para español; solo sirve para decidir el recorte.
CHARS_PER_TOKEN = 3.5
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_MAX_RETRIES = "max_retries"
RECOMMENDED_MAX_RETRIES = 0

DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_HISTORY_BUDGET: RECOMMENDED_HISTORY_BUDGET,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_MAX_RETRIES: RECOMMENDED_MAX_RETRIES,
}