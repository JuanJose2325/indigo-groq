"""Constantes y claves de opciones de la integración.

Frontera: acá no hay lógica, solo nombres y valores. Es el único módulo que
todos los demás pueden importar sin pensar, porque no depende de ninguno.
El vocabulario de esfuerzo de razonamiento NO vive acá a propósito: se deriva
de `_EQUIVALENCIAS` en razonamiento.py, para que la tabla que traduce y la
lista que muestra la UI no puedan divergir.
"""

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

# Cuánto piensa el modelo principal CUANDO piensa. El enrutador de
# razonamiento no elige este valor: elige si se usa o si se manda "none".
# Vacío significa "no configurado", que no es lo mismo que "none": vacío manda
# reasoning_format="hidden" sin esfuerzo y deja que el modelo use su criterio.
CONF_REASONING_EFFORT = "reasoning_effort"

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

CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_MAX_RETRIES = "max_retries"
RECOMMENDED_MAX_RETRIES = 0

# Estimación conservadora para español; solo sirve para decidir el recorte.
CHARS_PER_TOKEN = 3.5

# Máximo de vueltas de ida y vuelta con el modelo dentro de un mismo turno.
# Es un cortacircuitos: un modelo que se obceca pidiendo la misma herramienta
# gastaría el cupo entero del minuto sin llegar nunca a contestar.
MAX_TOOL_ITERATIONS = 10

# Piso al que puede bajar max_tokens cuando Groq rechaza por tamaño. Por
# debajo de esto la respuesta sale cortada a media frase, que por voz se
# entiende peor que un "no pude": ahí conviene dejar de encoger y fallar.
TOPE_MINIMO_TOKENS = 400

# --- Enrutador de casa -------------------------------------------------------
# Decide si el turno necesita el contexto y las herramientas de la casa. Cuando
# dice que no, la petición se manda sin el bloque de herramientas: son 4150
# tokens de definiciones contra un límite de 8000 por minuto, o sea que más de
# la mitad del cupo se iba en describir la casa para preguntas que no la tocan.
CONF_CASA_ROUTER_ENABLED = "casa_router_enabled"
CONF_CASA_ROUTER_MODEL = "casa_router_model"
CONF_CASA_ROUTER_EFFORT = "casa_router_effort"
CONF_CASA_ROUTER_THRESHOLD = "casa_router_threshold"

# --- Enrutador de razonamiento -----------------------------------------------
# Decide si el turno necesita pensar. No elige cuánto: eso es
# CONF_REASONING_EFFORT. Existe porque razonar de más devolvía respuestas
# vacías (el pensamiento se come max_tokens antes de llegar a contestar).
CONF_RAZON_ROUTER_ENABLED = "razon_router_enabled"
CONF_RAZON_ROUTER_MODEL = "razon_router_model"
CONF_RAZON_ROUTER_EFFORT = "razon_router_effort"
CONF_RAZON_ROUTER_THRESHOLD = "razon_router_threshold"

# --- Comunes a los dos enrutadores (sin campo propio en la UI) ---------------
# Los dos arrancan APAGADOS: actualizar la integración no puede cambiarle el
# comportamiento a una instalación que está andando. Con los dos apagados el
# turno tiene que ser indistinguible del de la versión anterior.
RECOMMENDED_ROUTER_ENABLED = False
RECOMMENDED_ROUTER_MODEL = "openai/gpt-oss-20b"
RECOMMENDED_ROUTER_THRESHOLD = 0.7
# Vacío = no configurado; lo resuelve `_esfuerzo_inicial` con la familia del
# modelo que el usuario haya elegido para esa fila.
RECOMMENDED_ROUTER_EFFORT = ""
# Techo del enrutador: solo tiene que devolver un JSON de tres claves. Darle
# más sería regalarle cupo del minuto a una decisión que no se escucha.
ROUTER_MAX_TOKENS = 150
# Por voz no hay spinner, así que un enrutador colgado es peor que un
# enrutador equivocado: el silencio largo es indistinguible de un cuelgue y el
# usuario repite el comando, encadenando dos peticiones contra el mismo cupo.
TIEMPO_MAXIMO_ENRUTADOR = 5.0
# Turnos previos que ve el enrutador de casa. Con menos no se resuelve el caso
# real: "¿tengo luces en el cuarto?" -> "sí, una" -> "apagala". El "apagala"
# solo se entiende viendo la respuesta anterior de la IA, no solo la del
# usuario.
TURNOS_CONTEXTO_CASA = 3

# Lo que se guarda al crear la entrada. No incluye ninguna clave de enrutador
# a propósito: así una entrada nueva y una que viene de la versión anterior
# arrancan con los dos enrutadores apagados por igual, y la única fuente de
# verdad de ese default es RECOMMENDED_ROUTER_ENABLED.
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
