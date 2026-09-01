"""Conversation support for Groq Cloud."""

from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Literal

import groq
from groq._types import NOT_GIVEN
from groq.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.chat_completion_message_tool_call_param import Function
from groq.types.shared_params import FunctionDefinition
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConverseError,
    SystemContent,
    ToolResultContent,
    UserContent,
    async_get_result_from_chat_log,
    trace,
)
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import GroqConfigEntry
from .const import (
    CHARS_PER_TOKEN,
    CONF_CF_ACCOUNT,
    CONF_CF_MODEL,
    CONF_CF_TOKEN,
    CONF_CHAT_MODEL,
    CONF_HISTORY_BUDGET,
    CONF_MAX_RETRIES,
    CONF_MAX_TOKENS,
    CONF_MODEL_CHAIN,
    CONF_MODEL_COOLDOWN,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SUPPORTS_REASONING,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    PREFIJO_CF,
    RECOMMENDED_CF_MODEL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HISTORY_BUDGET,
    RECOMMENDED_MAX_RETRIES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MODEL_COOLDOWN,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# Segundos que se le conceden a Cloudflare antes de darlo por perdido.
TIEMPO_MAXIMO_CF = 15

# Piso al que puede bajar `max_tokens` cuando Groq rechaza por tamaño. Por
# debajo de esto la respuesta sale cortada a media frase, que por voz se
# entiende peor que un "no pude": ahí conviene dejar de encoger y fallar.
TOPE_MINIMO_TOKENS = 400


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GroqConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = GroqConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _assistant_content_to_message(
    content: AssistantContent,
) -> ChatCompletionAssistantMessageParam:
    """Convert AssistantContent to a Groq assistant message."""
    tool_calls: list[ChatCompletionMessageToolCallParam] = []

    if content.tool_calls:
        tool_calls = [
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=json_dumps(tool_call.tool_args),
                    name=tool_call.tool_name,
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ]

    assistant_message = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=content.content,
    )
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def _chat_log_to_messages(
    chat_log: conversation.ChatLog,
) -> list[
    ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
]:
    """Convert chat log content to Groq chat completion messages."""
    messages: list[
        ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
    ] = []

    for content in chat_log.content:
        if isinstance(content, SystemContent):
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system", content=content.content
                )
            )
        elif isinstance(content, UserContent):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.content)
            )
        elif isinstance(content, AssistantContent):
            messages.append(_assistant_content_to_message(content))
        elif isinstance(content, ToolResultContent):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=content.tool_call_id,
                    content=json_dumps(content.tool_result),
                )
            )

    return messages


# Cada familia acepta un vocabulario distinto de esfuerzo de razonamiento, y
# la cadena de respaldo cruza de una a otra. Sin traducir, saltar de un Qwen
# (que usa "default"/"none") a un gpt-oss (que exige "low"/"medium"/"high")
# devuelve HTTP 400 y el usuario escucha el error crudo en voz alta.
_EQUIVALENCIAS = {
    "qwen": {"low": "default", "medium": "default", "high": "default",
             "default": "default", "none": "none"},
    # OJO: "default" en Qwen equivale a esfuerzo máximo, pero traducirlo a
    # "high" en gpt-oss es contraproducente acá. En los modelos de razonamiento
    # los tokens de pensamiento SALEN DE max_tokens, así que con un presupuesto
    # chico (500) el razonamiento alto se lo come entero y la respuesta llega
    # VACÍA: el usuario no escucha nada. "medium" deja lugar para contestar.
    "gpt-oss": {"default": "medium", "none": "low",
                "low": "low", "medium": "medium", "high": "high"},
}


def _esfuerzo_para(model: str, pedido: str | None) -> str | None:
    """Traduce el esfuerzo configurado al vocabulario del modelo destino."""
    familia = ("qwen" if model.startswith("qwen/")
               else "gpt-oss" if model.startswith("openai/gpt-oss") else None)
    if familia is None:
        return pedido
    tabla = _EQUIVALENCIAS[familia]
    if pedido in tabla:
        return tabla[pedido]
    # Valor desconocido (modelo nuevo, config vieja): mejor el máximo de la
    # familia que un 400 que el usuario escucha como respuesta.
    LOGGER.warning(
        "Esfuerzo de razonamiento %r no válido para %s; uso el de por defecto",
        pedido, model,
    )
    return "default" if familia == "qwen" else "high"


async def _llamar_cloudflare(clientes: Any, kwargs: dict) -> Any:
    """Pide a Cloudflare por HTTP y devuelve el mismo tipo que el SDK de Groq.

    No se puede reusar el SDK de Groq con otro `base_url`: ese SDK le agrega
    `openai/v1/` a la ruta (porque la API de Groq vive en
    `api.groq.com/openai/v1/...`), así que apuntándolo a Cloudflare terminaba
    pidiendo `.../ai/v1/openai/v1/chat/completions` y Cloudflare respondía
    `No route for that URI`. Por eso acá se arma la petición a mano y se
    valida la respuesta contra el modelo de datos del SDK, que es compatible.
    """
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{clientes.cf_cuenta}/ai/v1/chat/completions")
    cuerpo = {k: v for k, v in kwargs.items()
              # `user` es de Groq; Cloudflare no lo espera y no aporta nada.
              if k != "user" and v is not NOT_GIVEN}
    # 120 s era demasiado: por voz no hay spinner, así que un silencio largo es
    # indistinguible de un cuelgue y el usuario repite el comando, encadenando
    # dos peticiones. Más vale decir "no pude" a los 15 s que callarse 40.
    r = await clientes.http.post(
        url, json=cuerpo, timeout=TIEMPO_MAXIMO_CF,
        headers={"Authorization": f"Bearer {clientes.cf_ficha}",
                 "Content-Type": "application/json"})
    if r.status_code >= 400:
        raise ConverseError(
            f"Cloudflare respondió {r.status_code}: {r.text[:300]}",
            conversation_id=None,
            response=intent.IntentResponse(language="es"),
        )
    return ChatCompletion.model_validate(r.json())


def _vacia_por_truncado(result: Any) -> bool:
    """¿El modelo gastó todo max_tokens razonando y no llegó a contestar?

    `finish_reason == "length"` con el contenido vacío es la firma exacta, y es
    el peor fallo posible porque no se parece a un fallo: no hay excepción, el
    pipeline da la vuelta entera como si todo hubiera salido bien, no emite
    `synthesize` y el usuario se queda escuchando silencio. Hay que tratarlo
    como un modelo que falló, no como una respuesta.
    """
    eleccion = (getattr(result, "choices", None) or [None])[0]
    if eleccion is None or getattr(eleccion, "finish_reason", None) != "length":
        return False
    mensaje = getattr(eleccion, "message", None)
    # Truncar DESPUÉS de pedir una herramienta no es este caso: la petición
    # está completa y sirve, aunque no venga texto acompañándola.
    if getattr(mensaje, "tool_calls", None):
        return False
    return not (getattr(mensaje, "content", None) or "").strip()


async def _pedir(clientes: Any, kwargs: dict, es_cf: bool) -> Any:
    """Una llamada al proveedor que toque, con la misma forma de respuesta."""
    if es_cf:
        return await _llamar_cloudflare(clientes, kwargs)
    return await clientes.groq.chat.completions.create(**kwargs)


def _detalle_error(err: Any) -> str:
    """El mensaje que manda Groq, no solo el número de estado.

    Sin esto en el log queda únicamente el código, y 413 y 429 se vuelven
    indistinguibles de un vistazo siendo problemas OPUESTOS: uno se arregla
    pidiendo menos, el otro esperando. El cuerpo del 413 además dice el límite
    y cuánto se pidió, que es el dato con el que se calibra `max_tokens`.
    """
    # Todo con isinstance en vez de `.get` encadenado: esto corre en el camino
    # de la respuesta al usuario, y un proveedor que devuelva `error` como
    # texto suelto en lugar de objeto no puede tumbar la petición entera.
    cuerpo = getattr(err, "body", None)
    if isinstance(cuerpo, dict):
        error = cuerpo.get("error")
        if isinstance(error, dict) and error.get("message") is not None:
            return str(error["message"])[:300]
    return str(err)[:300]


async def _pedir_encogiendo(clientes: Any, kwargs: dict, es_cf: bool,
                            etiqueta: str) -> Any:
    """Como `_pedir`, pero achica la petición si la rechazan por tamaño.

    HTTP 413 NO es falta de cupo, aunque el código lo tratara igual que un 429
    durante mucho tiempo: significa que la petición entera —la entrada más el
    techo de generación— no entra de una sola vez. Rotar de modelo ahí no
    arregla nada, porque al siguiente le llega exactamente lo mismo y lo
    rechaza igual; en el log eso se veía como pares 413→413→Cloudflare
    instantáneos. Lo único que ayuda es pedir menos.

    Se baja `max_tokens` a la mitad y se reintenta el MISMO modelo. El cambio
    se deja escrito en `kwargs` a propósito: si la petición no entraba acá,
    tampoco va a entrar en el siguiente candidato.
    """
    while True:
        try:
            return await _pedir(clientes, kwargs, es_cf)
        except groq.APIStatusError as err:
            tope = kwargs.get("max_tokens") or 0
            if err.status_code != 413 or tope <= TOPE_MINIMO_TOKENS:
                raise
            kwargs["max_tokens"] = max(TOPE_MINIMO_TOKENS, tope // 2)
            LOGGER.warning(
                "%s rechazó la petición por TAMAÑO (HTTP 413), no por cupo: "
                "%s. Bajo max_tokens de %s a %s y reintento el mismo modelo.",
                etiqueta, _detalle_error(err), tope, kwargs["max_tokens"],
            )


def _aplicar_razonamiento(kwargs: dict, model: str, options: Any,
                          forzar: str | None = None) -> None:
    """Parámetros de razonamiento, que dependen del modelo concreto.

    Extraído a una función porque con la cadena de respaldo el modelo cambia
    entre reintentos y hay que recalcularlos para cada uno.

    `forzar` pisa el esfuerzo configurado; se usa para repetir la pregunta sin
    pensamiento cuando el pensamiento se comió el presupuesto entero.
    """
    kwargs.pop("reasoning_format", None)
    kwargs.pop("reasoning_effort", None)
    kwargs.pop("include_reasoning", None)
    if not options.get(CONF_SUPPORTS_REASONING):
        return
    pedido = forzar if forzar is not None else options.get(CONF_REASONING_EFFORT)
    esfuerzo = _esfuerzo_para(model, pedido)
    # Antes esto era "qwen/qwen3-32b" (modelo deprecado por Groq en jun 2026),
    # así que qwen/qwen3.8-27b NO entraba acá y se quedaba sin
    # reasoning_format="hidden": el texto del pensamiento volvía DENTRO de la
    # respuesta y se comía max_tokens antes de llegar a contestar.
    if model.startswith("qwen/"):
        kwargs["reasoning_format"] = "hidden"
        kwargs["reasoning_effort"] = esfuerzo or "default"
    elif model.startswith("openai/gpt-oss"):
        # `include_reasoning=False` NO alcanza: con eso los gpt-oss devolvían
        # la cadena de pensamiento entera DENTRO de `content`, en inglés y con
        # la respuesta real pegada al final, y el TTS lo leía en voz alta.
        # El que de verdad la oculta es `reasoning_format`, igual que en Qwen.
        kwargs["reasoning_format"] = "hidden"
        kwargs["include_reasoning"] = False
        if esfuerzo:
            kwargs["reasoning_effort"] = esfuerzo
    # Familia desconocida: no se manda NADA de razonamiento. Antes acá se
    # colaba `reasoning_effort` a cualquier modelo, y los que no razonan
    # —llama-3.3-70b-versatile, por ejemplo— contestan HTTP 400 al recibirlo.
    # Eso importa justo ahora que la cadena se ensancha con modelos de otras
    # familias: un 400 en un eslabón de respaldo se escucha como un error
    # crudo en voz alta, que es peor que la respuesta algo peor de un modelo
    # sin pensamiento. Si más adelante entra un razonador nuevo, se agrega su
    # familia a `_EQUIVALENCIAS` y vuelve a recibir los parámetros.


def _candidatos(principal: str, cadena: list[str], ultimo_uso: dict[str, float],
                enfriamiento: float) -> list[str]:
    """Orden en que se van a probar los modelos, del mejor al peor.

    Rota ANTES de que Groq rechace, no después: si el modelo preferido se usó
    hace menos de `enfriamiento` segundos es muy probable que su ventana de
    tokens por minuto siga ocupada, y esperar el 429 para recién ahí saltar
    costaría un viaje de red entero. Como acá lo que manda es la latencia, se
    prefiere el segundo modelo antes que la espera.

    Los que están en enfriamiento no se descartan: se mandan al final, para que
    sigan sirviendo de última red si todos están calientes.
    """
    orden: list[str] = []
    for m in [principal, *cadena]:
        if m and m not in orden:
            orden.append(m)
    ahora = time.monotonic()
    frios = [m for m in orden if ahora - ultimo_uso.get(m, -1e9) >= enfriamiento]
    calientes = [m for m in orden if m not in frios]
    return frios + calientes


def _coste_aproximado(mensaje: Any) -> int:
    """Tokens estimados de un mensaje ya convertido al formato de Groq.

    Solo se usa para decidir el recorte, así que alcanza con una estimación
    conservadora: pasarse un poco de largo es inofensivo, quedarse corto no.
    """
    total = len(str(mensaje.get("content") or ""))
    for llamada in mensaje.get("tool_calls") or ():
        total += len(json_dumps(llamada))
    return int(total / CHARS_PER_TOKEN) + 4  # +4 por el envoltorio del rol


def _recortar_historial(messages: list, presupuesto: int) -> list:
    """Deja el prompt de sistema y los turnos más recientes que entren.

    El límite de Groq cuenta ENTRADA + SALIDA por minuto, y la petición crece
    en cada turno porque el historial entero se reenvía. Sin recorte, la
    conversación termina superando el límite y a partir de ahí falla SIEMPRE
    (no de a ratos): por eso antes se "arreglaba" sola al reiniciar, que es
    cuando Home Assistant descarta el conversation_id.

    Se descartan turnos viejos ENTEROS, nunca pedazos de un mensaje.
    """
    sistema = [m for m in messages if m.get("role") == "system"]
    resto = [m for m in messages if m.get("role") != "system"]

    disponible = presupuesto - sum(_coste_aproximado(m) for m in sistema)
    if disponible <= 0:
        # El prompt de sistema solo ya no entra: no hay nada que recortar que
        # ayude, y mandar la conversación vacía sería peor que dejarla pasar.
        LOGGER.warning(
            "El prompt de sistema (~%d tokens) supera el presupuesto de %d; "
            "no se recorta nada",
            presupuesto - disponible, presupuesto,
        )
        return messages

    ventana: list = []
    for mensaje in reversed(resto):
        coste = _coste_aproximado(mensaje)
        if coste > disponible:
            break
        disponible -= coste
        ventana.insert(0, mensaje)

    # Una ventana que empiece por un resultado de herramienta, o por un
    # assistant cuyas tool_calls quedaron fuera, es un JSON inválido para la
    # API. Se recorta hasta el primer mensaje de usuario, que siempre es un
    # corte limpio.
    while ventana and ventana[0].get("role") != "user":
        ventana.pop(0)

    if len(ventana) < len(resto):
        LOGGER.debug(
            "Historial recortado: %d de %d mensajes (presupuesto %d tokens)",
            len(ventana), len(resto), presupuesto,
        )

    return sistema + ventana


class GroqConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Groq conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: GroqConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        # Cuándo se usó por última vez cada modelo, para rotar la cadena de
        # respaldo sin esperar a que Groq devuelva un 429.
        self._ultimo_uso: dict[str, float] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Groq",
            model="Groq Cloud",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except ConverseError as err:
            return err.as_conversation_result()

        llm_api = chat_log.llm_api
        tools: list[ChatCompletionToolParam] | None = None
        if llm_api:
            tools = [
                _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]
            # Fix #25: Groq API limits tools to 128 per request
            if len(tools) > 128:
                LOGGER.warning(
                    "Too many tools (%d) for Groq API, truncating to 128",
                    len(tools),
                )
                tools = tools[:128]

        messages = _recortar_historial(
            _chat_log_to_messages(chat_log),
            options.get(CONF_HISTORY_BUDGET, RECOMMENDED_HISTORY_BUDGET),
        )

        LOGGER.debug("Prompt: %s", messages)
        LOGGER.debug("Tools: %s", tools)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "tools": llm_api.tools if llm_api else None},
        )

        clientes = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        tools_fallback_attempted = False
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                model_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": tools or NOT_GIVEN,
                    "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    "user": chat_log.conversation_id,
                }

                cadena = list(options.get(CONF_MODEL_CHAIN) or [])
                candidatos = _candidatos(
                    model, cadena, self._ultimo_uso,
                    float(options.get(CONF_MODEL_COOLDOWN,
                                      RECOMMENDED_MODEL_COOLDOWN)),
                )
                # Cloudflare va SIEMPRE al final y fuera de la rotación por
                # enfriamiento: es la red de última instancia, no un escalón
                # más. Es más lento, así que solo se usa si Groq no da.
                if clientes.hay_cf:
                    candidatos.append(PREFIJO_CF + options.get(
                        CONF_CF_MODEL, RECOMMENDED_CF_MODEL))
                result = None
                for pos, candidato in enumerate(candidatos):
                    es_cf = candidato.startswith(PREFIJO_CF)
                    nombre = candidato[len(PREFIJO_CF):] if es_cf else candidato
                    model_kwargs["model"] = nombre
                    if es_cf:
                        # Cloudflare no documenta reasoning_effort ni
                        # reasoning_format; mandarlos sería arriesgar un 400
                        # justo cuando es el último recurso que queda.
                        for clave in ("reasoning_format", "reasoning_effort",
                                      "include_reasoning"):
                            model_kwargs.pop(clave, None)
                    else:
                        _aplicar_razonamiento(model_kwargs, nombre, options)
                    try:
                        result = await _pedir_encogiendo(
                            clientes, model_kwargs, es_cf, candidato)
                    except groq.APIStatusError as err:
                        limitado = (err.status_code in (413, 429)
                                    or "rate_limit" in str(err))
                        # El modelo se marca como usado igual al fallar: su
                        # ventana de tokens está ocupada, que es justo lo que
                        # hay que recordar para no volver a elegirlo ya mismo.
                        self._ultimo_uso[candidato] = time.monotonic()
                        if limitado and pos + 1 < len(candidatos):
                            # Un 413 que llega hasta acá ya se encogió todo lo
                            # que se podía, así que rotar es lo último que
                            # queda aunque no sea probable que ayude.
                            LOGGER.warning(
                                "%s no pudo (HTTP %s: %s), salto a %s sin "
                                "esperar", candidato, err.status_code,
                                _detalle_error(err), candidatos[pos + 1],
                            )
                            continue
                        raise
                    self._ultimo_uso[candidato] = time.monotonic()

                    # Se reintenta UNA vez sin pensamiento. Una respuesta más
                    # seca es infinitamente mejor que el silencio, y volver a
                    # preguntarle al mismo modelo sale más barato que saltar:
                    # el siguiente de la cadena suele ser peor, y el salto
                    # gasta la ventana de otro modelo que quizá haga falta
                    # después. No aplica a Cloudflare, que va sin razonamiento.
                    if not es_cf and _vacia_por_truncado(result):
                        LOGGER.warning(
                            "%s gastó los %s tokens de max_tokens pensando y "
                            "volvió vacío; repito sin pensamiento",
                            candidato, model_kwargs["max_tokens"],
                        )
                        _aplicar_razonamiento(model_kwargs, nombre, options,
                                              forzar="none")
                        try:
                            result = await _pedir(clientes, model_kwargs, es_cf)
                        except groq.APIStatusError as err:
                            # Sin cupo para el reintento: nos quedamos con el
                            # vacío y que decida el salto de abajo.
                            LOGGER.warning(
                                "el reintento sin pensamiento de %s tampoco "
                                "entró (HTTP %s)", candidato, err.status_code,
                            )
                        self._ultimo_uso[candidato] = time.monotonic()

                    if _vacia_por_truncado(result) and pos + 1 < len(candidatos):
                        LOGGER.warning(
                            "%s sigue devolviendo vacío, salto a %s",
                            candidato, candidatos[pos + 1],
                        )
                        continue

                    model = candidato
                    break
                # Sin esto hay que adivinar cuánto pesa cada petición. El
                # límite de Groq es por minuto contando entrada + salida, así
                # que estos tres números son los que dicen si el recorte del
                # historial está bien calibrado o no.
                if (uso := getattr(result, "usage", None)) is not None:
                    # cached_tokens es EL número que decide si conviene partir
                    # el trabajo en dos modelos: los tokens cacheados no cuentan
                    # contra el límite por minuto. Si se queda cerca de 0, el
                    # caché no está pegando y no hay ahorro que perseguir.
                    detalle = getattr(uso, "prompt_tokens_details", None)
                    cacheados = getattr(detalle, "cached_tokens", None)
                    eleccion = (result.choices or [None])[0]
                    motivo = getattr(eleccion, "finish_reason", None)
                    texto = getattr(getattr(eleccion, "message", None),
                                    "content", None) or ""
                    # Llegar acá vacío significa que ya se probó todo: cada
                    # modelo de la cadena, y cada uno también sin pensamiento.
                    # El usuario va a escuchar silencio y esto es lo único que
                    # va a quedar escrito, así que va como ERROR.
                    if _vacia_por_truncado(result):
                        LOGGER.error(
                            "RESPUESTA VACÍA tras agotar la cadena entera: %s "
                            "gastó los %s tokens de max_tokens razonando y ni "
                            "sin pensamiento llegó a contestar. Subí "
                            "max_tokens o poné reasoning_effort en 'none'.",
                            model, uso.completion_tokens,
                        )
                    # WARNING a propósito: la config de Juan tiene
                    # `logger: default: warning`, así que con INFO esto no se
                    # vería. Bajar a INFO cuando termine la medición.
                    LOGGER.warning(
                        "Groq OK [%s] — entrada %s (cacheados %s), salida %s, "
                        "total %s (%s mensajes, fin=%s, %s car.)",
                        model, uso.prompt_tokens, cacheados,
                        uso.completion_tokens, uso.total_tokens, len(messages),
                        motivo, len(texto),
                    )
            except groq.BadRequestError as err:
                # Fix #20: Smaller models may produce malformed tool calls;
                # retry once without tools as a text-only fallback.
                if (
                    "tool_use_failed" in str(err)
                    and not tools_fallback_attempted
                    and tools
                ):
                    LOGGER.warning(
                        "Groq returned tool_use_failed error, retrying without "
                        "tools: %s",
                        err,
                    )
                    tools_fallback_attempted = True
                    tools = None
                    continue
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem talking to Groq: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )
            except groq.AuthenticationError as err:
                # Fix #27: Better auth error messaging
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Sorry, Groq authentication failed. Please check your API key "
                    "for leading/trailing whitespace or incorrect format.",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )
            except groq.APIStatusError as err:
                # Fix #19: Better error messaging for rate limit / payload errors
                error_str = str(err)
                if err.status_code == 413 or "rate_limit_exceeded" in error_str:
                    # El texto de Groq trae los números concretos
                    # ("Limit 8000, Requested 9234"). Sin loguearlo, el mensaje
                    # amable de abajo los tapa y no queda con qué calibrar.
                    LOGGER.error(
                        "Groq RECHAZÓ (HTTP %s) con %s mensajes tras el "
                        "recorte. Respuesta cruda: %s",
                        err.status_code, len(messages), error_str,
                    )
                    intent_response = intent.IntentResponse(
                        language=user_input.language
                    )
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        "Sorry, Groq rejected the request. "
                        "Try reducing max_tokens in the integration options or "
                        "upgrading your Groq API tier.",
                    )
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                    )
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem talking to Groq: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )
            except groq.GroqError as err:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem talking to Groq: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )

            LOGGER.debug("Response %s", result)
            response = result.choices[0].message

            groq_tool_calls = response.tool_calls or []
            assistant_tool_calls = [
                llm.ToolInput(
                    id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_args=json.loads(tool_call.function.arguments),
                )
                for tool_call in groq_tool_calls
            ]

            assistant_content = AssistantContent(
                agent_id=self.entity_id,
                content=response.content,
                tool_calls=assistant_tool_calls or None,
            )

            messages.append(_assistant_content_to_message(assistant_content))

            if not assistant_tool_calls:
                chat_log.async_add_assistant_content_without_tools(assistant_content)
                break

            for tool_call in assistant_tool_calls:
                LOGGER.debug(
                    "Tool call: %s(%s)", tool_call.tool_name, tool_call.tool_args
                )

            if llm_api is None:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Tool call requested but no LLM API configured",
                )
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )

            async for tool_result_content in chat_log.async_add_assistant_content(
                assistant_content
            ):
                LOGGER.debug(
                    "Tool response: %s -> %s",
                    tool_result_content.tool_name,
                    tool_result_content.tool_result,
                )
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=tool_result_content.tool_call_id,
                        content=json_dumps(tool_result_content.tool_result),
                    )
                )

        return async_get_result_from_chat_log(user_input, chat_log)
