"""Entidad de conversación de Groq y orquestación de un turno.

Frontera de este módulo: es el ÚNICO archivo que ve `hass`, `self` y
`chat_log`. Acá no se calcula nada que se pueda calcular en un módulo puro;
solo se pegan piezas y se traduce entre lo que pide Home Assistant y lo que
deciden los módulos de decisión. Por eso ninguna función de este archivo se
carga por AST desde `pruebas/cargar.py`: todo lo que vale la pena probar vive
en `cadena.py`, `historial.py`, `respuestas.py`, `razonamiento.py` y
`enrutadores.py`.
"""

from __future__ import annotations

from typing import Any, Literal

import groq
from groq._types import NOT_GIVEN

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConverseError,
    async_get_result_from_chat_log,
    trace,
)
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import GroqConfigEntry
from .cadena import _candidatos
from .const import (
    CONF_CHAT_MODEL,
    CONF_HISTORY_BUDGET,
    CONF_MAX_TOKENS,
    CONF_MODEL_CHAIN,
    CONF_MODEL_COOLDOWN,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HISTORY_BUDGET,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MODEL_COOLDOWN,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    TOPE_MINIMO_TOKENS,
    TURNOS_CONTEXTO_CASA,
)
from .enrutadores import _ajustes_enrutadores, _decidir_enrutadores
from .historial import _recortar_historial, _sin_herramientas, _ultimos_turnos
from .mensajes import (
    _assistant_content_to_message,
    _chat_log_to_messages,
    _format_tool,
    _limitar_herramientas,
)
from .peticiones import _responder_con_cadena
from .respuestas import (
    _argumentos_de_herramienta,
    _detalle_error,
    _peticiones_de_herramienta,
    _resumen_uso,
    _texto_de,
    _vacia_por_truncado,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GroqConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Registra la entidad de conversación de la entrada."""
    async_add_entities([GroqConversationEntity(config_entry)])


class GroqConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Agente de conversación de Groq con enrutadores en paralelo y cadena de respaldo."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: GroqConfigEntry) -> None:
        """Guarda la entrada y el registro de cuándo se usó cada modelo."""
        self.entry = entry
        # Cuándo se usó por última vez cada modelo, para rotar la cadena ANTES
        # de que Groq devuelva un 429. Se pierde entero cada vez que el usuario
        # guarda las opciones (el listener recarga la entrada y recrea la
        # entidad), y está bien: lo peor que pasa es un 429 evitable en el
        # primer turno después de tocar la config.
        self._ultimo_uso: dict[str, float] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Groq",
            model="Groq Cloud",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # Es una propiedad de la ENTRADA, no del turno: dice si el usuario
        # configuró un API de asistente, no si el enrutador de casa lo va a
        # usar en esta pregunta. El enrutador NO la toca; si dependiera del
        # turno, la tarjeta de Assist parpadearía entre "controla la casa" y
        # "no controla la casa" según lo que se le pregunte.
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Todos los idiomas: quien traduce es el modelo, no la integración."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Se registra como agente de conversación."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Se da de baja como agente de conversación."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _nombres_expuestos(self) -> list[str]:
        """Los nombres de las entidades expuestas a conversación.

        Es lo único que ve el enrutador de casa de la instalación: los NOMBRES
        y nada más. El volcado YAML completo con áreas y alias es justamente lo
        que se está tratando de ahorrar, así que mandárselo al enrutador sería
        pagar dos veces el precio que se quiere evitar.

        El cálculo es síncrono y sin red, así que no cuesta nada dentro del
        turno. Ante cualquier problema devuelve la lista vacía en vez de
        levantar: el enrutador ya es fail-safe hacia "sí, conservá las
        herramientas", así que quedarse sin nombres degrada la decisión pero no
        deja al usuario sin control de la casa.
        """
        try:
            from homeassistant.components.homeassistant.llm import (  # noqa: PLC0415
                async_get_exposed_entities,
            )

            expuestas = async_get_exposed_entities(
                self.hass, "conversation", include_state=False
            )
            nombres = [
                str(datos["names"])
                for datos in (expuestas or {}).values()
                if isinstance(datos, dict) and datos.get("names")
            ]
        except ImportError:
            # `async_get_exposed_entities` se mudó hace poco de `helpers/llm.py`
            # a `components/homeassistant/llm.py`. Si el core la vuelve a mover,
            # abajo está la API que él mismo usa por debajo, que lleva mucho más
            # tiempo estable.
            LOGGER.debug("Sin async_get_exposed_entities; uso el plan B")
            return self._nombres_por_estado()
        except Exception:  # noqa: BLE001
            LOGGER.debug("No pude leer las entidades expuestas", exc_info=True)
            return self._nombres_por_estado()
        return nombres

    def _nombres_por_estado(self) -> list[str]:
        """Plan B: filtra `hass.states` con `async_should_expose`."""
        try:
            from homeassistant.components.homeassistant.exposed_entities import (  # noqa: PLC0415
                async_should_expose,
            )

            return [
                estado.name
                for estado in self.hass.states.async_all()
                if async_should_expose(self.hass, "conversation", estado.entity_id)
            ]
        except Exception:  # noqa: BLE001
            LOGGER.debug("Tampoco pude listar por estado", exc_info=True)
            return []

    def _resultado_de_error(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        texto: str,
    ) -> conversation.ConversationResult:
        """Un resultado de error hablado, sin agregar contenido de asistente.

        Volver por acá no agrega nada al chat log por su cuenta. Home Assistant
        descarta el turno entero comparando si el último elemento del historial
        sigue siendo el que había al entrar, así que el mensaje del usuario que
        disparó el error no queda pegado reenviándose en todos los turnos
        siguientes.

        La garantía vale MIENTRAS el turno no haya ejecutado ninguna
        herramienta. Si el bucle ya dio una vuelta, `async_add_assistant_content`
        dejó su AssistantContent con las `tool_calls` y el ToolResultContent
        correspondiente, el último elemento ya cambió y el descarte no ocurre:
        el turno queda en el historial con la acción que de verdad se ejecutó
        sobre la casa, que es lo correcto —deshacerla no es opción— pero no es
        lo que decía este docstring.
        """
        respuesta = intent.IntentResponse(language=user_input.language)
        respuesta.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, texto)
        return conversation.ConversationResult(
            response=respuesta, conversation_id=chat_log.conversation_id
        )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Atiende un turno: dos enrutadores en paralelo y después la llamada final."""
        options = self.entry.options
        cliente = self.entry.runtime_data.groq
        ajustes = _ajustes_enrutadores(options)

        # PRIMERA conversión del historial, más la lista de entidades expuestas.
        # Las dos cosas existen SOLO para armarle la entrada al enrutador de
        # casa, así que no se pagan cuando está apagado: una instalación que
        # viene de la versión anterior lo tiene en falso, y ahí el turno tiene
        # que ser indistinguible del de antes —lo que incluye no hacer trabajo
        # que antes no se hacía. Recorrer `hass.states` entero es barato, pero
        # es barato POR TURNO, y este código corre en cada frase que se dice.
        contexto = ""
        expuestas: list[str] = []
        if ajustes["casa_activo"]:
            # En este punto `content[0]` todavía es el prompt de sistema del
            # turno ANTERIOR, porque `async_provide_llm_data` no corrió: esta
            # lista NO sirve para pedirle nada al modelo. Y hacen falta los DOS
            # lados de la conversación: "apagala" no se puede clasificar sin ver
            # el "sí, una" que contestó la IA antes.
            contexto = _ultimos_turnos(
                _chat_log_to_messages(chat_log), TURNOS_CONTEXTO_CASA
            )
            expuestas = self._nombres_expuestos()

        # Los dos enrutadores tienen que terminar ANTES de tocar el chat log:
        # su veredicto es lo que decide qué se le pasa a
        # `async_provide_llm_data`, y esa llamada tiene que ser una sola por
        # turno y la primera cosa que toque el log.
        casa, razon = await _decidir_enrutadores(
            cliente,
            ajustes,
            user_input.text,
            expuestas,
            contexto,
        )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                # Pasar None acá no borra solo las herramientas: borra también
                # el `api_prompt`, que es donde va el volcado YAML de todas las
                # entidades expuestas con su área. Por eso el enrutador de casa
                # pregunta "¿tiene que ver con la casa?" y no "¿es una orden?":
                # sin ese bloque, "¿tengo luces en el dormitorio?" (que no es
                # una orden) se contesta mal.
                options.get(CONF_LLM_HASS_API) if casa[0] else None,
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except ConverseError as err:
            return err.as_conversation_result()

        llm_api = chat_log.llm_api
        tools: list | None = None
        if llm_api:
            tools = _limitar_herramientas(
                [_format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools]
            )

        # SEGUNDA conversión, a propósito. Recién ahora `content[0]` tiene el
        # prompt de sistema de ESTE turno, que es el que corresponde mandar.
        # Convertir es barato; mandar el prompt del turno anterior no lo es.
        messages = _chat_log_to_messages(chat_log)
        if not tools:
            # Ir sin herramientas obliga a sacar también los `role="tool"` y
            # las `tool_calls` viejas: un resultado de herramienta sin su
            # petición es un 400. La limpieza va sobre los dicts de Groq y
            # NUNCA sobre `chat_log.content`, que es la lista que HA persiste y
            # cuyos elementos son dataclasses frozen.
            #
            # La guarda mira si el turno LLEVA herramientas, no el veredicto del
            # enrutador, y la diferencia importa: `tools` también queda vacío
            # cuando `llm_hass_api` no está configurado, y ahí el veredicto vale
            # "sí" (con el enrutador apagado es su valor de reposo). Atarla al
            # veredicto dejaba ese camino sin limpiar, con un historial que
            # todavía puede traer resultados de herramienta de cuando la API sí
            # estaba puesta. Mirar el efecto en vez de la causa cubre los dos.
            messages = _sin_herramientas(messages)
        # Limpiar primero y recortar después, en ese orden: al revés el
        # presupuesto se calibraría contra mensajes que están por desaparecer, y
        # la regla de "la ventana empieza en un user" quedaría aplicada sobre
        # una lista que después cambia.
        messages = _recortar_historial(
            messages, options.get(CONF_HISTORY_BUDGET, RECOMMENDED_HISTORY_BUDGET)
        )

        # El enrutador de razonamiento no elige CUÁNTO piensa el modelo: elige
        # SI piensa. El cuánto sale del campo del principal, y ese campo vacío
        # significa "no configurado" (se manda reasoning_format="hidden" y
        # ningún reasoning_effort), que no es lo mismo que mandar el máximo.
        # Medido: el titular contesta pensando en 120-180 tokens; el suplente
        # quema los 1200 razonando y vuelve vacío.
        esfuerzo = options.get(CONF_REASONING_EFFORT) if razon[0] else "none"

        principal = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        cadena = list(options.get(CONF_MODEL_CHAIN) or [])
        enfriamiento = float(
            options.get(CONF_MODEL_COOLDOWN, RECOMMENDED_MODEL_COOLDOWN)
        )

        # El diccionario de la petición se arma UNA SOLA VEZ por turno. Antes se
        # reconstruía desde `options` en cada vuelta del bucle de herramientas,
        # así que el `max_tokens` que `_pedir_encogiendo` había bajado por un
        # 413 se perdía en la iteración siguiente y el 413 volvía a pasar,
        # vuelta tras vuelta, hasta agotar las diez.
        model_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools or NOT_GIVEN,
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "user": chat_log.conversation_id,
        }

        LOGGER.debug("Mensajes: %s", messages)
        LOGGER.debug("Herramientas: %s", tools)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "tools": llm_api.tools if llm_api else None},
        )

        # Una sola vez por turno se puede pasar a modo texto: o porque Groq
        # rechazó la tool_call con `tool_use_failed`, o porque los argumentos
        # que devolvió el modelo no son JSON. Dos veces sería quedarse dando
        # vueltas sin herramientas y sin nada nuevo que probar.
        respaldo_sin_herramientas = False

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                # Se recalculan en cada vuelta a propósito: `self._ultimo_uso`
                # se va marcando dentro de `_responder_con_cadena`, así que la
                # segunda llamada del mismo turno ya sabe qué modelo acaba de
                # gastar su ventana.
                candidatos = _candidatos(
                    principal, cadena, self._ultimo_uso, enfriamiento
                )
                if not candidatos:
                    # Sin ningún modelo válido no hay a quién preguntarle. Se
                    # corta acá y no en `_responder_con_cadena` porque este es
                    # el único lugar que puede convertirlo en algo que el
                    # usuario escuche: una excepción que no sea
                    # HomeAssistantError se escapa de `async_converse` y rompe
                    # el pipeline entero de Assist, no solo este turno.
                    LOGGER.error(
                        "No hay ningún modelo configurado: revisá el modelo "
                        "principal y la cadena de respaldo en las opciones."
                    )
                    return self._resultado_de_error(
                        user_input,
                        chat_log,
                        "Perdón, no tengo ningún modelo configurado.",
                    )

                result = await _responder_con_cadena(
                    cliente,
                    model_kwargs,
                    candidatos,
                    principal,
                    esfuerzo,
                    self._ultimo_uso,
                    TOPE_MINIMO_TOKENS,
                )

                # WARNING y no INFO: el usuario tiene `logger: default: warning`
                # en configuration.yaml y audita el cupo con ~/simular-assist/
                # cupo.sh. Con INFO esta línea no existe para él, y sin ella hay
                # que adivinar cuánto pesa cada petición contra los 8000 TPM.
                if resumen := _resumen_uso(
                    result, len(messages), str(model_kwargs.get("model") or principal)
                ):
                    LOGGER.warning("%s", resumen)

                if _vacia_por_truncado(result):
                    # Llegar acá vacío significa que ya se probó todito: cada
                    # modelo de la cadena, y cada uno también sin pensamiento. El
                    # usuario va a escuchar silencio —no una excepción, silencio—
                    # y esta línea es lo único que va a quedar escrito.
                    LOGGER.error(
                        "RESPUESTA VACÍA tras agotar la cadena entera: %s gastó "
                        "los %s tokens de max_tokens razonando y ni sin "
                        "pensamiento llegó a contestar. Subí max_tokens o poné "
                        "el esfuerzo de razonamiento en 'none'.",
                        model_kwargs.get("model"),
                        model_kwargs.get("max_tokens"),
                    )
            except groq.BadRequestError as err:
                # Los modelos chicos devuelven de vez en cuando una tool_call
                # que la propia API rechaza. Repetir el turno en modo texto da
                # una respuesta peor, pero da una respuesta.
                if (
                    "tool_use_failed" in str(err)
                    and not respaldo_sin_herramientas
                    and tools
                ):
                    LOGGER.warning(
                        "Groq devolvió tool_use_failed; repito sin herramientas: %s",
                        _detalle_error(err),
                    )
                    respaldo_sin_herramientas = True
                    tools = None
                    # Este camino pasa de con-herramientas a sin-herramientas a
                    # mitad de turno, igual que el enrutador de casa, así que
                    # tiene que rehacer la lista: si no, quedan `role="tool"`
                    # huérfanos de la vuelta anterior y la API los rechaza con
                    # otro 400, esta vez sin respaldo que lo salve.
                    messages = _sin_herramientas(messages)
                    model_kwargs["messages"] = messages
                    model_kwargs["tools"] = NOT_GIVEN
                    continue
                return self._resultado_de_error(
                    user_input,
                    chat_log,
                    f"Perdón, tuve un problema hablando con Groq: {_detalle_error(err)}",
                )
            except groq.AuthenticationError:
                return self._resultado_de_error(
                    user_input,
                    chat_log,
                    "Perdón, Groq rechazó la clave de API. Fijate que no tenga "
                    "espacios de más al principio o al final.",
                )
            except groq.APIStatusError as err:
                if err.status_code in (413, 429) or "rate_limit" in str(err):
                    # `_detalle_error` y no `str(err)`: el cuerpo trae "Limit
                    # 8000, Requested 8441", que son los dos números con los que
                    # se calibran max_tokens y el presupuesto de historial. Con
                    # el código de estado solo, un 413 y un 429 se leen igual
                    # siendo problemas opuestos: uno se arregla pidiendo menos,
                    # el otro esperando.
                    LOGGER.error(
                        "Groq RECHAZÓ (HTTP %s) con %s mensajes tras el recorte: %s",
                        err.status_code,
                        len(messages),
                        _detalle_error(err),
                    )
                    return self._resultado_de_error(
                        user_input,
                        chat_log,
                        "Perdón, Groq rechazó la petición. Probá bajando "
                        "max_tokens en las opciones de la integración, o sumá "
                        "otro modelo a la cadena de respaldo.",
                    )
                return self._resultado_de_error(
                    user_input,
                    chat_log,
                    f"Perdón, tuve un problema hablando con Groq: {_detalle_error(err)}",
                )
            except groq.GroqError as err:
                return self._resultado_de_error(
                    user_input,
                    chat_log,
                    f"Perdón, tuve un problema hablando con Groq: {err}",
                )

            # Lectura tolerante del contenido: `_responder_con_cadena` puede
            # devolver una respuesta vacía cuando ya se agotó todo, y romper acá
            # con un AttributeError sonaría como un error hablado.
            #
            # Va por `_texto_de` y no por un getattr suelto para heredar su
            # filtro de tipo. Un `content` que no sea texto —una lista de
            # bloques, como devuelven algunas APIs— entraba crudo en el
            # AssistantContent y de ahí a `async_set_speech`, así que el TTS
            # terminaba tratando de pronunciar una estructura de datos. El
            # `or None` conserva la distinción que usa el resto del turno:
            # cadena vacía y "no hubo texto" son lo mismo acá.
            texto = _texto_de(result) or None

            llamadas: list[llm.ToolInput] = []
            malformada = False
            for peticion in _peticiones_de_herramienta(result):
                funcion = getattr(peticion, "function", None)
                argumentos = _argumentos_de_herramienta(
                    getattr(funcion, "arguments", None)
                )
                if argumentos is None:
                    malformada = True
                    break
                llamadas.append(
                    llm.ToolInput(
                        id=peticion.id,
                        tool_name=funcion.name,
                        tool_args=argumentos,
                    )
                )

            if malformada:
                # Antes esto era un `json.loads` pelado: el JSONDecodeError
                # salía fuera de todos los except del bucle y rompía el pipeline
                # de Assist. Unos argumentos que no parsean son exactamente el
                # mismo problema que un `tool_use_failed`, así que van por el
                # mismo camino.
                if respaldo_sin_herramientas or not tools:
                    LOGGER.error(
                        "El modelo devolvió argumentos de herramienta que no son "
                        "JSON y ya se había probado el modo texto; abandono el turno."
                    )
                    return self._resultado_de_error(
                        user_input,
                        chat_log,
                        "Perdón, no entendí la acción que quiso ejecutar el modelo.",
                    )
                LOGGER.warning(
                    "Argumentos de herramienta malformados; repito sin herramientas"
                )
                respaldo_sin_herramientas = True
                tools = None
                messages = _sin_herramientas(messages)
                model_kwargs["messages"] = messages
                model_kwargs["tools"] = NOT_GIVEN
                continue

            assistant_content = AssistantContent(
                agent_id=self.entity_id,
                content=texto,
                tool_calls=llamadas or None,
            )
            messages.append(_assistant_content_to_message(assistant_content))

            if not llamadas:
                chat_log.async_add_assistant_content_without_tools(assistant_content)
                break

            if llm_api is None:
                # Red que atrapa el caso nuevo: el enrutador de casa dijo que NO
                # y el modelo pidió una herramienta igual. Sin este chequeo,
                # `async_add_assistant_content` levanta
                # ValueError("No LLM API configured"), que NO es
                # HomeAssistantError: se escapa de `async_converse` y rompe el
                # pipeline entero de Assist, no solo este turno.
                LOGGER.warning(
                    "El modelo pidió %d herramienta(s) en un turno resuelto sin "
                    "control de la casa (motivo del enrutador: %s)",
                    len(llamadas),
                    casa[2],
                )
                return self._resultado_de_error(
                    user_input,
                    chat_log,
                    "Perdón, quise hacer algo en la casa pero este turno no "
                    "tenía el control habilitado. Probá de nuevo.",
                )

            async for tool_result_content in chat_log.async_add_assistant_content(
                assistant_content
            ):
                LOGGER.debug(
                    "Resultado de %s: %s",
                    tool_result_content.tool_name,
                    tool_result_content.tool_result,
                )
                # Dict plano y no el tipo del SDK: esta lista solo viaja hacia
                # `_recortar_historial` y hacia la API, y las dos tratan los
                # mensajes como dicts.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result_content.tool_call_id,
                        "content": json_dumps(tool_result_content.tool_result),
                    }
                )

        return async_get_result_from_chat_log(user_input, chat_log)
