"""Traducción entre los tipos de Home Assistant y los dicts que entiende Groq.

Frontera: este es el ÚNICO archivo donde viven los `isinstance` contra las
clases de `chat_log` y las clases del SDK de Groq. Todo lo que sale de acá son
dicts planos, y por eso `historial.py` puede quedarse puro y probable por AST.

Regla que no se rompe: nada de acá muta `chat_log` ni sus elementos. Esa lista
es la que Home Assistant persiste entre turnos y sus cuatro tipos de contenido
son dataclasses frozen; la limpieza del historial se hace después, sobre las
copias que devuelve `_chat_log_to_messages`.
"""

from __future__ import annotations

from groq.types.chat import (
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

from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.helpers import llm
from homeassistant.helpers.json import json_dumps

from .const import LOGGER


def _format_tool(tool: llm.Tool, custom_serializer: object | None) -> ChatCompletionToolParam:
    """Convierte una herramienta de Home Assistant al formato de herramientas de Groq."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    # La descripción se agrega solo si existe: mandar `description: None` es un
    # 400 de esquema, y hay herramientas de HA que no traen ninguna.
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _assistant_content_to_message(content: AssistantContent) -> dict:
    """Convierte un turno del asistente a mensaje de Groq, con sus tool_calls si las hubo."""
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
    # La clave `tool_calls` se agrega SOLO si hay: una lista vacía no es lo
    # mismo que ausente para la API.
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def _aporta_algo(texto: object, llamadas: object) -> bool:
    """¿Este turno del asistente vale los tokens de reenviarlo?

    Extraída del recorrido del historial para que tenga prueba: la que la llama
    usa `isinstance` contra tipos de Home Assistant y el arnés no la puede leer.

    Un turno sin texto y sin llamadas a herramienta no dice nada, y aun así se
    reenviaba en cada petición: cuando la cadena se agota y se acepta el vacío
    queda persistido un AssistantContent con `content=None`, que desde entonces
    viajaba como {"role": "assistant", "content": null} en todos los turnos
    siguientes de la sesión. Gastaba cupo del minuto para no decir nada, y hay
    APIs que rechazan un mensaje de asistente sin contenido ni llamadas.
    """
    if llamadas:
        # Con herramientas SIEMPRE se reenvía, aunque no venga texto: el
        # resultado de la herramienta que viene detrás se empareja por
        # `tool_call_id` y quedaría huérfano, que es un 400 seguro.
        return True
    return bool(texto and str(texto).strip())


def _chat_log_to_messages(chat_log: ChatLog) -> list[dict]:
    """Convierte el historial de Home Assistant a la lista de mensajes de Groq."""
    messages: list[dict] = []

    for content in chat_log.content:
        if isinstance(content, SystemContent):
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=content.content)
            )
        elif isinstance(content, UserContent):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.content)
            )
        elif isinstance(content, AssistantContent):
            if not _aporta_algo(content.content, content.tool_calls):
                continue
            # `thinking_content` y `native` se descartan a propósito: el
            # pensamiento no vuelve al historial. Reenviarlo sería pagar dos
            # veces los tokens que `reasoning_format="hidden"` ya evita, y en
            # familias que razonan el pensamiento de un turno viejo confunde al
            # modelo más de lo que lo ayuda.
            messages.append(_assistant_content_to_message(content))
        elif isinstance(content, ToolResultContent):
            # Ojo con el rol: en Home Assistant es "tool_result", en Groq es
            # "tool". La traducción la hace la integración, y `tool_name` se
            # descarta porque la API empareja por `tool_call_id`.
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=content.tool_call_id,
                    content=json_dumps(content.tool_result),
                )
            )

    return messages


def _limitar_herramientas(tools: list, tope: int = 128) -> list:
    """Recorta la lista de herramientas al máximo que acepta Groq, avisando por el log.

    Groq acepta como mucho 128 herramientas por petición. Assist manda del orden
    de una docena, así que en una instalación normal esto no recorta nunca: está
    para que una casa con cientos de entidades expuestas reciba una respuesta
    peor en vez de un 400 que el TTS le lee en voz alta al usuario.
    """
    if len(tools) <= tope:
        return tools
    # Va en warning y no en debug porque el usuario tiene `logger: default:
    # warning`: si se cae una herramienta, tiene que poder enterarse de por qué
    # el asistente dejó de poder hacer algo que antes hacía. Y el bloque de
    # herramientas ya pesa unos 4150 de los 8000 tokens por minuto, así que
    # pasarse de 128 también es una señal de que hay entidades de más expuestas.
    LOGGER.warning(
        "Se expusieron %d herramientas y Groq acepta %d; se mandan las primeras %d",
        len(tools), tope, tope,
    )
    return tools[:tope]
