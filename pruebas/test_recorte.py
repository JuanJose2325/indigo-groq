#!/usr/bin/env python3
"""Pruebas del recorte de historial (`_recortar_historial`).

Lo que se está protegiendo acá: el límite de Groq cuenta entrada + salida por
minuto, y como el historial entero se reenvía en cada turno, sin recorte la
conversación termina superando el límite y a partir de ahí falla SIEMPRE.

Se ejecuta con `python3 pruebas/test_recorte.py` — no hace falta pytest ni
tener Home Assistant instalado.
"""

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_coste_aproximado", "_recortar_historial"])


def usuario(texto):
    return {"role": "user", "content": texto}


def asistente(texto, tool_calls=None):
    m = {"role": "assistant", "content": texto}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def herramienta(texto):
    return {"role": "tool", "tool_call_id": "x", "content": texto}


SISTEMA = {"role": "system", "content": "s" * 350}  # ~104 tokens


# 1. Con presupuesto de sobra no se toca nada.
historial = [SISTEMA, usuario("hola"), asistente("qué tal"), usuario("bien")]
comprobar(
    "presupuesto amplio deja el historial intacto",
    M._recortar_historial(historial, 100_000) == historial,
)

# 2. El prompt de sistema NUNCA se descarta, aunque el resto no entre.
recortado = M._recortar_historial([SISTEMA, usuario("x" * 3500)], 150)
comprobar(
    "el sistema sobrevive al recorte",
    recortado[0] is SISTEMA and len(recortado) == 1,
)

# 3. Se conservan los turnos MÁS RECIENTES, no los primeros.
viejos = [usuario(f"pregunta {i}" + "z" * 300) for i in range(10)]
recortado = M._recortar_historial([SISTEMA, *viejos], 400)
comprobar(
    "se queda con la cola, no con la cabeza",
    recortado[-1] is viejos[-1] and viejos[0] not in recortado,
)

# 4. Si ni el prompt de sistema entra, devuelve todo sin tocar: recortar no
#    arregla nada y mandar la conversación vacía sería peor.
entrada = [SISTEMA, usuario("hola")]
comprobar(
    "sistema más grande que el presupuesto: no recorta",
    M._recortar_historial(entrada, 10) == entrada,
)

# 5. La ventana nunca puede empezar por un resultado de herramienta huérfano:
#    la API devuelve 400 si ve un `tool` sin su `assistant` con `tool_calls`.
llamada = {"id": "a", "type": "function",
           "function": {"name": "luz", "arguments": "{}"}}
conv = [
    SISTEMA,
    usuario("apagá la luz" + "y" * 400),
    asistente(None, [llamada]),
    herramienta("ok"),
    asistente("Listo"),
]
recortado = M._recortar_historial(conv, 250)
sin_sistema = [m for m in recortado if m["role"] != "system"]
comprobar(
    "no deja tool results huérfanos",
    not sin_sistema or sin_sistema[0]["role"] == "user",
)

# 6. El coste cuenta también los argumentos de las tool_calls, no solo el texto.
comprobar(
    "las tool_calls suman al coste estimado",
    M._coste_aproximado(asistente(None, [llamada]))
    > M._coste_aproximado(asistente(None)),
)

resumen("recorte de historial")
