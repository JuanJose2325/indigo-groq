#!/usr/bin/env python3
"""Pruebas del detector de respuesta vacía por truncado (`_vacia_por_truncado`).

Lo que se está protegiendo acá es el peor fallo que tuvo la integración, porque
no se parecía a un fallo: con `reasoning_effort` alto, un modelo de razonamiento
gasta los `max_tokens` enteros pensando y devuelve `finish_reason="length"` con
el contenido VACÍO. No hay excepción, no hay 429, el pipeline de Assist da la
vuelta completa como si todo hubiera salido bien, no emite `synthesize` y el
usuario se queda escuchando silencio. En los logs del micrófono no aparece nada.

Medido el 31 ago 2026 contra la instalación real: el modelo principal
(qwen3.8-27b) nunca truncaba, pero el de respaldo (qwen3.6-27b) devolvía
`length` las 2 de 2 veces, una de ellas con 0 caracteres.

El caso que más fácil se rompe al tocar esto es el del truncado CON tool_calls:
ahí la petición de herramienta está completa y sirve, así que no es este fallo.
Tratarlo como vacío haría reintentar llamadas a la casa que ya se pidieron.
"""

from types import SimpleNamespace

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_vacia_por_truncado"])


def respuesta(motivo, contenido=None, tool_calls=None):
    """Imita la forma de una ChatCompletion del SDK de Groq."""
    mensaje = SimpleNamespace(content=contenido, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=motivo, message=mensaje)]
    )


# 1. El caso real: cortado por longitud y sin una sola letra.
comprobar(
    "truncado y vacío es el fallo",
    M._vacia_por_truncado(respuesta("length", "")) is True,
)
comprobar(
    "content en None cuenta como vacío",
    M._vacia_por_truncado(respuesta("length", None)) is True,
)
comprobar(
    "solo espacios en blanco cuenta como vacío",
    M._vacia_por_truncado(respuesta("length", "  \n\t ")) is True,
)

# 2. Truncado PERO con texto: el usuario escucha una respuesta cortada, que es
#    molesto pero no es silencio. Reintentar le costaría otra ventana de cupo
#    para ganar poco, así que esto no se toca.
comprobar(
    "truncado con texto no es el fallo",
    M._vacia_por_truncado(respuesta("length", "Sí, podés cambiar el")) is False,
)

# 3. Un final normal jamás es este fallo, ni siquiera si viene vacío: eso es
#    otro problema (el modelo no quiso contestar) y se trata en otro lado.
comprobar(
    "fin normal no es el fallo",
    M._vacia_por_truncado(respuesta("stop", "Hola.")) is False,
)
comprobar(
    "fin normal vacío tampoco entra acá",
    M._vacia_por_truncado(respuesta("stop", "")) is False,
)

# 4. Truncado tras pedir una herramienta: la petición está completa y sirve.
#    Si esto devolviera True se repetirían acciones sobre la casa.
comprobar(
    "truncado con tool_calls NO es el fallo",
    M._vacia_por_truncado(
        respuesta("length", "", tool_calls=[SimpleNamespace(id="x")])
    ) is False,
)

# 5. Respuestas deformes: nunca deben reventar. Esto corre en el camino de la
#    respuesta al usuario, y una excepción acá se oye como un error hablado.
comprobar(
    "sin choices no revienta",
    M._vacia_por_truncado(SimpleNamespace(choices=[])) is False,
)
comprobar(
    "choices en None no revienta",
    M._vacia_por_truncado(SimpleNamespace(choices=None)) is False,
)
comprobar(
    "objeto sin choices no revienta",
    M._vacia_por_truncado(SimpleNamespace()) is False,
)
comprobar(
    "message en None no revienta",
    M._vacia_por_truncado(
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                 message=None)])
    ) is True,
)

resumen("detector de respuesta vacía por truncado")
