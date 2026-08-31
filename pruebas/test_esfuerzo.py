#!/usr/bin/env python3
"""Pruebas de la traducción del esfuerzo de razonamiento (`_esfuerzo_para`).

Lo que se está protegiendo acá es un fallo real que se escuchó en voz alta:
la cadena de respaldo cruza de una familia de modelos a otra, y cada familia
acepta un vocabulario distinto. Saltar de un Qwen (que usa "default"/"none")
a un gpt-oss (que exige "low"/"medium"/"high") con el valor sin traducir
devuelve HTTP 400, y el mensaje de error terminaba leído por el TTS.
"""

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_EQUIVALENCIAS", "_esfuerzo_para"])

QWEN = "qwen/qwen3.8-27b"
OSS = "openai/gpt-oss-120b"
OTRO = "meta-llama/llama-4-scout-17b-16e-instruct"

VALIDOS_OSS = {"low", "medium", "high"}
VALIDOS_QWEN = {"default", "none"}


# 1. EL fallo real: "default" viniendo de Qwen no puede llegar así a gpt-oss.
comprobar(
    "default de Qwen se traduce a un valor que gpt-oss acepta",
    M._esfuerzo_para(OSS, "default") in VALIDOS_OSS,
)

# 2. Y concretamente a "medium", no a "high". En los modelos de razonamiento
#    los tokens de pensamiento salen de max_tokens: con un presupuesto chico,
#    "high" se lo come entero y la respuesta llega VACÍA.
comprobar(
    "default -> medium (high dejaría al usuario sin respuesta)",
    M._esfuerzo_para(OSS, "default") == "medium",
)

# 3. En el otro sentido: los valores de gpt-oss no rompen a Qwen.
for valor in VALIDOS_OSS:
    comprobar(
        f"gpt-oss {valor!r} -> valor válido para Qwen",
        M._esfuerzo_para(QWEN, valor) in VALIDOS_QWEN,
    )

# 4. "none" se respeta en las dos familias: es la única forma de apagar el
#    razonamiento, y convertirlo en esfuerzo bajo no sería lo mismo.
comprobar("none sobrevive en Qwen", M._esfuerzo_para(QWEN, "none") == "none")
comprobar("none en gpt-oss es el mínimo", M._esfuerzo_para(OSS, "none") == "low")

# 5. Valor desconocido (config vieja, modelo nuevo): devuelve algo válido, no
#    un 400. Vale más una respuesta con el esfuerzo equivocado que un error.
comprobar(
    "valor desconocido no llega crudo a Qwen",
    M._esfuerzo_para(QWEN, "ultra") in VALIDOS_QWEN,
)
comprobar(
    "valor desconocido no llega crudo a gpt-oss",
    M._esfuerzo_para(OSS, "ultra") in VALIDOS_OSS,
)

# 6. Familia no reconocida: se pasa tal cual, sin inventar traducciones.
comprobar(
    "familia desconocida pasa el valor sin tocar",
    M._esfuerzo_para(OTRO, "loquesea") == "loquesea",
)

# 7. El prefijo importa: el bug original comparaba contra "qwen/qwen3-32b"
#    (modelo deprecado), así que qwen3.8-27b no entraba y se quedaba sin
#    reasoning_format="hidden" — el pensamiento volvía dentro de la respuesta.
comprobar(
    "cualquier qwen/ entra en la familia qwen",
    M._esfuerzo_para("qwen/qwen3.6-27b", "default") == "default",
)

# 8. Toda la tabla es cerrada: ningún valor de salida cae fuera de lo válido.
comprobar(
    "la tabla de equivalencias no produce valores inválidos",
    set(M._EQUIVALENCIAS["qwen"].values()) <= VALIDOS_QWEN
    and set(M._EQUIVALENCIAS["gpt-oss"].values()) <= VALIDOS_OSS,
)

resumen("traducción del esfuerzo de razonamiento")
