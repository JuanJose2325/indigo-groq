#!/usr/bin/env python3
"""Pruebas de la rotación de modelos (`_candidatos`).

Lo que se está protegiendo acá: los límites de Groq son POR MODELO, así que
cada modelo de la cadena aporta su propia ventana de tokens por minuto. La
rotación tiene que ocurrir ANTES de que Groq devuelva un 429 — esperar el
rechazo cuesta un viaje de red entero, y acá lo que manda es la latencia.

Y el detalle que más fácil se rompe: un modelo en enfriamiento NO se descarta,
se manda al final. Si se descartara, con todos calientes no quedaría ninguno.
"""

import time

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_candidatos"])

AHORA = time.monotonic()
PRINCIPAL = "qwen/qwen3.8-27b"
CADENA = ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"]


# 1. Sin uso previo, el orden es el configurado: principal primero.
comprobar(
    "sin historial de uso respeta el orden configurado",
    M._candidatos(PRINCIPAL, CADENA, {}, 60) == [PRINCIPAL, *CADENA],
)

# 2. El principal recién usado se va al final, no desaparece.
orden = M._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA}, 60)
comprobar(
    "el modelo caliente baja al final",
    orden[0] == CADENA[0] and orden[-1] == PRINCIPAL,
)
comprobar(
    "el modelo caliente sigue estando",
    set(orden) == {PRINCIPAL, *CADENA},
)

# 3. Con TODOS calientes se devuelven igual, en su orden de preferencia
#    original. Devolver la lista vacía dejaría al usuario sin respuesta.
calientes = {m: AHORA for m in [PRINCIPAL, *CADENA]}
comprobar(
    "todos calientes: no se pierde ninguno",
    M._candidatos(PRINCIPAL, CADENA, calientes, 60) == [PRINCIPAL, *CADENA],
)

# 4. Un uso viejo ya no cuenta como caliente.
comprobar(
    "pasado el enfriamiento vuelve a ser preferido",
    M._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA - 120}, 60)[0]
    == PRINCIPAL,
)

# 5. El enfriamiento en 0 desactiva la rotación preventiva.
comprobar(
    "enfriamiento 0 deja el orden configurado",
    M._candidatos(PRINCIPAL, CADENA, calientes, 0) == [PRINCIPAL, *CADENA],
)

# 6. Duplicados: el principal repetido dentro de la cadena no se prueba dos
#    veces (sería gastar un viaje de red en un modelo que ya falló).
comprobar(
    "quita duplicados conservando la primera aparición",
    M._candidatos(PRINCIPAL, [PRINCIPAL, CADENA[0]], {}, 60)
    == [PRINCIPAL, CADENA[0]],
)

# 7. Cadena vacía: queda solo el principal.
comprobar(
    "cadena vacía deja solo el principal",
    M._candidatos(PRINCIPAL, [], {}, 60) == [PRINCIPAL],
)

# 8. Entradas vacías en la cadena (campo de texto en blanco en la UI) se
#    ignoran; si no, se pediría un modelo llamado "" y sería un 400.
comprobar(
    "ignora entradas vacías de la cadena",
    M._candidatos(PRINCIPAL, ["", CADENA[0]], {}, 60)
    == [PRINCIPAL, CADENA[0]],
)

# 9. Entre varios fríos se conserva la preferencia relativa.
orden = M._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA, CADENA[0]: AHORA}, 60)
comprobar(
    "los calientes mantienen su orden relativo al final",
    orden == [CADENA[1], PRINCIPAL, CADENA[0]],
)

resumen("rotación de la cadena de modelos")
