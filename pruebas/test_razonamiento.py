#!/usr/bin/env python3
"""Pruebas de los parámetros de razonamiento: qué se manda, a quién y traducido cómo.

EL FALLO REAL QUE ESTO PROTEGE SON DOS, y los dos se escuchan por el parlante.

El primero es un HTTP 400 leído en voz alta. La cadena de respaldo cruza de una
familia de modelos a otra, y cada familia acepta un vocabulario de esfuerzo
distinto: saltar de un Qwen (que usa "default"/"none") a un gpt-oss (que exige
"low"/"medium"/"high") con el valor sin traducir devuelve 400, y el mensaje de
error terminaba siendo la respuesta hablada del asistente. La otra mitad del
mismo fallo es al revés: mandarle `reasoning_effort` a un modelo que NO razona
—`llama-3.3-70b-versatile`, por ejemplo— también es un 400, y ahora que la
cadena se ensancha con familias distintas eso pasa en cualquier eslabón.

El segundo es el silencio. Los tokens de pensamiento SALEN de `max_tokens`, así
que un esfuerzo demasiado alto se come el presupuesto entero y la respuesta
vuelve VACÍA: ninguna excepción, ningún error en el log, y el TTS no dice nada.
Medido en la instalación real: el principal contesta pensando en 120-180 tokens
y un respaldo quema los 1200 de `max_tokens` razonando y vuelve sin nada. De ahí
salen las dos precauciones que se prueban acá: "default" se traduce a "medium" y
nunca a "high", y los respaldos van con "none" mientras el principal conserva el
esfuerzo decidido.

Y el tercero, que no es un error sino algo peor porque parece que funciona: sin
`reasoning_format="hidden"` el pensamiento vuelve DENTRO de `content`, en inglés,
con la respuesta real pegada al final, y el TTS lo lee todo en voz alta.

Cubre los comportamientos 4 (traducción entre familias), 5 (familia desconocida
no recibe nada), 6 (campo vacío != valor inválido), 7 (formato oculto siempre) y
8 (higiene del diccionario entre candidatos), más la política de
`_esfuerzo_del_candidato`. Absorbe lo que probaba `test_esfuerzo.py`.
"""

from cargar import cargar
from runner import comprobar, resumen

# Los cinco nombres van en UNA sola llamada porque comparten el mismo dict de
# globals: `_aplicar_razonamiento` llama a `_esfuerzo_para`, que llama a
# `_familia_de`, que lee `_EQUIVALENCIAS`. Pedir de a uno da NameError recién al
# ejecutar, que es cuando ya no se entiende de dónde salió.
M = cargar(
    [
        "_EQUIVALENCIAS",
        "_familia_de",
        "_esfuerzo_para",
        "_aplicar_razonamiento",
        "_esfuerzo_del_candidato",
    ],
    "razonamiento",
)

QWEN = "qwen/qwen3.6-27b"
QWEN_VIEJO = "qwen/qwen3.8-27b"
OSS = "openai/gpt-oss-120b"
OSS_CHICO = "openai/gpt-oss-20b"
LLAMA = "llama-3.3-70b-versatile"
SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"

VALIDOS_QWEN = {"default", "none"}
VALIDOS_OSS = {"low", "medium", "high"}

# Las tres claves que la función tiene permitido tocar. El filtro es parte de la
# prueba: si algún día escribiera una cuarta, el check de "no pisa el resto de la
# petición" lo levanta.
CLAVES = ("reasoning_format", "reasoning_effort", "include_reasoning")


def aplicar(modelo, esfuerzo, previo=None):
    """Corre `_aplicar_razonamiento` sobre un dict y devuelve solo sus tres claves."""
    kwargs = dict(previo or {})
    M._aplicar_razonamiento(kwargs, modelo, esfuerzo)
    return {k: v for k, v in kwargs.items() if k in CLAVES}


# ---------------------------------------------------------------------------
# COMPORTAMIENTO 4 — Traducción entre familias
#
# Lo que era `test_esfuerzo.py`. `_esfuerzo_para` decide QUÉ valor tendría el
# esfuerzo; `_aplicar_razonamiento` decide SI se manda y con qué compañía. La
# distinción importa: un valor puede ser perfectamente válido y aun así no
# corresponder mandarlo.
# ---------------------------------------------------------------------------

# 1. EL fallo real: "default" viniendo de Qwen no puede llegar así a gpt-oss.
comprobar(
    "default de Qwen se traduce a un valor que gpt-oss acepta",
    M._esfuerzo_para(OSS, "default") in VALIDOS_OSS,
)

# 2. Y concretamente a "medium", NO a "high". Los tokens de pensamiento salen de
#    max_tokens: con el presupuesto chico, "high" se lo come entero y la
#    respuesta llega vacía. Traducir el máximo de una familia al máximo de la
#    otra sería lo intuitivo y es justo lo que deja al usuario sin respuesta.
comprobar(
    "default -> medium (high dejaría al usuario sin respuesta)",
    M._esfuerzo_para(OSS, "default") == "medium",
)

# 3. En el otro sentido: los tres valores de gpt-oss tampoco rompen a Qwen. La
#    traducción tiene que ser bidireccional porque la cadena cruza para los dos
#    lados según qué modelo esté en enfriamiento.
for valor in sorted(VALIDOS_OSS):
    comprobar(
        f"gpt-oss {valor!r} -> valor válido para Qwen",
        M._esfuerzo_para(QWEN, valor) in VALIDOS_QWEN,
    )

# 4. "none" se respeta en las dos familias: es la única forma de apagar el
#    razonamiento, y convertirlo en esfuerzo bajo no sería lo mismo. En gpt-oss
#    no existe "none", así que lo más cerca que se llega es su mínimo.
comprobar("none sobrevive en Qwen", M._esfuerzo_para(QWEN, "none") == "none")
comprobar("none en gpt-oss es el mínimo", M._esfuerzo_para(OSS, "none") == "low")

# 5. Valor desconocido —config vieja, modelo nuevo—: devuelve algo válido, no un
#    400. Vale más una respuesta con el esfuerzo equivocado que un error hablado.
#    Emite un LOGGER.warning, que es la línea que se ve en el log de la Pi.
comprobar(
    "valor desconocido no llega crudo a Qwen",
    M._esfuerzo_para(QWEN, "ultra") in VALIDOS_QWEN,
)
comprobar(
    "valor desconocido no llega crudo a gpt-oss",
    M._esfuerzo_para(OSS, "ultra") in VALIDOS_OSS,
)

# 6. Familia no reconocida: se pasa tal cual, sin inventar traducciones. El
#    filtrado de verdad —no mandar nada— lo hace `_aplicar_razonamiento`, y esa
#    división de trabajo es deliberada: acá no se sabe si el valor se va a usar.
comprobar(
    "familia desconocida pasa el valor sin tocar",
    M._esfuerzo_para(SCOUT, "loquesea") == "loquesea",
)

# 7. La detección es por PREFIJO y nunca por igualdad. El bug original comparaba
#    contra "qwen/qwen3-32b" (deprecado por Groq en jun 2026), así que
#    qwen3.8-27b quedaba afuera de la familia y se iba sin
#    reasoning_format="hidden": el pensamiento volvía dentro de la respuesta.
comprobar(
    "cualquier qwen/ entra en la familia qwen",
    M._familia_de(QWEN) == "qwen" and M._familia_de(QWEN_VIEJO) == "qwen",
)
comprobar(
    "los dos tamaños de gpt-oss entran en la misma familia",
    M._familia_de(OSS) == "gpt-oss" and M._familia_de(OSS_CHICO) == "gpt-oss",
)
comprobar(
    "un modelo que no razona no tiene familia",
    M._familia_de(LLAMA) is None and M._familia_de(SCOUT) is None,
)

# 8. La tabla es CERRADA: ninguna entrada puede producir un valor que la API
#    rechace. Es una propiedad estructural, no un caso: cubre también las
#    entradas que hoy nadie ejercita.
comprobar(
    "la tabla de equivalencias no produce valores inválidos",
    set(M._EQUIVALENCIAS["qwen"].values()) <= VALIDOS_QWEN
    and set(M._EQUIVALENCIAS["gpt-oss"].values()) <= VALIDOS_OSS,
)


# ---------------------------------------------------------------------------
# COMPORTAMIENTO 5 — Familia desconocida no recibe NADA
#
# Antes esto caía en un `elif` que le mandaba reasoning_effort igual. La
# estructura `if qwen / elif gpt-oss / (nada)` es load-bearing: el default del
# `if` es no mandar, no mandar lo genérico.
# ---------------------------------------------------------------------------

# 9. El caso exacto: un llama con un esfuerzo perfectamente válido no recibe una
#    sola clave de razonamiento.
comprobar(
    "un modelo de familia desconocida no recibe parámetros de razonamiento",
    aplicar(LLAMA, "default") == {},
)

# 10. Y tampoco por la puerta de atrás: "none" es un valor legítimo, pero un
#     modelo que no razona rechaza el campo aunque diga que no razone. La guarda
#     de familia gana siempre, venga el esfuerzo de donde venga.
comprobar(
    "un esfuerzo 'none' explícito tampoco se le manda a un modelo que no razona",
    aplicar(LLAMA, "none") == {},
)
comprobar(
    "el otro no-razonador tampoco recibe nada",
    aplicar(SCOUT, "high") == {},
)


# ---------------------------------------------------------------------------
# COMPORTAMIENTO 7 — reasoning_format="hidden" SIEMPRE
#
# Sin esta clave el pensamiento vuelve dentro de `content` y el TTS lo lee en voz
# alta. No es un error: es una respuesta válida, larga, en inglés y con lo que el
# usuario preguntó al final de todo.
# ---------------------------------------------------------------------------

# 11. Las dos familias razonadoras lo reciben.
comprobar(
    "Qwen recibe el formato oculto",
    aplicar(QWEN, "default").get("reasoning_format") == "hidden",
)
comprobar(
    "gpt-oss recibe el formato oculto",
    aplicar(OSS, "default").get("reasoning_format") == "hidden",
)

# 12. En gpt-oss include_reasoning=False NO alcanza —con eso solo, devolvía la
#     cadena de pensamiento entera dentro de content—, pero se manda igual junto
#     con el formato. El que de verdad oculta es reasoning_format.
comprobar(
    "gpt-oss recibe además include_reasoning en falso",
    aplicar(OSS, "default").get("include_reasoning") is False,
)
comprobar(
    "Qwen no recibe include_reasoning, que no es de su familia",
    "include_reasoning" not in aplicar(QWEN, "default"),
)

# 13. El esfuerzo que acompaña sale traducido al vocabulario propio de cada uno:
#     es la composición con `_esfuerzo_para` la que se está comprobando.
comprobar(
    "gpt-oss recibe un esfuerzo de su propio vocabulario",
    aplicar(OSS, "default").get("reasoning_effort") in VALIDOS_OSS,
)
comprobar(
    "Qwen recibe un esfuerzo de su propio vocabulario",
    aplicar(QWEN, "high").get("reasoning_effort") in VALIDOS_QWEN,
)

# 14. Apagar el pensamiento no es apagar el ocultamiento. Este es el check que
#     antes protegía el interruptor global `supports_reasoning`, que ya no
#     existe: era un flag para toda la integración y por eso apagaba
#     reasoning_format="hidden" incluso para un Qwen de la cadena, devolviendo el
#     pensamiento al TTS. Ahora el soporte lo decide la familia, y pedir "none"
#     baja el esfuerzo sin destapar nada.
comprobar(
    "un esfuerzo 'none' no le quita el formato oculto a Qwen",
    aplicar(QWEN, "none").get("reasoning_format") == "hidden",
)
comprobar(
    "un esfuerzo 'none' llega a Qwen tal cual",
    aplicar(QWEN, "none").get("reasoning_effort") == "none",
)
comprobar(
    "un esfuerzo 'none' no le quita el formato oculto a gpt-oss",
    aplicar(OSS, "none").get("reasoning_format") == "hidden",
)
comprobar(
    "un esfuerzo 'none' se traduce al mínimo de gpt-oss, no se manda crudo",
    aplicar(OSS, "none").get("reasoning_effort") == "low",
)


# ---------------------------------------------------------------------------
# COMPORTAMIENTO 6 — Campo vacío != valor inválido
#
# Son casos OPUESTOS y durante un tiempo se trataron igual: el vacío caía en la
# red de seguridad contra el 400, que devuelve el máximo de la familia, y en Qwen
# el máximo se llama "default". O sea que vaciar el campo para QUITARLE
# pensamiento al modelo se lo subía al tope. Se vio en la instalación real, en la
# línea «Esfuerzo de razonamiento None no válido ...; uso el de por defecto».
# El corte lo hace el `if esfuerzo`, que cortocircuita ANTES de la red.
# ---------------------------------------------------------------------------

for vacio in (None, ""):
    comprobar(
        f"esfuerzo {vacio!r} no se convierte en el máximo de Qwen",
        "reasoning_effort" not in aplicar(QWEN, vacio),
    )
    # "Sin esfuerzo" no es "sin razonamiento": se le deja al modelo su propio
    # criterio, pero oculto. Perder esta línea devuelve el pensamiento al TTS.
    comprobar(
        f"esfuerzo {vacio!r} conserva igual el formato oculto en Qwen",
        aplicar(QWEN, vacio).get("reasoning_format") == "hidden",
    )
    comprobar(
        f"esfuerzo {vacio!r} tampoco inventa un esfuerzo para gpt-oss",
        "reasoning_effort" not in aplicar(OSS, vacio),
    )
    comprobar(
        f"esfuerzo {vacio!r} conserva el paquete completo de gpt-oss",
        aplicar(OSS, vacio) == {"reasoning_format": "hidden",
                                "include_reasoning": False},
    )

# 15. El caso opuesto, que tiene que seguir coexistiendo: un valor de VERDAD
#     desconocido sí usa la red de seguridad. Vacío = no configurado; basura =
#     red. Si se arregla uno rompiendo el otro no se arregló nada.
comprobar(
    "un valor desconocido sigue traduciéndose a algo válido",
    aplicar(QWEN, "altísimo").get("reasoning_effort") in VALIDOS_QWEN,
)


# ---------------------------------------------------------------------------
# COMPORTAMIENTO 8 — Higiene del diccionario entre candidatos
#
# El diccionario de la petición se arma UNA sola vez por turno (si no, el
# max_tokens que bajó un 413 se perdería en la vuelta siguiente) y se reusa en
# cada salto de la cadena. Así que lo que dejó el candidato anterior tiene que
# desaparecer, y tiene que desaparecer ANTES de cualquier salida temprana: la
# salida temprana es justo el caso del salto a un modelo que no razona.
# ---------------------------------------------------------------------------

SUCIO = {"reasoning_format": "hidden", "reasoning_effort": "default",
         "include_reasoning": False}

# 16. El salto Qwen -> llama, que es el que de verdad pasa cuando el principal
#     entra en enfriamiento: sin los pop, el reasoning_format del Qwen viaja
#     pegado a la petición del llama y vuelve un 400.
comprobar(
    "limpia los restos del candidato anterior al pasar a uno que no razona",
    aplicar(LLAMA, "default", previo=SUCIO) == {},
)

# 17. Y el salto gpt-oss -> Qwen: Qwen nunca escribe include_reasoning, así que
#     sin el pop el residuo de la familia anterior sobrevive silenciosamente.
comprobar(
    "limpia include_reasoning al pasar de gpt-oss a Qwen",
    "include_reasoning" not in aplicar(QWEN, "default", previo=SUCIO),
)

# 18. La higiene también hace falta cuando NO hay salida temprana: el mismo
#     modelo, con el esfuerzo que pasó a "no configurado", tiene que perder el
#     reasoning_effort de la vuelta anterior en vez de arrastrarlo.
comprobar(
    "un esfuerzo que pasa a vacío borra el reasoning_effort anterior",
    aplicar(QWEN, "", previo=SUCIO) == {"reasoning_format": "hidden"},
)

# 19. Es el camino del reintento sin pensamiento del comportamiento 1: el mismo
#     candidato, el mismo dict, ahora con "none" forzado encima de lo que había.
comprobar(
    "el reintento con 'none' pisa el esfuerzo que ya estaba escrito",
    aplicar(QWEN, "none", previo=SUCIO).get("reasoning_effort") == "none",
)

# 20. La función comparte diccionario con el resto de la petición: toca sus tres
#     claves y ninguna más. Si le pisara el max_tokens que bajó `_pedir_encogiendo`
#     por un 413, el turno entero se cae con otro 413.
kwargs = {"model": LLAMA, "max_tokens": 1200, "messages": [],
          "reasoning_effort": "default"}
devuelto = M._aplicar_razonamiento(kwargs, LLAMA, "default")
comprobar(
    "no pisa el resto de la petición",
    kwargs == {"model": LLAMA, "max_tokens": 1200, "messages": []},
)

# 21. Muta in-place y devuelve None. Si alguien la convirtiera en una función que
#     devuelve un dict nuevo, el llamador seguiría compilando y perdería en
#     silencio todo lo que escribió.
comprobar("muta in-place y no devuelve nada", devuelto is None)


# ---------------------------------------------------------------------------
# `_esfuerzo_del_candidato` — el principal piensa, los respaldos no
#
# Esta política era el campo de configuración reasoning_effort_chain, que ya no
# existe. Medido en la instalación real: el titular contesta pensando en 120-180
# tokens y el suplente quema los 1200 de max_tokens razonando y vuelve VACÍO. Un
# suplente existe para contestar cuando el titular no puede; gastarle el
# presupuesto en pensar lo vuelve inútil, y la diferencia no es entre una
# respuesta mejor y una peor sino entre una respuesta y ninguna.
# ---------------------------------------------------------------------------

# 22. El principal usa el esfuerzo que se decidió para el turno, tal cual.
comprobar(
    "el principal usa el esfuerzo decidido",
    M._esfuerzo_del_candidato("default", True) == "default",
)
comprobar(
    "el principal conserva también un esfuerzo alto",
    M._esfuerzo_del_candidato("high", True) == "high",
)

# 23. Los respaldos van con "none", venga el esfuerzo que venga.
comprobar(
    "un respaldo va sin pensamiento",
    M._esfuerzo_del_candidato("default", False) == "none",
)
comprobar(
    "un respaldo no hereda el esfuerzo del principal ni siendo el máximo",
    M._esfuerzo_del_candidato("high", False) == "none",
)

# 24. "No configurado" sobrevive en el principal: no se convierte en "none" ni en
#     el máximo. Es la interacción con el comportamiento 6 — si el campo está
#     vacío y el enrutador dice que sí razone, el esfuerzo queda sin configurar,
#     que significa formato oculto y ningún reasoning_effort.
comprobar(
    "el principal con el campo vacío sigue sin esfuerzo configurado",
    M._esfuerzo_del_candidato(None, True) is None,
)
comprobar(
    "el principal con el campo en cadena vacía tampoco inventa un esfuerzo",
    M._esfuerzo_del_candidato("", True) == "",
)

# 25. Pero un respaldo con el campo vacío sí baja a "none" explícito: acá no
#     alcanza con "que decida el modelo", porque lo que decide el modelo es
#     pensar, y pensando se queda sin presupuesto para contestar.
comprobar(
    "un respaldo con el campo vacío igual va con 'none'",
    M._esfuerzo_del_candidato(None, False) == "none",
)

# 26. La composición completa: el esfuerzo del respaldo pasa igual por la
#     traducción de familia, así que en gpt-oss el "none" termina siendo "low".
#     Sin esto el respaldo recibiría un valor que su familia rechaza con 400.
comprobar(
    "el respaldo también se traduce a la familia que toque",
    aplicar(OSS, M._esfuerzo_del_candidato("default", False))
    .get("reasoning_effort") == "low",
)
comprobar(
    "y en Qwen el respaldo llega con 'none' pero sigue oculto",
    aplicar(QWEN, M._esfuerzo_del_candidato("default", False))
    == {"reasoning_format": "hidden", "reasoning_effort": "none"},
)

# 27. Quien decide es la IDENTIDAD del candidato, NUNCA su posición en la lista.
#     El principal reaparece más abajo en la rotación cuando está en
#     enfriamiento, y ahí sigue siendo el titular aunque entre último. Esto imita
#     el bucle de `_responder_con_cadena` con la cadena ya rotada.
PRINCIPAL = QWEN
CADENA_ROTADA = [OSS, LLAMA, PRINCIPAL]
aplicados = [
    aplicar(c, M._esfuerzo_del_candidato("default", c == PRINCIPAL))
    for c in CADENA_ROTADA
]
comprobar(
    "el principal en última posición conserva su esfuerzo",
    aplicados[2].get("reasoning_effort") == "default",
)
comprobar(
    "el respaldo en primera posición no lo hereda",
    aplicados[0].get("reasoning_effort") == "low",
)
comprobar(
    "el respaldo que no razona sigue sin recibir nada aunque vaya en el medio",
    aplicados[1] == {},
)

resumen("razonamiento: traducción, familias, campo vacío e higiene del dict")
