#!/usr/bin/env python3
"""Pruebas de la tabla de fallos de los dos enrutadores (`_veredicto_casa`, `_veredicto_razonamiento`).

Lo que se está protegiendo acá son DOS fallos reales, y son opuestos entre sí.
Por eso las dos tablas también lo son, y por eso esta tanda existe: si algún día
alguien las "unifica" para que se parezcan, el código va a seguir andando y las
dos averías vuelven en silencio.

El enrutador de casa está para no pagar los 4.150 tokens del bloque de
herramientas —el 52 % del techo de 8.000 TPM del plan gratuito— en preguntas que
no tocan la casa. Pero cuando ese enrutador se rompe, resolver hacia "no hace
falta la casa" deja al usuario sin poder prender la luz, que es la función
primaria del aparato. ANTE FALLO CONSERVA LAS HERRAMIENTAS: solo puede ahorrar,
nunca romper. Para romper el control de la casa tiene que estar seguro Y
equivocado.

El enrutador de razonamiento está para no razonar de gusto, y ahí el error caro
es el otro. Medido el 31 ago 2026 contra la instalación real: el principal
contesta pensando en 120-180 tokens, y el respaldo qwen3.6-27b quemaba los 1200
de `max_tokens` razonando y volvía con la respuesta VACÍA. No hay excepción ni
429: Assist da la vuelta entera, no emite `synthesize` y el usuario escucha
silencio. ANTE FALLO NO RAZONA.

La tercera cosa que se fija acá es la nota 2.6: APAGADO NO ES LO MISMO QUE
FALLO. En el de casa coinciden por casualidad (los dos conservan), pero en el de
razonamiento son opuestos: apagado SÍ piensa con el esfuerzo configurado —porque
apagado significa "comportamiento de siempre" y actualizar la integración no
puede cambiarle el comportamiento a una instalación que anda— y el fallo NO
piensa.
"""

import asyncio
from types import SimpleNamespace

from cargar import cargar
from runner import comprobar, resumen

M = cargar(
    [
        "VEREDICTO_CASA_APAGADO",
        "VEREDICTO_RAZONAMIENTO_APAGADO",
        "_texto_del_enrutador",
        "_extraer_json",
        "_veredicto_casa",
        "_veredicto_razonamiento",
        "_linea_decision",
    ],
    "enrutadores",
)

UMBRAL = 0.7


def respuesta(contenido, motivo="stop"):
    """Imita la forma de una ChatCompletion del SDK de Groq."""
    # Por getattr y con SimpleNamespace, nunca por isinstance contra clases del
    # SDK: es lo que permite probar esto sin instalar groq, y también lo que
    # evita que un proveedor con la respuesta apenas distinta tumbe el turno.
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=motivo,
                message=SimpleNamespace(content=contenido),
            )
        ]
    )


# El JSON que un enrutador roto podría llegar a escupir con las claves puestas
# hacia el lado PELIGROSO de cada tabla: para casa, "no hace falta la casa";
# para razonamiento, "sí, pensá". Las dos claves viajan en el mismo texto a
# propósito, así el mismo caso de fallo empuja a los dos veredictos hacia el
# lado equivocado y la prueba mide si aguantan.
PELIGROSO = '{"needs_home": false, "reasoning_required": true, "confidence": 0.99}'


# 1. LA TABLA DE FALLOS ENTERA. Cada fila es una forma distinta en que un
#    enrutador puede fallar, y todas tienen que resolverse hacia el mismo lado
#    dentro de cada tabla. El listado va explícito, fila por fila, porque cada
#    una se rompió o se puede romper por su cuenta: una excepción no se parece
#    en nada a un JSON con una clave de más.
FALLOS = [
    # Lo que devuelve asyncio.gather(return_exceptions=True) cuando el SDK
    # levanta: la excepción viene EN la lista de resultados, no se propaga.
    ("una excepción del SDK", RuntimeError("500 del proveedor")),
    # asyncio.timeout levanta TimeoutError. Por voz no hay spinner: un
    # enrutador colgado cuesta el turno entero, no una decisión peor.
    ("un timeout", TimeoutError()),
    # CancelledError es BaseException y NO Exception. Un `except Exception`
    # la dejaría escapar y se llevaría puesto el turno; por eso el filtro es
    # por BaseException.
    ("una cancelación", asyncio.CancelledError()),
    ("nada en absoluto", None),
    ("una respuesta sin choices", SimpleNamespace(choices=[])),
    ("choices en None", SimpleNamespace(choices=None)),
    ("un objeto sin el atributo choices", SimpleNamespace()),
    (
        "message en None",
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=None)]
        ),
    ),
    ("una respuesta vacía", respuesta("")),
    ("contenido en None", respuesta(None)),
    ("solo espacios en blanco", respuesta("   \n\t ")),
    # Truncado ES fallo acá, aunque en la respuesta principal no siempre lo
    # sea: el enrutador tiene 150 tokens de techo para un JSON de tres claves,
    # así que cortado por longitud significa JSON a medio cerrar. El contenido
    # de esta fila está entero para que solo el finish_reason decida.
    ("un corte por longitud", respuesta(PELIGROSO, motivo="length")),
    ("un JSON a medio cerrar", respuesta('{"needs_home": false, "confi')),
    ("prosa sin ningún JSON", respuesta("Me parece que no hace falta la casa.")),
    ("un JSON que no es un objeto", respuesta("[false, 0.99]")),
    ("un JSON sin las claves que importan", respuesta('{"confidence": 0.99}')),
    # El modelo escribe "false" con comillas: en Python eso es una cadena no
    # vacía, o sea TRUTHY. Un `if datos.get("needs_home")` lo leería como True
    # y un `if not datos.get(...)` como False. Se exige bool de verdad.
    (
        "el veredicto como texto en vez de booleano",
        respuesta('{"needs_home": "false", "reasoning_required": "true", '
                  '"confidence": 0.99}'),
    ),
    (
        "el veredicto como número en vez de booleano",
        respuesta('{"needs_home": 0, "reasoning_required": 1, '
                  '"confidence": 0.99}'),
    ),
    (
        "la confianza ausente",
        respuesta('{"needs_home": false, "reasoning_required": true}'),
    ),
    (
        "la confianza como texto",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": "0.99"}'),
    ),
    # bool es subclase de int en Python: sin descartarlo a mano, `true` pasaría
    # como 1.0 y le daría al enrutador una certeza que nunca declaró, justo la
    # certeza que hace falta para apagar las herramientas.
    (
        "la confianza como booleano",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": true}'),
    ),
    (
        "la confianza por encima de 1",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": 1.4}'),
    ),
    (
        "la confianza negativa",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": -0.2}'),
    ),
    # NaN es el único valor que derrota a las DOS tablas al mismo tiempo, y por
    # eso está acá: json.loads acepta el literal NaN sin chistar, y toda
    # comparación con NaN da False, así que un rango escrito como
    # `< 0.0 or > 1.0` lo deja pasar y el `< umbral` posterior tampoco lo
    # frena. Resultado: las herramientas se apagaban y el modelo se ponía a
    # pensar, las dos cosas con una certeza que el enrutador nunca declaró.
    (
        "la confianza en NaN",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": NaN}'),
    ),
    (
        "la confianza en Infinity",
        respuesta('{"needs_home": false, "reasoning_required": true, '
                  '"confidence": Infinity}'),
    ),
    # El contenido como lista de bloques, que es lo que devuelven otros
    # proveedores: si se le aplica .strip() directo salta un AttributeError que
    # no tiene ningún except arriba, sube por _decidir_enrutadores y se lleva
    # puesto el turno entero. Un enrutador roto no puede costar más que la
    # decisión que iba a tomar.
    (
        "el contenido como lista de bloques en vez de texto",
        respuesta([{"type": "text", "text": PELIGROSO}]),
    ),
    ("el contenido como número", respuesta(7)),
]

for descripcion, resultado in FALLOS:
    casa = M._veredicto_casa(resultado, UMBRAL)
    comprobar(
        f"casa ante {descripcion}: CONSERVA las herramientas",
        casa[0] is True and casa[1] == 0.0,
    )
    razon = M._veredicto_razonamiento(resultado, UMBRAL)
    comprobar(
        f"razonamiento ante {descripcion}: NO piensa",
        razon[0] is False and razon[1] == 0.0,
    )

# 2. Las dos tablas son OPUESTAS ante el mismo dato, y eso vale para todas las
#    filas a la vez. Es la comprobación que se rompería si alguien decidiera
#    que "los dos enrutadores se comportan igual ante un fallo".
comprobar(
    "ninguna fila de la tabla resuelve para el mismo lado en los dos enrutadores",
    all(
        M._veredicto_casa(r, UMBRAL)[0] is not M._veredicto_razonamiento(r, UMBRAL)[0]
        for _, r in FALLOS
    ),
)

# 3. Un fallo tiene que dejar rastro legible: el motivo es lo único que después
#    permite distinguir en el log un 429 del enrutador de una respuesta suya
#    deforme. Un motivo vacío convierte el log de auditoría en ruido.
comprobar(
    "todo fallo de casa trae un motivo escrito",
    all(isinstance(M._veredicto_casa(r, UMBRAL)[2], str)
        and M._veredicto_casa(r, UMBRAL)[2].strip()
        for _, r in FALLOS),
)
comprobar(
    "todo fallo de razonamiento trae un motivo escrito",
    all(isinstance(M._veredicto_razonamiento(r, UMBRAL)[3], str)
        and M._veredicto_razonamiento(r, UMBRAL)[3].strip()
        for _, r in FALLOS),
)
comprobar(
    "todo fallo de razonamiento normaliza la categoría a general",
    all(M._veredicto_razonamiento(r, UMBRAL)[2] == "general" for _, r in FALLOS),
)

# 4. La forma de la tupla es parte del contrato: el llamador desempaqueta
#    casa[0..2] y razon[0..3] para armar la línea del log. Una tupla de otro
#    largo revienta ahí, en el camino de la respuesta al usuario.
comprobar(
    "casa devuelve siempre tres campos",
    all(len(M._veredicto_casa(r, UMBRAL)) == 3 for _, r in FALLOS),
)
comprobar(
    "razonamiento devuelve siempre cuatro campos",
    all(len(M._veredicto_razonamiento(r, UMBRAL)) == 4 for _, r in FALLOS),
)


# 5. LA CONFIANZA BAJO EL UMBRAL NO ES UN FALLO, pero se resuelve para el mismo
#    lado que el fallo en los dos casos. La diferencia con el fallo es que acá
#    la confianza declarada SÍ se conserva: en el log se ve un 0.40, no un 0.00,
#    y así se distingue "el modelo dudó" de "el modelo no contestó".
dudoso_casa = respuesta('{"needs_home": false, "confidence": 0.4, '
                        '"reason": "parece charla"}')
casa = M._veredicto_casa(dudoso_casa, UMBRAL)
comprobar(
    "casa con poca confianza conserva las herramientas igual",
    casa[0] is True,
)
comprobar(
    "casa con poca confianza conserva la confianza declarada, no la del fallo",
    casa[1] == 0.4,
)
comprobar(
    "casa con poca confianza lo dice en el motivo",
    "poca confianza" in casa[2],
)

dudoso_razon = respuesta('{"reasoning_required": true, "confidence": 0.4, '
                         '"category": "math", "brief_reason": "capaz hay cuentas"}')
razon = M._veredicto_razonamiento(dudoso_razon, UMBRAL)
comprobar(
    "razonamiento con poca confianza no piensa",
    razon[0] is False,
)
comprobar(
    "razonamiento con poca confianza conserva la confianza declarada",
    razon[1] == 0.4,
)
comprobar(
    "razonamiento con poca confianza lo dice en el motivo",
    "poca confianza" in razon[3],
)

# 6. El umbral es un ">=", no un ">": justo en el umbral alcanza. Si fuera
#    estricto, el 0.7 que la UI ofrece por defecto no se cumpliría nunca con un
#    modelo que devuelve exactamente 0.7 y el enrutador no ahorraría jamás.
comprobar(
    "casa: la confianza justo en el umbral alcanza para quitar las herramientas",
    M._veredicto_casa(
        respuesta('{"needs_home": false, "confidence": 0.7}'), 0.7
    )[0] is False,
)
comprobar(
    "razonamiento: la confianza justo en el umbral alcanza para pensar",
    M._veredicto_razonamiento(
        respuesta('{"reasoning_required": true, "confidence": 0.7}'), 0.7
    )[0] is True,
)

# 7. El umbral solo se aplica del lado que puede hacer daño. Del lado seguro no
#    hay nada que proteger: exigir confianza para CONSERVAR las herramientas, o
#    para NO pensar, sería pedirle permiso al modelo para no hacer nada.
comprobar(
    "casa: needs_home true con poca confianza sigue siendo true",
    M._veredicto_casa(
        respuesta('{"needs_home": true, "confidence": 0.1}'), UMBRAL
    )[0] is True,
)
comprobar(
    "razonamiento: reasoning_required false con poca confianza sigue siendo false",
    M._veredicto_razonamiento(
        respuesta('{"reasoning_required": false, "confidence": 0.1}'), UMBRAL
    )[0] is False,
)


# 8. Y AHORA EL LADO QUE AHORRA, porque una tabla de fallos que resuelve siempre
#    hacia el lado seguro se puede satisfacer con `return (True, 0.0, "")` y
#    dejar al enrutador sin servir para nada. Estas dos filas son las únicas que
#    justifican que el enrutador exista.
ahorro = M._veredicto_casa(
    respuesta('{"needs_home": false, "confidence": 0.95, '
              '"reason": "pregunta de historia"}'),
    UMBRAL,
)
comprobar(
    "casa seguro y en false SÍ quita las herramientas: es el ahorro de 4150 tokens",
    ahorro == (False, 0.95, "pregunta de historia"),
)
piensa = M._veredicto_razonamiento(
    respuesta('{"reasoning_required": true, "confidence": 0.95, '
              '"category": "math", "brief_reason": "sistema de ecuaciones"}'),
    UMBRAL,
)
comprobar(
    "razonamiento seguro y en true SÍ manda a pensar",
    piensa == (True, 0.95, "math", "sistema de ecuaciones"),
)

# 9. La categoría es informativa: sirve para auditar el log, no para decidir.
#    Una inventada por el modelo se normaliza a "general" y NO puede tirar abajo
#    un veredicto que ya venía respaldado por la clave y la confianza.
inventada = M._veredicto_razonamiento(
    respuesta('{"reasoning_required": true, "confidence": 0.95, '
              '"category": "astrología", "brief_reason": "vaya a saber"}'),
    UMBRAL,
)
comprobar(
    "una categoría inventada no invalida el veredicto",
    inventada[0] is True,
)
comprobar(
    "una categoría inventada se normaliza a general",
    inventada[2] == "general",
)


# 10. EL JSON ENVUELTO EN MARKDOWN NO ES UN FALLO. A un clasificador al que se le
#     pide "solo JSON" igual le sale una cerca ```json o una frase de cortesía
#     adelante. Mandar eso a la rama de fallo sería tirar una decisión buena por
#     un detalle de formato: en el enrutador de casa se pierde el ahorro, y en el
#     de razonamiento se pierde el pensamiento en un problema de matemática.
envuelto = respuesta(
    'Claro, acá va:\n```json\n{"needs_home": false, "confidence": 0.95, '
    '"reason": "es una definición"}\n```\nEspero que sirva.'
)
comprobar(
    "casa parsea el JSON aunque venga con cerca de código y prosa alrededor",
    M._veredicto_casa(envuelto, UMBRAL) == (False, 0.95, "es una definición"),
)
comprobar(
    "razonamiento también atraviesa la cerca de código",
    M._veredicto_razonamiento(
        respuesta('```json\n{"reasoning_required": true, "confidence": 0.9, '
                  '"category": "coding", "brief_reason": "hay que depurar"}\n```'),
        UMBRAL,
    ) == (True, 0.9, "coding", "hay que depurar"),
)
# Un objeto anidado no puede cortar la búsqueda en la primera llave que cierra.
comprobar(
    "un objeto con llaves anidadas se extrae entero",
    M._extraer_json('{"a": {"b": 1}, "needs_home": false}')
    == {"a": {"b": 1}, "needs_home": False},
)
# Y una llave dentro de una cadena no cuenta como estructura: si contara, el
# primer objeto quedaría mal balanceado y una respuesta buena se perdería.
comprobar(
    "una llave dentro de una cadena no desbalancea la extracción",
    M._extraer_json('{"reason": "sube el {volumen}", "needs_home": true}')
    == {"reason": "sube el {volumen}", "needs_home": True},
)
comprobar(
    "sin ningún objeto devuelve None en vez de reventar",
    M._extraer_json("no hay json acá") is None,
)
comprobar(
    "un texto que no es texto devuelve None en vez de reventar",
    M._extraer_json(None) is None,
)


# 11. El motivo va al log, y el log del usuario ya se inunda solo. Un modelo que
#     ignora el "menos de 8 palabras" no puede escribirle un párrafo a cada
#     turno.
largo = M._veredicto_casa(
    respuesta('{"needs_home": true, "confidence": 0.9, "reason": "'
              + "palabra " * 60 + '"}'),
    UMBRAL,
)
comprobar(
    "el motivo desmedido se recorta",
    len(largo[2]) <= 80,
)
comprobar(
    "un motivo vacío no deja el campo en blanco",
    M._veredicto_casa(
        respuesta('{"needs_home": true, "confidence": 0.9, "reason": "   "}'),
        UMBRAL,
    )[2] == "sin motivo",
)
comprobar(
    "un motivo que no es texto no revienta el formateo",
    M._veredicto_razonamiento(
        respuesta('{"reasoning_required": false, "confidence": 0.9, '
                  '"brief_reason": 7}'),
        UMBRAL,
    )[3] == "sin motivo",
)


# 12. NOTA 2.6: APAGADO NO ES FALLO. Las dos constantes están en el nivel
#     superior justamente para poder clavar acá la asimetría.
comprobar(
    "casa apagado conserva las herramientas",
    M.VEREDICTO_CASA_APAGADO[0] is True,
)
comprobar(
    "casa apagado tiene la misma forma que un veredicto de casa",
    len(M.VEREDICTO_CASA_APAGADO) == 3,
)
comprobar(
    "razonamiento apagado tiene la misma forma que un veredicto de razonamiento",
    len(M.VEREDICTO_RAZONAMIENTO_APAGADO) == 4,
)
# ESTA es la comprobación que importa de toda la nota: apagado y fallo del
# enrutador de razonamiento son OPUESTOS. Apagado significa "comportamiento de
# siempre", y el de siempre es que el modelo principal piensa con el esfuerzo
# que el usuario configuró. Actualizar la integración no puede quitarle el
# pensamiento a una instalación que anda.
comprobar(
    "razonamiento APAGADO sí piensa, al revés que razonamiento FALLADO",
    M.VEREDICTO_RAZONAMIENTO_APAGADO[0] is True
    and M._veredicto_razonamiento(None, UMBRAL)[0] is False,
)
# En el de casa, en cambio, apagado y fallo coinciden en el veredicto. Coinciden
# por casualidad, no porque sea la misma regla: conservar es a la vez "lo de
# siempre" y "lo seguro".
comprobar(
    "casa APAGADO y casa FALLADO coinciden en el veredicto",
    M.VEREDICTO_CASA_APAGADO[0] is M._veredicto_casa(None, UMBRAL)[0],
)
# Pero se distinguen en el log por la confianza: apagado declara 1.00 (no hubo
# duda, no se preguntó) y el fallo 0.00. Sin esa diferencia, auditar el log no
# permitiría separar "el enrutador está apagado" de "el enrutador se cae
# siempre", que necesitan arreglos distintos.
comprobar(
    "apagado y fallo se distinguen en el log por la confianza",
    M.VEREDICTO_CASA_APAGADO[1] == 1.0
    and M._veredicto_casa(None, UMBRAL)[1] == 0.0
    and M.VEREDICTO_RAZONAMIENTO_APAGADO[1] == 1.0,
)
comprobar(
    "los dos apagados se nombran como tales en su motivo",
    "apagado" in M.VEREDICTO_CASA_APAGADO[2]
    and "apagado" in M.VEREDICTO_RAZONAMIENTO_APAGADO[3],
)
comprobar(
    "la categoría del apagado de razonamiento es una de las válidas",
    M.VEREDICTO_RAZONAMIENTO_APAGADO[2]
    in ("math", "coding", "logic", "general", "creative"),
)


# 13. NOTA 2.22: la línea de auditoría lleva LOS CINCO DATOS. Con esto se
#     verifica si el enrutador se está equivocando (~/simular-assist/cupo.sh) en
#     vez de confiar en él a ciegas; si falta uno, la línea no alcanza para
#     decidir nada y hay que reproducir el turno a mano.
linea = M._linea_decision("casa", "openai/gpt-oss-20b", False, 0.93,
                          "es una pregunta de historia", 312)
comprobar(
    "la línea trae los cinco datos: modelo, veredicto, confianza, motivo y ms",
    "openai/gpt-oss-20b" in linea
    and "veredicto=no" in linea
    and "confianza=0.93" in linea
    and '"es una pregunta de historia"' in linea
    and "ms=312" in linea,
)
comprobar(
    "la línea dice de qué enrutador es",
    "casa" in linea,
)
# Una sola línea: el log se lee con grep, y un motivo con salto de línea partiría
# la entrada en dos y rompería el conteo.
comprobar(
    "la línea es una sola línea",
    "\n" not in linea,
)
comprobar(
    "el veredicto afirmativo se escribe distinto del negativo",
    "veredicto=sí" in M._linea_decision("casa", "m", True, 1.0, "x", 1),
)
# La confianza siempre con dos decimales: es lo que hace que las líneas se
# puedan ordenar y contar sin parsear números de largo variable.
comprobar(
    "la confianza va con dos decimales aunque llegue entera",
    "confianza=1.00" in M._linea_decision("casa", "m", True, 1, "x", 1),
)
comprobar(
    "los milisegundos van enteros aunque llegue un float",
    "ms=312" in M._linea_decision("casa", "m", True, 0.5, "x", 312.7),
)
# El motivo entre comillas para que se pueda recortar del resto de la línea aun
# teniendo espacios adentro.
comprobar(
    "el motivo va entrecomillado",
    '"x y z"' in M._linea_decision("razonamiento", "m", True, 0.5, "x y z", 1),
)
# La línea de un fallo tiene que verse como fallo de un vistazo: confianza en
# cero y el motivo diciendo qué pasó.
fallado = M._veredicto_casa(RuntimeError("boom"), UMBRAL)
comprobar(
    "un fallo se reconoce en el log por la confianza en cero",
    "confianza=0.00"
    in M._linea_decision("casa", "m", fallado[0], fallado[1], fallado[2], 5),
)


# 14. `_texto_del_enrutador` es el embudo por el que pasan todos los modos de
#     fallo antes de llegar a los veredictos, y NUNCA re-levanta: el fallo de un
#     enrutador no puede tumbar al otro ni al turno entero. Quien traduce ese
#     None en una decisión es el veredicto, que es el único que sabe hacia qué
#     lado se falla.
comprobar(
    "una excepción se traga y se convierte en None",
    M._texto_del_enrutador(RuntimeError("boom")) is None,
)
comprobar(
    "una cancelación, que no es Exception, también",
    M._texto_del_enrutador(asyncio.CancelledError()) is None,
)
comprobar(
    "una respuesta buena devuelve el texto sin los espacios de los bordes",
    M._texto_del_enrutador(respuesta('  {"needs_home": true}  '))
    == '{"needs_home": true}',
)

resumen("tabla de fallos de los enrutadores")
