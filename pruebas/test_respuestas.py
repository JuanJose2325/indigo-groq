#!/usr/bin/env python3
"""Pruebas de cómo se lee una respuesta de Groq y qué se decide con ella.

EL FALLO REAL QUE PROTEGE ESTE ARCHIVO ES EL SILENCIO. Con el pensamiento
activo, un modelo de razonamiento gasta los `max_tokens` enteros pensando y
devuelve `finish_reason="length"` con el contenido VACÍO. No hay excepción, no
hay 429, el pipeline de Assist da la vuelta completa como si todo hubiera salido
bien, no emite `synthesize` y el usuario se queda escuchando nada. En los logs
del micrófono tampoco aparece nada. Medido el 31 ago 2026 contra la instalación
real: el principal contestaba pensando en 120-180 tokens y el de respaldo
quemaba los 1200 de `max_tokens` razonando y volvía vacío 2 de 2 veces.

Acá se prueban las dos mitades de ese fallo. La detectora (`_vacia_por_truncado`)
ya tenía pruebas y se conservan enteras. La REACTIVA —repetir una vez el mismo
modelo con el pensamiento forzado a "none" y recién después saltar— vivía dentro
de un método de la entidad y no tenía ni una comprobación; ahora son dos
funciones puras (`_decidir_tras_error` y `_decidir_tras_respuesta`) y se prueban
tabla por tabla.

Además:

- `_encoger_tope`, la aritmética del 413. Un 413 no es falta de cupo: es que la
  petición entera no entra. Rotar de modelo ahí no arregla nada, porque al
  siguiente candidato le llega exactamente lo mismo; en el log eso se veía como
  pares 413 -> 413 instantáneos sin que nadie contestara nunca. Un 429 SÍ es
  falta de cupo (8000 TPM medidos) y ahí lo correcto es rotar, no encoger.
- `_detalle_error`, con cuerpos deformes en cada nivel. Sin el mensaje del
  proveedor en el log, 413 y 429 se leen igual —"no pudo"— siendo problemas
  OPUESTOS, y se pierde el "Limit 8000, Requested 8441", que son los dos números
  con los que se calibra `max_tokens`.

Todo esto corre en el camino de la respuesta al usuario: una excepción acá no se
lee en una traza, se ESCUCHA como un error hablado. De ahí que la mitad de las
comprobaciones sean formas rotas que nunca deben reventar.
"""

from types import SimpleNamespace

from cargar import cargar
from runner import comprobar, resumen

# `_decidir_tras_respuesta` llama a `_vacia_por_truncado`, y el cargador por AST
# compila todos los nodos elegidos en UN mismo espacio de nombres: si la llamada
# no viniera en la misma tanda, el NameError saltaría recién al ejecutar.
M = cargar(
    [
        "_vacia_por_truncado",
        "_detalle_error",
        "_encoger_tope",
        "_decidir_tras_error",
        "_decidir_tras_respuesta",
        "_argumentos_de_herramienta",
    ],
    "respuestas",
)

# Los cuatro veredictos son literales de texto exactos y el llamador los compara
# con `==`: una errata acá no rompe nada visible, simplemente hace que la rama
# nunca se tome. Por eso se fijan como constantes de la prueba.
ACEPTAR = "aceptar"
REINTENTAR = "reintentar_sin_pensamiento"
SALTAR = "saltar"
PROPAGAR = "propagar"

# El piso real de const.py. Se copia a mano a propósito: si alguien lo cambia,
# conviene que la prueba lo note en vez de seguirlo sin chistar.
PISO = 400


def respuesta(motivo, contenido=None, tool_calls=None):
    """Imita la forma de una ChatCompletion del SDK de Groq."""
    mensaje = SimpleNamespace(content=contenido, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=motivo, message=mensaje)]
    )


VACIA = respuesta("length", "")
BUENA = respuesta("stop", "Prendí la luz del living.")


# =============================================================================
# A. `_vacia_por_truncado`: la mitad DETECTORA del fallo silencioso.
# =============================================================================

# 1. El caso real, tal cual llegó de la instalación: cortado por longitud y sin
#    una sola letra. La comparación es contra `is True` y no contra truthiness
#    porque el llamador lo mete en una máquina de estados: tiene que ser un bool
#    de verdad, no algo que "parezca" verdadero.
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

# 2. Truncado PERO con texto: el usuario escucha una respuesta cortada a media
#    frase, que es molesto pero no es silencio. Reintentar le costaría otra
#    ventana de cupo del minuto para ganar poco, así que esto no se toca.
comprobar(
    "truncado con texto no es el fallo",
    M._vacia_por_truncado(respuesta("length", "Sí, podés cambiar el")) is False,
)

# 3. Un final normal jamás es este fallo, ni siquiera viniendo vacío: eso es otro
#    problema (el modelo no quiso contestar) y se trata en otro lado. La firma
#    exige las DOS condiciones, no una.
comprobar(
    "fin normal no es el fallo",
    M._vacia_por_truncado(respuesta("stop", "Hola.")) is False,
)
comprobar(
    "fin normal vacío tampoco entra acá",
    M._vacia_por_truncado(respuesta("stop", "")) is False,
)
comprobar(
    "sin finish_reason no es el fallo",
    M._vacia_por_truncado(respuesta(None, "")) is False,
)

# 4. GUARDA INTOCABLE. Truncado tras pedir una herramienta: la petición está
#    completa y sirve aunque no venga texto acompañándola. Si esto devolviera
#    True se repetirían acciones sobre la casa del usuario —apagar dos veces,
#    abrir dos veces—, que es un daño real y no una molestia. El chequeo de
#    tool_calls va ANTES del de content justamente por eso.
comprobar(
    "truncado con tool_calls NO es el fallo",
    M._vacia_por_truncado(
        respuesta("length", "", tool_calls=[SimpleNamespace(id="x")])
    ) is False,
)
comprobar(
    "la guarda protege también cuando el content viene en None",
    M._vacia_por_truncado(
        respuesta("length", None, tool_calls=[SimpleNamespace(id="x")])
    ) is False,
)

# 5. La otra cara de la guarda: una lista de tool_calls VACÍA no es una petición
#    de herramienta, es la ausencia de una. Si acá se mirara `is not None` en vez
#    de truthiness, un vacío real quedaría tapado y volvería el silencio.
comprobar(
    "tool_calls vacía no tapa un vacío real",
    M._vacia_por_truncado(respuesta("length", "", tool_calls=[])) is True,
)

# 6. Respuestas deformes. Nunca deben reventar: esto corre en el camino de la
#    respuesta al usuario y una excepción acá se oye como un error hablado.
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
    "None entero no revienta",
    M._vacia_por_truncado(None) is False,
)

# 7. Sutil y facilísimo de romper al reescribir: `message=None` con
#    `finish_reason="length"` se clasifica como EL FALLO, no como caso raro. Si
#    se cortó por longitud y no hay mensaje, no hay nada que decirle al usuario;
#    lo que corresponde es reintentar, no aceptar el silencio.
comprobar(
    "message en None con length es el fallo, no un caso raro",
    M._vacia_por_truncado(
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                 message=None)])
    ) is True,
)


# =============================================================================
# B. `_encoger_tope`: la aritmética del 413, por fin probable sin async ni SDK.
# =============================================================================

# 1. El caso que motivó todo: rechazo por tamaño, se pide la mitad y se
#    reintenta el MISMO modelo. Rotar acá sería mandarle al siguiente candidato
#    exactamente la misma petición sobredimensionada.
comprobar(
    "413 baja el tope a la mitad",
    M._encoger_tope(413, 1200, PISO) == 600,
)
comprobar(
    "413 vuelve a bajar a la mitad en la vuelta siguiente",
    M._encoger_tope(413, 600, PISO) == PISO,
)

# 2. EL invariante rector: 413 y 429 son problemas OPUESTOS. El 429 es falta de
#    cupo (8000 TPM medidos) y se arregla rotando de modelo, no pidiendo menos;
#    encoger ahí gastaría un viaje de red para volver a chocar con el mismo
#    límite. None es la señal de "no encojas".
comprobar(
    "429 no encoge: eso es falta de cupo y se rota",
    M._encoger_tope(429, 1200, PISO) is None,
)
comprobar(
    "400 no encoge",
    M._encoger_tope(400, 1200, PISO) is None,
)
comprobar(
    "401 no encoge",
    M._encoger_tope(401, 1200, PISO) is None,
)
comprobar(
    "500 no encoge",
    M._encoger_tope(500, 1200, PISO) is None,
)

# 3. El piso. Por debajo de él la respuesta sale cortada a media frase, y por voz
#    eso se entiende PEOR que un "no pude": no hay pantalla donde leer el resto.
#    Ahí conviene dejar de encoger y que el error se propague.
comprobar(
    "en el piso justo ya no encoge",
    M._encoger_tope(413, PISO, PISO) is None,
)
comprobar(
    "por debajo del piso tampoco encoge",
    M._encoger_tope(413, 300, PISO) is None,
)
comprobar(
    "un tope apenas por encima del piso se recorta hasta el piso, no más abajo",
    M._encoger_tope(413, PISO + 1, PISO) == PISO,
)
comprobar(
    "un tope ausente (0) no encoge",
    M._encoger_tope(413, 0, PISO) is None,
)


def encoger_hasta_el_fondo(tope, piso):
    """Repite el bucle real de `_pedir_encogiendo`: encoge mientras haya 413.

    Devuelve la lista de topes visitados, o None si el bucle no convergió. Un
    `_encoger_tope` que se estancara en un valor haría girar ese `while True`
    para siempre contra la API, quemando el cupo del minuto entero.
    """
    visitados = []
    for _ in range(50):
        nuevo = M._encoger_tope(413, tope, piso)
        if nuevo is None:
            return visitados
        visitados.append(nuevo)
        tope = nuevo
    return None


camino = encoger_hasta_el_fondo(8000, PISO)

# 4. Convergencia. El `while True` de `_pedir_encogiendo` no tiene contador de
#    vueltas: lo único que lo termina es que este cálculo llegue al piso y
#    devuelva None. Si alguna vez dejara de decrecer, la integración se colgaría
#    reintentando y el usuario escucharía silencio otra vez, pero ahora eterno.
comprobar(
    "el encogido converge y no cuelga el bucle de reintentos",
    camino is not None,
)
comprobar(
    "cada vuelta pide estrictamente menos que la anterior",
    camino is not None and all(b < a for a, b in zip([8000, *camino], camino)),
)
comprobar(
    "ninguna vuelta baja del piso",
    camino is not None and all(t >= PISO for t in camino),
)
comprobar(
    "la última vuelta se planta justo en el piso",
    camino is not None and camino[-1] == PISO,
)


# =============================================================================
# C. `_detalle_error`: lo que queda escrito cuando algo falla.
# =============================================================================


class ErrorFalso(Exception):
    """Imita un groq.APIStatusError, que trae el cuerpo ya parseado."""

    def __init__(self, texto, body=None):
        super().__init__(texto)
        self.body = body


CUERPO_413 = {"error": {
    "message": ("Request too large for model `qwen/qwen3.8-27b` on tokens per "
                "minute (TPM): Limit 8000, Requested 8441."),
    "type": "tokens", "code": "rate_limit_exceeded"}}


# 1. Lo que importa y lo que costó una hora de diagnóstico: el mensaje con los
#    números sale entero, sin resumir ni reformatear. "Limit 8000, Requested
#    8441" es lo único que dice CUÁNTO hay que bajar.
detalle = M._detalle_error(ErrorFalso("413 boom", CUERPO_413))
comprobar(
    "saca el mensaje del cuerpo, no el str() de la excepción",
    detalle.startswith("Request too large"),
)
comprobar(
    "conserva el límite y lo pedido, que es con lo que se calibra",
    "8000" in detalle and "8441" in detalle,
)

# 2. Sin cuerpo utilizable se cae al texto de la excepción. Un log pobre es malo;
#    un log que revienta el camino de la respuesta es muchísimo peor. Cada nivel
#    del cuerpo se mira con isinstance sobre dict y no con `.get` encadenado.
comprobar(
    "sin body usa el texto de la excepción",
    M._detalle_error(ErrorFalso("429 Too Many Requests")) == "429 Too Many Requests",
)
comprobar(
    "body None no revienta",
    M._detalle_error(ErrorFalso("boom", None)) == "boom",
)
comprobar(
    "body que no es un dict no revienta",
    M._detalle_error(ErrorFalso("boom", "texto suelto")) == "boom",
)
comprobar(
    "body que es una lista no revienta",
    M._detalle_error(ErrorFalso("boom", [{"error": "x"}])) == "boom",
)
comprobar(
    "body sin la clave error no revienta",
    M._detalle_error(ErrorFalso("boom", {"otra": 1})) == "boom",
)

# 3. Esta comprobación existe porque un proveedor devolvía `error` como texto
#    suelto en lugar de objeto. Ese proveedor ya no está en el código, pero el
#    chequeo se queda: el que viene puede hacer lo mismo.
comprobar(
    "error que no es un dict no revienta",
    M._detalle_error(ErrorFalso("boom", {"error": "texto"})) == "boom",
)
comprobar(
    "error en None no revienta",
    M._detalle_error(ErrorFalso("boom", {"error": None})) == "boom",
)
comprobar(
    "error sin message no revienta",
    M._detalle_error(ErrorFalso("boom", {"error": {"type": "x"}})) == "boom",
)
comprobar(
    "message explícitamente en None cae al texto de la excepción",
    M._detalle_error(ErrorFalso("boom", {"error": {"message": None}})) == "boom",
)

# 4. La guarda es `is not None`, NO truthiness, y la diferencia se ve solo acá:
#    un message de "" o de 0 es un mensaje pobre, pero es el que mandó el
#    proveedor. Taparlo con el str() de la excepción sería inventar información
#    que no vino.
comprobar(
    "un message vacío se respeta en vez de taparse con el str()",
    M._detalle_error(ErrorFalso("boom", {"error": {"message": ""}})) == "",
)
comprobar(
    "un message de 0 se respeta igual",
    M._detalle_error(ErrorFalso("boom", {"error": {"message": 0}})) == "0",
)

# 5. Cota de largo. Esto va a un log que ya se inunda solo, y un proveedor con un
#    balanceador delante puede devolver una página HTML de error entera. El tope
#    aplica a las DOS ramas de salida, no solo a la del cuerpo.
largo = M._detalle_error(ErrorFalso("x", {"error": {"message": "y" * 5000}}))
comprobar(
    "recorta los mensajes desmedidos",
    len(largo) <= 300,
)
comprobar(
    "también recorta cuando cae al texto de la excepción",
    len(M._detalle_error(ErrorFalso("z" * 5000))) <= 300,
)

# 6. Un message que no es texto no debe romper el %s del log.
comprobar(
    "un message numérico se devuelve como texto",
    M._detalle_error(ErrorFalso("boom", {"error": {"message": 413}})) == "413",
)

# 7. Y lo que llega no siempre es una excepción del SDK: los enrutadores corren
#    con `asyncio.gather(return_exceptions=True)` y por ahí puede pasar
#    cualquier cosa.
comprobar(
    "algo que ni siquiera es una excepción no revienta",
    M._detalle_error("429 Too Many Requests") == "429 Too Many Requests",
)
comprobar(
    "None no revienta",
    M._detalle_error(None) == "None",
)


# =============================================================================
# D. `_decidir_tras_error`: rotar de modelo o dar la cara.
# =============================================================================

# 1. Los límites de Groq son POR MODELO, así que ante un error de cupo cada
#    eslabón de la cadena aporta su propia ventana: saltar no es resignarse, es
#    usar cupo que estaba libre.
comprobar(
    "429 con siguiente candidato salta",
    M._decidir_tras_error(429, "Rate limit reached", True) == SALTAR,
)

# 2. Un 413 que llega hasta acá ya se encogió todo lo que se podía (el piso de
#    400 lo frenó), así que rotar es lo último que queda aunque no sea probable
#    que ayude: al siguiente le llega la misma petición.
comprobar(
    "413 agotado con siguiente candidato salta",
    M._decidir_tras_error(413, "Request too large", True) == SALTAR,
)

# 3. Groq devuelve algunos rechazos de cupo con estados que no son 413 ni 429, y
#    lo único que los identifica es el `rate_limit_exceeded` del texto. Sin esta
#    rama esos errores se propagaban con toda la cadena de respaldo sin usar.
comprobar(
    "el texto rate_limit alcanza aunque el estado no sea 413 ni 429",
    M._decidir_tras_error(400, "rate_limit_exceeded", True) == SALTAR,
)

# 4. Cualquier otra cosa se propaga sin tocar el resto de la cadena. Un 400 por
#    un parámetro mal armado o un 401 por la clave le van a pasar IGUAL al
#    siguiente modelo: reintentar esconde el problema real y encima gasta la
#    ventana de tokens de un modelo que después va a hacer falta.
comprobar(
    "401 por la clave se propaga, no se disimula rotando",
    M._decidir_tras_error(401, "Invalid API Key", True) == PROPAGAR,
)
comprobar(
    "400 por un parámetro mal armado se propaga",
    M._decidir_tras_error(400, "decommissioned model", True) == PROPAGAR,
)
comprobar(
    "404 de modelo inexistente se propaga",
    M._decidir_tras_error(404, "model not found", True) == PROPAGAR,
)
comprobar(
    "500 del proveedor se propaga",
    M._decidir_tras_error(500, "internal server error", True) == PROPAGAR,
)

# 5. Sin siguiente candidato no hay a dónde saltar, y propagar es lo correcto:
#    la excepción entra en los manejadores del turno y sale como un "no pude"
#    hablado. Devolver "saltar" acá haría caer el bucle por el final, que es el
#    camino donde `result` quedaba en None y reventaba con un AttributeError
#    fuera de todo manejador.
comprobar(
    "429 en el último candidato se propaga",
    M._decidir_tras_error(429, "Rate limit reached", False) == PROPAGAR,
)
comprobar(
    "413 en el último candidato se propaga",
    M._decidir_tras_error(413, "Request too large", False) == PROPAGAR,
)
comprobar(
    "rate_limit en el último candidato se propaga",
    M._decidir_tras_error(400, "rate_limit_exceeded", False) == PROPAGAR,
)
comprobar(
    "401 en el último candidato se propaga",
    M._decidir_tras_error(401, "Invalid API Key", False) == PROPAGAR,
)

# 6. Los literales son parte del contrato: el llamador los compara con `==` y una
#    errata no rompe nada visible, simplemente hace que la rama nunca se tome.
comprobar(
    "nunca inventa un veredicto fuera del vocabulario",
    all(
        M._decidir_tras_error(estado, texto, hay) in (SALTAR, PROPAGAR)
        for estado in (400, 401, 404, 413, 429, 500, 503)
        for texto in ("", "rate_limit_exceeded", "boom")
        for hay in (True, False)
    ),
)


# =============================================================================
# E. `_decidir_tras_respuesta`: la mitad REACTIVA del fallo silencioso.
# =============================================================================

# 1. El camino normal: si hay respuesta, se acepta y se termina el turno. Da
#    igual lo que haya pasado antes.
comprobar(
    "una respuesta con texto se acepta",
    M._decidir_tras_respuesta(BUENA, False, True) == ACEPTAR,
)
comprobar(
    "una respuesta con texto se acepta también después de haber reintentado",
    M._decidir_tras_respuesta(BUENA, True, False) == ACEPTAR,
)

# 2. EL comportamiento que no tenía ninguna prueba: ante el vacío se repite UNA
#    sola vez contra el MISMO modelo con el pensamiento forzado a "none". Una
#    respuesta más seca es infinitamente mejor que el silencio, y volver a
#    preguntarle al mismo sale más barato que saltar: el siguiente de la cadena
#    suele ser peor y el salto gasta la ventana de otro modelo que quizá haga
#    falta después.
comprobar(
    "la primera respuesta vacía se repite sin pensamiento",
    M._decidir_tras_respuesta(VACIA, False, True) == REINTENTAR,
)

# 3. Y se repite aunque no haya ningún candidato detrás: el reintento no depende
#    de que exista un plan B, porque no consume otro modelo.
comprobar(
    "se repite sin pensamiento aun siendo el último candidato",
    M._decidir_tras_respuesta(VACIA, False, False) == REINTENTAR,
)

# 4. Segundo vacío del mismo modelo: ese ya no piensa, así que insistir sería
#    pedir lo mismo por tercera vez. Recién ahí se salta.
comprobar(
    "el segundo vacío seguido salta al candidato siguiente",
    M._decidir_tras_respuesta(VACIA, True, True) == SALTAR,
)

# 5. El final del camino: vacío, ya se repitió sin pensamiento y no queda ningún
#    modelo más. Se ACEPTA el vacío en vez de propagar un error, porque ya se
#    probó todo y el llamador tiene que devolver algo; el log lo grita en ERROR,
#    que es lo único que va a quedar escrito de un turno que el usuario escuchó
#    como silencio.
comprobar(
    "sin candidatos ni reintentos por delante se acepta el vacío",
    M._decidir_tras_respuesta(VACIA, True, False) == ACEPTAR,
)

# 6. LA GUARDA DE LA CASA ATRAVIESA LA MÁQUINA DE ESTADOS. Una respuesta truncada
#    que trae tool_calls no es este fallo, así que acá se acepta directo. Si esto
#    devolviera "reintentar_sin_pensamiento" se volvería a pedir la misma acción
#    sobre la casa y el usuario vería la luz apagarse dos veces.
CON_HERRAMIENTA = respuesta("length", "", tool_calls=[SimpleNamespace(id="x")])
comprobar(
    "una truncada con tool_calls se acepta en vez de repetirse",
    M._decidir_tras_respuesta(CON_HERRAMIENTA, False, True) == ACEPTAR,
)
comprobar(
    "y se acepta igual en la segunda pasada",
    M._decidir_tras_respuesta(CON_HERRAMIENTA, True, True) == ACEPTAR,
)

# 7. Formas deformes: la decisión también corre en el camino de la respuesta al
#    usuario, y la reacción por defecto ante algo que no se entiende es
#    aceptarlo, no reintentar. Reintentar sobre basura gastaría cupo en vano.
comprobar(
    "un objeto sin choices se acepta en vez de reventar",
    M._decidir_tras_respuesta(SimpleNamespace(), False, True) == ACEPTAR,
)
comprobar(
    "choices vacío se acepta en vez de reventar",
    M._decidir_tras_respuesta(SimpleNamespace(choices=[]), False, True) == ACEPTAR,
)

# 8. "propagar" es vocabulario del otro lado de la máquina de estados: acá ya hay
#    una respuesta en la mano, no una excepción, así que no hay nada que
#    propagar. Mezclarlos haría que el llamador no reconociera el veredicto.
comprobar(
    "nunca devuelve un veredicto de la rama de errores",
    all(
        M._decidir_tras_respuesta(res, ya, hay) in (ACEPTAR, REINTENTAR, SALTAR)
        for res in (BUENA, VACIA, CON_HERRAMIENTA, SimpleNamespace())
        for ya in (True, False)
        for hay in (True, False)
    ),
)


# =============================================================================
# F. Las dos funciones juntas, que es como corren: el bucle de candidatos.
# =============================================================================


def recorrer(candidatos, mudos, mudos_solo_pensando=()):
    """Simula el bucle de `_responder_con_cadena` usando solo las puras.

    `mudos` son los que devuelven vacío siempre, incluso con el pensamiento en
    "none". `mudos_solo_pensando` son los que se callan pensando pero contestan
    en el reintento, que es el caso medido en la instalación real. Devuelve la
    traza de (candidato, veredicto) para poder exigir el ORDEN de las
    decisiones, no solo el resultado final.
    """
    traza = []
    ultimo = len(candidatos) - 1
    for pos, candidato in enumerate(candidatos):
        hay_siguiente = pos < ultimo
        callado = candidato in mudos or candidato in mudos_solo_pensando
        accion = M._decidir_tras_respuesta(VACIA if callado else BUENA,
                                           False, hay_siguiente)
        traza.append((candidato, accion))
        if accion == REINTENTAR:
            # El reintento va con el pensamiento forzado a "none": el que solo
            # se callaba por pensar, acá contesta.
            accion = M._decidir_tras_respuesta(
                VACIA if candidato in mudos else BUENA, True, hay_siguiente)
            traza.append((candidato, accion))
        if accion == SALTAR:
            continue
        return traza
    return traza


# 1. El caso feliz: el principal contesta y nadie salta.
comprobar(
    "con el principal sano no se toca la cadena",
    recorrer(["principal", "respaldo"], set()) == [("principal", ACEPTAR)],
)

# 2. El caso medido el 31 ago: el modelo se queda mudo porque gastó los 1200
#    tokens pensando, se lo repite sin pensamiento y con eso alcanza. La cadena
#    queda intacta, que es justo el ahorro que persigue esta política: el
#    siguiente eslabón suele contestar peor y su ventana de cupo va a hacer
#    falta más adelante.
comprobar(
    "un mudo que se arregla sin pensamiento no gasta otro modelo",
    recorrer(["principal", "respaldo"], set(), {"principal"})
    == [("principal", REINTENTAR), ("principal", ACEPTAR)],
)
traza = recorrer(["mudo", "respaldo"], {"mudo"})
comprobar(
    "un mudo irrecuperable primero se repite y recién después salta",
    [a for _, a in traza[:2]] == [REINTENTAR, SALTAR],
)
comprobar(
    "y el que termina contestando es el siguiente de la cadena",
    traza[-1] == ("respaldo", ACEPTAR),
)

# 3. El peor escenario, y el que garantiza que el bucle SIEMPRE termina: todos
#    mudos. Cada uno se repite una vez, se salta, y el último acepta su vacío en
#    vez de caer por el final del bucle. Ese final era el `result = None` que
#    reventaba con un AttributeError fuera de todos los manejadores del turno.
CADENA = ["uno", "dos", "tres"]
traza = recorrer(CADENA, set(CADENA))
comprobar(
    "con toda la cadena muda se prueba cada modelo dos veces",
    len(traza) == 2 * len(CADENA),
)
comprobar(
    "cada modelo se repite sin pensamiento antes de que se lo abandone",
    [a for _, a in traza] == [REINTENTAR, SALTAR, REINTENTAR, SALTAR,
                              REINTENTAR, ACEPTAR],
)
comprobar(
    "el bucle nunca se sale por el final: el último acepta",
    traza[-1] == ("tres", ACEPTAR),
)

# 4. Y la propiedad que hace inalcanzable el `raise` final de
#    `_responder_con_cadena`: para el último candidato ninguna de las dos
#    funciones puede decir "saltar", porque no hay a dónde.
comprobar(
    "el último candidato nunca recibe la orden de saltar",
    all(
        M._decidir_tras_respuesta(res, ya, False) != SALTAR
        for res in (BUENA, VACIA, CON_HERRAMIENTA, SimpleNamespace())
        for ya in (True, False)
    )
    and all(
        M._decidir_tras_error(estado, texto, False) != SALTAR
        for estado in (400, 401, 413, 429, 500)
        for texto in ("", "rate_limit_exceeded")
    ),
)


# ---------------------------------------------------------------------------
# _argumentos_de_herramienta — la segunda fragilidad de la nota 2.27
#
# Antes era un `json.loads` pelado. Un modelo chico que devolvía JSON cortado
# levantaba JSONDecodeError FUERA de todos los except del turno y rompía el
# pipeline entero de Assist, en vez de degradar a "rehacé el turno sin
# herramientas". Es de los pocos fallos que no terminan en silencio sino en que
# Assist deja de funcionar.
# ---------------------------------------------------------------------------

# 1. EL caso que hay que clavar, y la razón por la que la firma devuelve None y
#    no un dict vacío: una herramienta SIN parámetros manda "{}" legítimamente.
#    Eso es `{}`, que es falsy pero válido. Un llamador que preguntara `if not
#    argumentos` trataría una llamada correcta como malformada, así que la
#    distinción tiene que ser contra `is None` y acá se fija.
comprobar(
    "argumentos vacíos son un dict vacío, NO None",
    M._argumentos_de_herramienta("{}") == {} and M._argumentos_de_herramienta("{}") is not None,
)

# 2. El camino feliz.
comprobar(
    "un objeto JSON se devuelve como dict",
    M._argumentos_de_herramienta('{"name": "luz", "area": "dormitorio"}')
    == {"name": "luz", "area": "dormitorio"},
)

# 3. JSON cortado a la mitad: el caso real del modelo chico que se queda sin
#    tokens en medio de la llamada.
comprobar(
    "JSON truncado devuelve None en vez de reventar",
    M._argumentos_de_herramienta('{"name": "lu') is None,
)

# 4. None y vacío: el SDK puede devolver `arguments` ausente.
comprobar(
    "None no revienta",
    M._argumentos_de_herramienta(None) is None,
)
comprobar(
    "cadena vacía no revienta",
    M._argumentos_de_herramienta("") is None,
)

# 5. JSON VÁLIDO pero que no es un objeto. Es el caso traicionero: `json.loads`
#    no se queja, así que sin el chequeo de tipo la lista o el número viajaban
#    hasta donde se los desempaqueta como mapa de parámetros y reventaban ahí,
#    varias capas más adentro, donde ya no hay manejador.
for crudo in ('["luz"]', "42", '"texto"', "true", "null"):
    comprobar(
        f"JSON válido que no es objeto: {crudo} -> None",
        M._argumentos_de_herramienta(crudo) is None,
    )

resumen("lectura de respuestas y decisiones del bucle de candidatos")
