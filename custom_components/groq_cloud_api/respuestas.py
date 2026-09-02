"""Lectura de una respuesta de Groq (o de un error) y decisión de qué hacer con ella.

Frontera del módulo: acá no se habla con la red ni con Home Assistant. Todo lo
que entra es un objeto cualquiera con atributos —una `ChatCompletion` del SDK,
una excepción, o el `SimpleNamespace` que arman las pruebas— y todo lo que sale
es un bool, un texto o un veredicto. Ese es el motivo de que se lea TODO con
`getattr` sobre duck types y nunca con `isinstance` contra clases del SDK: esto
corre en el camino de la respuesta al usuario y una excepción acá se oye como
un error hablado.

Además de leer, acá vive la máquina de estados del bucle de candidatos
(`_decidir_tras_error` y `_decidir_tras_respuesta`), que antes estaba enterrada
dentro de un método de la entidad y por eso no tenía ni una prueba.
"""

from __future__ import annotations

import json


def _vacia_por_truncado(result: object) -> bool:
    """¿El modelo gastó todo max_tokens razonando y no llegó a contestar?

    `finish_reason == "length"` con el contenido vacío es la firma exacta, y es
    el peor fallo posible porque no se parece a un fallo: no hay excepción, el
    pipeline de Assist da la vuelta entera como si todo hubiera salido bien, no
    emite `synthesize` y el usuario se queda escuchando silencio. Hay que
    tratarlo como un modelo que falló, no como una respuesta.

    Medido el 31 ago 2026 contra la instalación real: el principal contestaba
    pensando en 120-180 tokens, mientras que el de respaldo quemaba los 1200 de
    `max_tokens` razonando y volvía vacío 2 de 2 veces.
    """
    eleccion = (getattr(result, "choices", None) or [None])[0]
    if eleccion is None or getattr(eleccion, "finish_reason", None) != "length":
        return False
    mensaje = getattr(eleccion, "message", None)
    # GUARDA INTOCABLE, y va ANTES del chequeo de contenido: truncar DESPUÉS de
    # pedir una herramienta no es este caso. La petición está completa y sirve
    # aunque no venga texto acompañándola, y darla por vacía haría repetir
    # acciones sobre la casa del usuario (apagar dos veces, abrir dos veces).
    if getattr(mensaje, "tool_calls", None):
        return False
    # `message` en None con finish_reason "length" cae acá y devuelve True a
    # propósito: si se cortó por longitud y no hay mensaje, no hay nada que
    # decirle al usuario y lo que corresponde es reintentar.
    return not (getattr(mensaje, "content", None) or "").strip()


def _detalle_error(err: object) -> str:
    """El mensaje que manda el proveedor, no solo el número de estado.

    Sin esto en el log queda únicamente el código, y 413 y 429 se vuelven
    indistinguibles de un vistazo siendo problemas OPUESTOS: uno se arregla
    pidiendo menos, el otro esperando. El cuerpo del 413 además dice el límite
    y cuánto se pidió ("Limit 8000, Requested 8441"), que son los dos números
    con los que se calibra `max_tokens`.
    """
    # Todo con isinstance sobre dicts en vez de `.get` encadenado: un proveedor
    # que devuelva `error` como texto suelto en lugar de objeto no puede tumbar
    # el camino de la respuesta al usuario.
    cuerpo = getattr(err, "body", None)
    if isinstance(cuerpo, dict):
        error = cuerpo.get("error")
        # `is not None` y no truthiness: un message de 0 o de "" es un mensaje
        # pobre pero es el que mandó el proveedor, y taparlo con el str() de la
        # excepción pierde información.
        if isinstance(error, dict) and error.get("message") is not None:
            return str(error["message"])[:300]
    # El recorte a 300 aplica a las DOS ramas: un proveedor puede devolver un
    # HTML de error entero a un log que ya se inunda solo.
    return str(err)[:300]


def _texto_de(result: object) -> str:
    """El contenido de la respuesta, o cadena vacía si no hay nada que leer."""
    eleccion = (getattr(result, "choices", None) or [None])[0]
    mensaje = getattr(eleccion, "message", None)
    texto = getattr(mensaje, "content", None)
    # Se filtra por tipo porque el llamador hace len() sobre esto para el log:
    # un content que no sea texto no puede reventar la línea de auditoría.
    return texto if isinstance(texto, str) else ""


def _peticiones_de_herramienta(result: object) -> list:
    """Los tool_calls crudos de la respuesta, o lista vacía.

    No interpreta los argumentos a propósito: de eso se ocupa
    `_argumentos_de_herramienta`, que puede fallar por su cuenta sin arrastrar
    a las demás llamadas del mismo turno.
    """
    eleccion = (getattr(result, "choices", None) or [None])[0]
    mensaje = getattr(eleccion, "message", None)
    return list(getattr(mensaje, "tool_calls", None) or [])


def _argumentos_de_herramienta(crudo: str | None) -> dict | None:
    """Los argumentos JSON de una tool_call, o None si volvieron malformados.

    Antes esto era un `json.loads` pelado sin manejador: un modelo chico que
    devolvía JSON cortado levantaba `JSONDecodeError` fuera de todos los except
    del turno y rompía el pipeline entero de Assist en vez de degradar.

    None es la señal de "tool call malformada" y el llamador la trata igual que
    un `tool_use_failed`: rehace el turno sin herramientas. OJO al leer el
    resultado: una herramienta sin parámetros manda `"{}"` y eso devuelve `{}`,
    que es falsy pero perfectamente válido. Hay que comparar con `is None`.
    """
    try:
        argumentos = json.loads(crudo)
    except (TypeError, ValueError):
        return None
    # Un JSON válido que no sea un objeto (una lista, un número) tampoco sirve:
    # del otro lado se lo desempaqueta como mapa de parámetros y reventaría más
    # adentro, donde ya no hay manejador que lo atrape.
    return argumentos if isinstance(argumentos, dict) else None


def _encoger_tope(estado: int, tope: int, piso: int) -> int | None:
    """El nuevo max_tokens tras un rechazo por TAMAÑO, o None si no hay que encoger.

    HTTP 413 no es falta de cupo, aunque durante mucho tiempo el código lo
    tratara igual que un 429: significa que la petición entera —la entrada, que
    con el volcado de herramientas de la casa ya arranca en unos 4150 tokens,
    más el techo de generación— no entra de una sola vez. Rotar de modelo ahí
    no arregla nada, porque al siguiente le llega exactamente lo mismo y lo
    rechaza igual. Lo único que ayuda es pedir menos.

    Un 429 SÍ es falta de cupo (el límite medido es de 8000 TPM), y ahí lo que
    corresponde es rotar de modelo, no encoger: por eso devuelve None.
    """
    if estado != 413:
        return None
    # Por debajo del piso la respuesta sale cortada a media frase, que por voz
    # se entiende peor que un "no pude": ahí conviene dejar de encoger y fallar.
    if tope <= piso:
        return None
    return max(piso, tope // 2)


def _decidir_tras_error(estado: int, texto: str, hay_siguiente: bool) -> str:
    """Ante un error del proveedor: saltar al siguiente candidato o propagar.

    Solo se salta si el error es de límite. Cualquier otra cosa (un 400 por un
    parámetro mal armado, un 401 por la clave) se propaga sin tocar el resto de
    la cadena: reintentar en otro modelo esconde el problema real y gasta la
    ventana de tokens de un modelo que después va a hacer falta.
    """
    # Un 413 que llega hasta acá ya se encogió todo lo que se podía, así que
    # rotar es lo último que queda aunque no sea probable que ayude.
    limitado = estado in (413, 429) or "rate_limit" in texto
    if limitado and hay_siguiente:
        return "saltar"
    return "propagar"


def _decidir_tras_respuesta(result: object, ya_reintento: bool,
                            hay_siguiente: bool) -> str:
    """Qué hacer con una respuesta ya recibida: aceptarla, repetirla o saltar.

    Es la mitad reactiva del fallo silencioso: `_vacia_por_truncado` lo detecta
    y esto decide. Se repite UNA sola vez contra el MISMO modelo con el
    pensamiento forzado a "none", porque una respuesta más seca es
    infinitamente mejor que el silencio y volver a preguntarle al mismo modelo
    sale más barato que saltar: el siguiente de la cadena suele ser peor y el
    salto gasta la ventana de otro modelo que quizá haga falta después.
    """
    if not _vacia_por_truncado(result):
        return "aceptar"
    if not ya_reintento:
        return "reintentar_sin_pensamiento"
    if hay_siguiente:
        return "saltar"
    # Se acepta el vacío: ya se probó cada modelo de la cadena y cada uno
    # también sin pensamiento. No queda nada mejor que devolver, y el llamador
    # lo grita en ERROR porque es lo único que va a quedar escrito de un turno
    # que el usuario escuchó como silencio.
    return "aceptar"


def _resumen_uso(result: object, cantidad_mensajes: int, modelo: str) -> str:
    """La línea de uso de tokens lista para el log, o "" si no hay usage.

    Sin estos números hay que adivinar cuánto pesa cada petición, y el límite
    de Groq es por minuto contando ENTRADA + SALIDA (8000 TPM en el plan
    gratuito), así que son los que dicen si el recorte del historial está bien
    calibrado o no.
    """
    uso = getattr(result, "usage", None)
    if uso is None:
        return ""
    # cached_tokens es EL número que decide si conviene partir el trabajo en
    # dos modelos: los tokens cacheados no cuentan contra el límite por minuto.
    # Si se queda cerca de 0, el caché no está pegando y no hay ahorro que
    # perseguir; y cada vez que el enrutador de casa cambia de opinión el
    # prompt de sistema cambia de tamaño y rompe el prefijo cacheado.
    detalle = getattr(uso, "prompt_tokens_details", None)
    cacheados = getattr(detalle, "cached_tokens", None)
    eleccion = (getattr(result, "choices", None) or [None])[0]
    motivo = getattr(eleccion, "finish_reason", None)
    return (
        f"Groq OK [{modelo}] — entrada {getattr(uso, 'prompt_tokens', None)} "
        f"(cacheados {cacheados}), salida "
        f"{getattr(uso, 'completion_tokens', None)}, total "
        f"{getattr(uso, 'total_tokens', None)} ({cantidad_mensajes} mensajes, "
        f"fin={motivo}, {len(_texto_de(result))} car.)"
    )
