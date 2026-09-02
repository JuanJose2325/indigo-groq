"""Transformaciones sobre la lista de mensajes ya convertida al formato de Groq.

Frontera: módulo PURO. Todo lo que entra y sale son dicts planos con las claves
que espera la API de Groq; ninguna función de acá toca `chat_log` ni un tipo de
Home Assistant. Esa separación no es estética: `chat_log.content` es la lista
que HA persiste y sus elementos son dataclasses frozen, así que cualquier
limpieza o recorte tiene que pasar por copias propias. De yapa es lo que deja
que `pruebas/cargar.py` cargue estas cuatro funciones por AST: el import de
Home Assistant de acá arriba nunca se ejecuta en las pruebas, porque el
cargador se queda solo con los nodos de función e inyecta `json_dumps`.
"""

from __future__ import annotations

from homeassistant.helpers.json import json_dumps

from .const import CHARS_PER_TOKEN, LOGGER


def _coste_aproximado(mensaje: dict) -> int:
    """Tokens estimados de un mensaje ya convertido al formato de Groq.

    Solo se usa para decidir el recorte, nunca se manda a la API, así que
    alcanza con una estimación conservadora: pasarse un poco de largo es
    inofensivo, quedarse corto significa comerse la ventana de 8000 tokens por
    minuto y que a partir de ahí el turno falle SIEMPRE.
    """
    total = len(str(mensaje.get("content") or ""))
    # Los argumentos serializados de una tool_call pesan tanto como el texto y
    # no viven en `content`: sin contarlos, un turno con herramientas se
    # subestima entero y el presupuesto se pasa sin que nadie se entere.
    for llamada in mensaje.get("tool_calls") or ():
        total += len(json_dumps(llamada))
    return int(total / CHARS_PER_TOKEN) + 4  # +4 por el envoltorio del rol


def _recortar_historial(messages: list, presupuesto: int) -> list:
    """Deja el prompt de sistema y los turnos más recientes que entren en el presupuesto.

    El límite de Groq cuenta ENTRADA + SALIDA por minuto, y la petición crece
    en cada turno porque el historial entero se reenvía. Sin recorte, la
    conversación termina superando los 8000 tokens y a partir de ahí falla
    SIEMPRE (no de a ratos): por eso antes se "arreglaba" sola al reiniciar,
    que es cuando Home Assistant descarta el conversation_id.

    Se descartan turnos viejos ENTEROS, nunca pedazos de un mensaje.
    """
    sistema = [m for m in messages if m.get("role") == "system"]
    resto = [m for m in messages if m.get("role") != "system"]

    disponible = presupuesto - sum(_coste_aproximado(m) for m in sistema)
    if disponible <= 0:
        # El prompt de sistema solo ya no entra: no hay nada que recortar que
        # ayude, y mandar la conversación vacía sería peor que dejarla pasar.
        LOGGER.warning(
            "El prompt de sistema (~%d tokens) supera el presupuesto de %d; "
            "no se recorta nada",
            presupuesto - disponible, presupuesto,
        )
        return messages

    ventana: list = []
    for mensaje in reversed(resto):
        coste = _coste_aproximado(mensaje)
        if coste > disponible:
            break
        disponible -= coste
        ventana.insert(0, mensaje)

    # Una ventana que empiece por un resultado de herramienta, o por un
    # assistant cuyas tool_calls quedaron fuera, es un JSON inválido para la
    # API. Se recorta hasta el primer mensaje de usuario, que siempre es un
    # corte limpio.
    while ventana and ventana[0].get("role") != "user":
        ventana.pop(0)

    if len(ventana) < len(resto):
        LOGGER.debug(
            "Historial recortado: %d de %d mensajes (presupuesto %d tokens)",
            len(ventana), len(resto), presupuesto,
        )

    # El sistema va primero aunque en la entrada estuviera en cualquier lado:
    # es lo único que la API exige que no se mueva de posición.
    return sistema + ventana


def _sin_herramientas(messages: list) -> list:
    """Copia del historial sin mensajes role=tool y sin tool_calls en los del asistente.

    Cuando el turno va sin herramientas hay que rehacer la lista, no alcanza
    con no mandar el bloque `tools`: un `role="tool"` suelto, o un assistant con
    `tool_calls` que apuntan a una herramienta que ya no existe, es un 400 seco.
    El bloque de herramientas de Assist pesa 4150 de los 8000 tokens por
    minuto, así que este camino es el que hace que valga la pena sacarlo.

    NO muta nada de lo que recibe: devuelve dicts nuevos. La razón está un
    escalón más arriba: estos dicts salen de `chat_log.content`, que es la lista
    que Home Assistant persiste, y sus elementos son dataclasses frozen.
    """
    limpios: list = []
    for mensaje in messages:
        rol = mensaje.get("role")
        if rol == "tool":
            continue
        if rol != "assistant":
            limpios.append(dict(mensaje))
            continue
        copia = {clave: valor for clave, valor in mensaje.items()
                 if clave != "tool_calls"}
        # Un assistant que solo traía tool_calls queda sin nada que decir, y si
        # se deja viaja como {"role":"assistant","content":null} en todos los
        # turnos siguientes de la sesión, sumando ruido y tokens para siempre.
        if not str(copia.get("content") or "").strip():
            continue
        limpios.append(copia)
    return limpios


def _ultimos_turnos(messages: list, cantidad: int) -> str:
    """Los últimos `cantidad` turnos de usuario y asistente, formateados para el enrutador de casa.

    Tienen que venir los DOS lados. El caso real que lo obliga: "¿tengo luces en
    el cuarto?" -> "sí, una" -> "apagala". Ese "apagala" no se puede clasificar
    sin ver la respuesta de la IA; mandando solo los turnos del usuario, el
    enrutador se queda sin el sustantivo y decide a ciegas.
    """
    utiles: list = []
    for mensaje in messages:
        rol = mensaje.get("role")
        # El system se saltea explícitamente: en la primera conversión del turno
        # todavía es el prompt del turno ANTERIOR, no vacío. Y los resultados de
        # herramienta son JSON crudo, ruido puro para clasificar una frase.
        if rol not in ("user", "assistant"):
            continue
        texto = str(mensaje.get("content") or "").strip()
        if rol == "assistant" and not texto:
            continue
        utiles.append((rol, texto))

    # El último user es la consulta que se está clasificando; va aparte en el
    # prompt, y repetida acá haría que el modelo la lea como contexto previo.
    if utiles and utiles[-1][0] == "user":
        utiles.pop()

    if cantidad <= 0:
        return ""
    utiles = utiles[-(cantidad * 2):]

    # 200 caracteres alcanzan para el sustantivo que resuelve el pronombre, y
    # el enrutador tiene un techo de 150 tokens de salida: inflarle la entrada
    # con párrafos enteros lo único que hace es sumarle latencia al turno.
    return "\n".join(
        f"{'usuario' if rol == 'user' else 'asistente'}: {texto[:200]}"
        for rol, texto in utiles
    )
