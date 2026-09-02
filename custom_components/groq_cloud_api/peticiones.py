"""La única capa que le habla a Groq para la respuesta final.

Frontera del módulo: acá se hace la llamada de red y se recorre la cadena de
candidatos, pero no se decide nada. Cada decisión —si encoger el techo de
tokens, si saltar de modelo, si repetir sin pensamiento— se la pregunta a las
funciones puras de `respuestas.py`, que sí tienen pruebas. Lo que queda acá es
el orden de las cosas y los efectos: la llamada, el log y la marca de uso.

Ninguna función de este archivo se carga por AST: todas son async y hablan con
el SDK de groq.
"""

from __future__ import annotations

import time

import groq

from .const import LOGGER
from .razonamiento import _aplicar_razonamiento, _esfuerzo_del_candidato
from .respuestas import (
    _decidir_tras_error,
    _decidir_tras_respuesta,
    _detalle_error,
    _encoger_tope,
)


async def _pedir(cliente: object, kwargs: dict) -> object:
    """Una llamada al modelo, sin encoger ni reintentar."""
    return await cliente.chat.completions.create(**kwargs)


async def _pedir_encogiendo(cliente: object, kwargs: dict, etiqueta: str,
                            piso: int) -> object:
    """Como `_pedir`, pero achica la petición si la rechazan por tamaño.

    Un HTTP 413 no se arregla rotando de modelo: al siguiente candidato le
    llega exactamente la misma petición sobredimensionada y la rechaza igual.
    En el log eso se veía como pares 413 -> 413 instantáneos, sin que nadie
    contestara nunca. Lo único que ayuda es pedir menos.

    El tope reducido se deja escrito en `kwargs` a propósito: si la petición no
    entraba acá, tampoco va a entrar en el siguiente candidato. El bucle
    converge porque el tope decrece a la mitad hasta tocar el piso, y ahí
    `_encoger_tope` devuelve None y el error se propaga.
    """
    while True:
        try:
            return await _pedir(cliente, kwargs)
        except groq.APIStatusError as err:
            tope = kwargs.get("max_tokens") or 0
            nuevo = _encoger_tope(err.status_code, tope, piso)
            if nuevo is None:
                raise
            kwargs["max_tokens"] = nuevo
            LOGGER.warning(
                "%s rechazó la petición por TAMAÑO (HTTP 413), no por cupo: "
                "%s. Bajo max_tokens de %s a %s y reintento el mismo modelo.",
                etiqueta, _detalle_error(err), tope, nuevo,
            )


async def _responder_con_cadena(cliente: object, kwargs: dict,
                                candidatos: list[str], principal: str,
                                esfuerzo: str | None,
                                ultimo_uso: dict[str, float],
                                piso: int) -> object:
    """Recorre los candidatos hasta conseguir una respuesta usable.

    Los límites de Groq son POR MODELO, así que cada eslabón de la cadena
    aporta su propia ventana de tokens por minuto: saltar no es resignarse, es
    usar cupo que estaba libre.
    """
    if not candidatos:
        # Antes esto dejaba `result` en None y reventaba con un AttributeError
        # fuera de todos los except del turno, que se escucha como un error
        # crudo en voz alta. Un GroqError entra en los manejadores del turno y
        # sale como un "no pude" hablado.
        raise groq.GroqError(
            "No hay ningún modelo configurado para responder: revisá el modelo "
            "principal y la cadena de respaldo en las opciones."
        )

    ultimo = len(candidatos) - 1
    for pos, candidato in enumerate(candidatos):
        hay_siguiente = pos < ultimo
        kwargs["model"] = candidato
        # Quién es el principal se decide por IDENTIDAD, nunca por posición: el
        # titular reaparece más abajo en la rotación cuando está en
        # enfriamiento, y ahí sigue siendo el titular.
        _aplicar_razonamiento(
            kwargs, candidato,
            _esfuerzo_del_candidato(esfuerzo, candidato == principal),
        )

        try:
            result = await _pedir_encogiendo(cliente, kwargs, candidato, piso)
        except groq.APIStatusError as err:
            # Se marca como usado TAMBIÉN al fallar, y antes de decidir si se
            # salta: su ventana de tokens quedó ocupada igual, que es justo lo
            # que hay que recordar para no volver a elegirlo ya mismo.
            ultimo_uso[candidato] = time.monotonic()
            if _decidir_tras_error(err.status_code, str(err),
                                   hay_siguiente) == "saltar":
                LOGGER.warning(
                    "%s no pudo (HTTP %s: %s), salto a %s sin esperar",
                    candidato, err.status_code, _detalle_error(err),
                    candidatos[pos + 1],
                )
                continue
            raise
        ultimo_uso[candidato] = time.monotonic()

        accion = _decidir_tras_respuesta(result, False, hay_siguiente)
        if accion == "reintentar_sin_pensamiento":
            LOGGER.warning(
                "%s gastó los %s tokens de max_tokens pensando y volvió "
                "vacío; repito sin pensamiento",
                candidato, kwargs.get("max_tokens"),
            )
            _aplicar_razonamiento(kwargs, candidato, "none")
            try:
                result = await _pedir_encogiendo(cliente, kwargs, candidato,
                                                 piso)
            except groq.APIStatusError as err:
                # No se re-lanza a propósito: nos quedamos con el `result`
                # vacío que ya teníamos y que decida el paso siguiente. Perder
                # el turno entero acá sería peor que saltar al que sigue.
                LOGGER.warning(
                    "el reintento sin pensamiento de %s tampoco entró "
                    "(HTTP %s: %s)",
                    candidato, err.status_code, _detalle_error(err),
                )
            finally:
                ultimo_uso[candidato] = time.monotonic()
            accion = _decidir_tras_respuesta(result, True, hay_siguiente)

        if accion == "saltar":
            LOGGER.warning(
                "%s sigue devolviendo vacío, salto a %s",
                candidato, candidatos[pos + 1],
            )
            continue
        return result

    # Inalcanzable mientras la máquina de estados se respete: para el último
    # candidato `hay_siguiente` es False, así que nunca devuelve "saltar" y
    # ante un error devuelve "propagar". Se deja igual porque salir de acá
    # devolviendo None es exactamente el fallo que se acaba de cerrar arriba.
    raise groq.GroqError(
        "Se agotó la cadena de modelos sin respuesta ni error que propagar."
    )
