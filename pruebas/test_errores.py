#!/usr/bin/env python3
"""Pruebas del detalle de error que se escribe en el log (`_detalle_error`).

Lo que se está protegiendo acá es una hora de diagnóstico perdida. El log
guardaba solo el número de estado, y con eso 413 y 429 se leen igual —«no
pudo»— siendo problemas OPUESTOS: el 429 se arregla esperando, el 413 se
arregla pidiendo menos. Rotar de modelo ante un 413 no sirve para nada, porque
al siguiente le llega la misma petición sobredimensionada.

El cuerpo del 413 de Groq además trae el límite y cuánto se pidió, que son los
dos números con los que se calibra `max_tokens`. Perder eso es perder la única
pista que dice cuánto hay que bajar.
"""

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_detalle_error"])


class ErrorFalso(Exception):
    """Imita un groq.APIStatusError, que trae el cuerpo ya parseado."""

    def __init__(self, texto, body=None):
        super().__init__(texto)
        self.body = body


CUERPO_413 = {"error": {
    "message": ("Request too large for model `qwen/qwen3.8-27b` on tokens per "
                "minute (TPM): Limit 8000, Requested 8441."),
    "type": "tokens", "code": "rate_limit_exceeded"}}


# 1. Lo que importa: el mensaje con los números sale entero.
detalle = M._detalle_error(ErrorFalso("413 boom", CUERPO_413))
comprobar(
    "saca el mensaje del cuerpo, no el str() de la excepción",
    detalle.startswith("Request too large"),
)
comprobar(
    "conserva el límite y lo pedido, que es con lo que se calibra",
    "8000" in detalle and "8441" in detalle,
)

# 2. Sin cuerpo utilizable se cae al texto de la excepción. Un log pobre es
#    malo; un log que revienta el camino de la respuesta es peor.
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
    "body sin la clave error no revienta",
    M._detalle_error(ErrorFalso("boom", {"otra": 1})) == "boom",
)
comprobar(
    "error que no es un dict no revienta",
    M._detalle_error(ErrorFalso("boom", {"error": "texto"})) == "boom",
)
comprobar(
    "error sin message no revienta",
    M._detalle_error(ErrorFalso("boom", {"error": {"type": "x"}})) == "boom",
)

# 3. Cota de largo: esto va a un log que ya se inunda solo, y un proveedor
#    puede devolver un HTML de error entero.
largo = M._detalle_error(ErrorFalso("x", {"error": {"message": "y" * 5000}}))
comprobar(
    "recorta los mensajes desmedidos",
    len(largo) <= 300,
)
comprobar(
    "también recorta cuando cae al texto de la excepción",
    len(M._detalle_error(ErrorFalso("z" * 5000))) <= 300,
)

# 4. Un message que no es texto no debe romper el formateo del log.
comprobar(
    "un message numérico se devuelve como texto",
    M._detalle_error(ErrorFalso("boom", {"error": {"message": 413}})) == "413",
)

resumen("detalle de error para el log")
