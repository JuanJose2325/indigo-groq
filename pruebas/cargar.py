"""Carga las funciones puras de la integración sin importar Home Assistant.

Los módulos de `custom_components/groq_cloud_api/` importan `homeassistant` y el
SDK de `groq` en las primeras líneas, así que un `import` normal desde una
máquina cualquiera falla. Pero las funciones que importa probar —recorte de
historial, rotación de la cadena, traducción del esfuerzo de razonamiento,
veredictos de los enrutadores— no tocan ninguna de esas dos cosas: son funciones
sobre diccionarios y listas.

Esto lee el archivo, se queda con los nodos del AST que hacen falta y los ejecuta
en un espacio de nombres con las pocas dependencias necesarias sustituidas por
versiones mínimas. Así las pruebas corren contra el código REAL —si alguien lo
edita, se rompen— sin necesidad de instalar HA.

El código dejó de vivir en un solo archivo, así que la ruta ya no puede ser fija:
`modulo` elige de cuál de los archivos del paquete se extraen los nodos. Que se
pueda mudar una función de archivo sin renombrarla es deliberado (enmienda E2):
lo que las pruebas buscan por AST es el NOMBRE, y el archivo es un parámetro.

TRES COSAS QUE HAY QUE SABER ANTES DE TOCAR ESTO, porque cuando fallan no fallan
de a poco: se pierden las 329 comprobaciones de golpe.

1. Los `import` del módulo NUNCA se ejecutan. Un archivo puede tener
   `from .const import LOGGER` arriba sin problema; lo que rompe es que el CUERPO
   de una función pura use un nombre importado que este cargador no inyecta.

2. `from __future__ import annotations` no viaja: ese `ImportFrom` no está entre
   los nodos elegidos y `compile()` arranca sin la flag de futuro, así que las
   anotaciones se evalúan en tiempo de definición. Por eso ninguna función pura
   puede anotarse con `Any` ni con tipos de HA o del SDK —para los duck types se
   usa `object`— y por eso `Any` sigue inyectado acá: como red, no como permiso.
   `str | None`, `list[str]` y `dict[str, float]` no necesitan nada, son nativos.

3. Todos los nodos comparten UN mismo dict de globals. Si una función llama a
   otra, o usa una tabla de nivel superior, las dos tienen que pedirse en la
   MISMA llamada a `cargar()`; si no, el `NameError` salta recién al ejecutar.
   Por eso las tandas piden listas y no nombres sueltos. Y como el espacio es uno
   solo por archivo, una función pura no puede llamar a una de OTRO módulo: eso
   no se arregla acá, se arregla duplicando el dato o pasándolo por parámetro.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

PAQUETE = (
    Path(__file__).resolve().parent.parent
    / "custom_components"
    / "groq_cloud_api"
)


def _nombres_definidos(nodo: ast.stmt) -> list[str]:
    """Los nombres de nivel superior que define un nodo elegible; [] si no define ninguno."""
    if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [nodo.name]
    if isinstance(nodo, ast.Assign):
        return [t.id for t in nodo.targets if isinstance(t, ast.Name)]
    return []


def _donde_mas_esta(nombre: str, salvo: Path) -> list[str]:
    """En qué otros módulos del paquete aparece ese nombre en el nivel superior."""
    # Solo para el mensaje de error. Enmienda E2: los nombres no se pueden
    # renombrar pero SÍ mudar de archivo, así que el fallo más probable de este
    # cargador no es "ya no existe" sino "está en otro lado". Decirlo ahorra la
    # búsqueda a mano.
    encontrados = []
    for archivo in sorted(PAQUETE.glob("*.py")):
        if archivo == salvo:
            continue
        try:
            arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for nodo in arbol.body:
            if nombre in _nombres_definidos(nodo):
                encontrados.append(archivo.stem)
                break
    return encontrados


def cargar(nombres: list[str], modulo: str = "conversation") -> SimpleNamespace:
    """Devuelve un espacio de nombres con las funciones pedidas del módulo indicado."""
    origen = PAQUETE / f"{modulo}.py"
    arbol = ast.parse(origen.read_text(encoding="utf-8"))
    querido = set(nombres)

    elegidos = [
        nodo
        for nodo in arbol.body
        if querido.intersection(_nombres_definidos(nodo))
    ]

    faltan = querido - {
        nombre for nodo in elegidos for nombre in _nombres_definidos(nodo)
    }
    if faltan:
        # AssertionError y no un skip silencioso: una comprobación que se saltea
        # sola es peor que ninguna, porque deja el tablero en verde.
        pistas = []
        for nombre in sorted(faltan):
            otros = _donde_mas_esta(nombre, origen)
            pistas.append(f"{nombre} (está en: {', '.join(otros)})" if otros
                          else f"{nombre} (no está en ningún módulo)")
        raise AssertionError(
            f"No encontré {'; '.join(pistas)} en {origen}. "
            "¿Se renombraron, se movieron o dejaron de ser de nivel superior?"
        )

    espacio: dict = {
        # Las constantes que usan las funciones puras, con el mismo valor que
        # const.py. Se copian a mano a propósito: si alguien cambia el valor
        # real, conviene que la prueba lo note en vez de seguirlo sin chistar.
        # Es la unión de lo que necesita cualquier módulo; sobrar acá no cuesta
        # nada, faltar es un NameError en tiempo de ejecución.
        "CHARS_PER_TOKEN": 3.5,
        "CONF_CASA_ROUTER_ENABLED": "casa_router_enabled",
        "CONF_CASA_ROUTER_MODEL": "casa_router_model",
        "CONF_CASA_ROUTER_EFFORT": "casa_router_effort",
        "CONF_CASA_ROUTER_THRESHOLD": "casa_router_threshold",
        "CONF_RAZON_ROUTER_ENABLED": "razon_router_enabled",
        "CONF_RAZON_ROUTER_MODEL": "razon_router_model",
        "CONF_RAZON_ROUTER_EFFORT": "razon_router_effort",
        "CONF_RAZON_ROUTER_THRESHOLD": "razon_router_threshold",
        # Los dos enrutadores arrancan APAGADOS: actualizar la integración no
        # puede cambiarle el comportamiento a una instalación que anda.
        "RECOMMENDED_ROUTER_ENABLED": False,
        "RECOMMENDED_ROUTER_MODEL": "openai/gpt-oss-20b",
        "RECOMMENDED_ROUTER_THRESHOLD": 0.7,
        # Vacío es "no configurado", que no es lo mismo que un valor inválido y
        # no es lo mismo que el máximo: la distinción tiene que sobrevivir acá.
        "RECOMMENDED_ROUTER_EFFORT": "",
        "ROUTER_MAX_TOKENS": 150,
        "LOGGER": logging.getLogger("pruebas"),
        # El json_dumps de HA es compacto; el separador cambia cuántos caracteres
        # mide _coste_aproximado, así que la imitación tiene que serlo también.
        "json_dumps": lambda x: json.dumps(x, separators=(",", ":")),
        "json": json,
        "time": time,
        # Ver el punto 2 del docstring: las anotaciones se evalúan igual. Ninguna
        # función pura debería necesitar esto, y que esté no lo autoriza.
        "Any": object,
    }
    exec(compile(ast.Module(body=elegidos, type_ignores=[]), str(origen), "exec"),
         espacio)
    return SimpleNamespace(**{n: espacio[n] for n in nombres})
