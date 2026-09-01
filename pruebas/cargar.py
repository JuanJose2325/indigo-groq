"""Carga las funciones puras de conversation.py sin importar Home Assistant.

`conversation.py` importa `homeassistant` y el SDK de `groq` en la primera
línea, así que un `import` normal desde una máquina cualquiera falla. Pero las
funciones que importa probar (recorte de historial, rotación de la cadena,
traducción del esfuerzo de razonamiento) no tocan ninguna de esas dos cosas:
son funciones sobre diccionarios y listas.

Esto lee el archivo, se queda con los nodos del AST que hacen falta y los
ejecuta en un espacio de nombres con las pocas dependencias necesarias
sustituidas por versiones mínimas. Así las pruebas corren contra el código
REAL — si alguien lo edita, se rompen — sin necesidad de instalar HA.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

ORIGEN = (
    Path(__file__).resolve().parent.parent
    / "custom_components"
    / "groq_cloud_api"
    / "conversation.py"
)


def cargar(nombres: list[str]) -> SimpleNamespace:
    """Devuelve un espacio de nombres con las funciones pedidas ya definidas."""
    arbol = ast.parse(ORIGEN.read_text(encoding="utf-8"))
    querido = set(nombres)

    elegidos = [
        nodo
        for nodo in arbol.body
        if (isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
            and nodo.name in querido)
        or (isinstance(nodo, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in querido
                    for t in nodo.targets))
    ]

    faltan = querido - {
        nodo.name if hasattr(nodo, "name") else nodo.targets[0].id
        for nodo in elegidos
    }
    if faltan:
        raise AssertionError(
            f"No encontré {sorted(faltan)} en {ORIGEN}. "
            "¿Se renombraron o se movieron?"
        )

    espacio: dict = {
        # Las constantes que usan las funciones puras, con el mismo valor que
        # const.py. Se copian a mano a propósito: si alguien cambia el valor
        # real, conviene que la prueba lo note.
        "CHARS_PER_TOKEN": 3.5,
        "CONF_REASONING_EFFORT": "reasoning_effort",
        "CONF_REASONING_EFFORT_CHAIN": "reasoning_effort_chain",
        "RECOMMENDED_REASONING_EFFORT_CHAIN": "none",
        "CONF_SUPPORTS_REASONING": "supports_reasoning",
        "LOGGER": logging.getLogger("pruebas"),
        "json_dumps": lambda x: json.dumps(x, separators=(",", ":")),
        "time": time,
        "Any": object,
    }
    exec(compile(ast.Module(body=elegidos, type_ignores=[]), str(ORIGEN), "exec"),
         espacio)
    return SimpleNamespace(**{n: espacio[n] for n in nombres})
