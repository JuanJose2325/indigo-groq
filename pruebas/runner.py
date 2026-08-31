"""Lo mínimo para correr las pruebas sin instalar pytest."""

from __future__ import annotations

import sys

_fallos: list[str] = []
_total = 0


def comprobar(descripcion: str, condicion: bool) -> None:
    """Registra una comprobación e imprime el resultado."""
    global _total
    _total += 1
    if condicion:
        print(f"  ok   {descripcion}")
    else:
        print(f"  FALLA {descripcion}")
        _fallos.append(descripcion)


def resumen(titulo: str) -> None:
    """Imprime el recuento y sale con código distinto de cero si algo falló."""
    if _fallos:
        print(f"\n{titulo}: {len(_fallos)} de {_total} fallaron")
        sys.exit(1)
    print(f"\n{titulo}: {_total} comprobaciones, todo bien")
