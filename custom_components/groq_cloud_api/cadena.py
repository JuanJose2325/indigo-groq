"""Orden en que se prueban los modelos de la cadena de respaldo.

Frontera: módulo PURO. No conoce Home Assistant, ni el SDK de Groq, ni las
opciones de la entrada: recibe listas y diccionarios y devuelve una lista de
nombres de modelo. Nada de lo que hay acá hace red ni mira el reloj de pared,
solo `time.monotonic()`, que es lo único que `pruebas/cargar.py` tiene que
inyectar para poder leer este archivo por AST sin tener instalado nada.
"""

from __future__ import annotations

import time


def _candidatos(principal: str, cadena: list[str], ultimo_uso: dict[str, float],
                enfriamiento: float) -> list[str]:
    """Orden en que se van a probar los modelos: los fríos primero, los calientes al final.

    Los límites de Groq son POR MODELO: cada eslabón de la cadena aporta su
    propia ventana de 8000 tokens por minuto, así que rotar multiplica el cupo
    disponible en vez de repartirlo.

    Rota ANTES de que Groq rechace, no después: si el modelo preferido se usó
    hace menos de `enfriamiento` segundos es muy probable que su ventana siga
    ocupada, y esperar el 429 para recién ahí saltar costaría un viaje de red
    entero. Como por voz lo que manda es la latencia, se prefiere el segundo
    modelo antes que la espera.

    Los que están en enfriamiento no se descartan: se mandan al final, para que
    sigan sirviendo de última red si todos están calientes. Descartarlos
    dejaría al usuario sin respuesta justo en el peor momento.
    """
    orden: list[str] = []
    for m in [principal, *cadena]:
        # El `if m` tapa el campo de texto en blanco de la UI: pedirle a Groq un
        # modelo llamado "" es un 400. Y el `not in` evita gastar un viaje de red
        # repitiendo el principal cuando el usuario también lo puso en la cadena.
        if m and m not in orden:
            orden.append(m)
    ahora = time.monotonic()
    # El centinela -1e9 hace que un modelo nunca usado cuente siempre como frío,
    # y la comparación es >= y no >: con enfriamiento 0 todos tienen que quedar
    # fríos, o sea el orden configurado tal cual, aunque el reloj no se haya
    # movido ni un microsegundo entre dos turnos seguidos.
    frios = [m for m in orden if ahora - ultimo_uso.get(m, -1e9) >= enfriamiento]
    calientes = [m for m in orden if m not in frios]
    # Partición estable: dentro de cada grupo se conserva la preferencia que
    # configuró el usuario. Reordenar ahí sería decidir por él.
    return frios + calientes
