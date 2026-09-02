"""Todo lo que depende de la FAMILIA del modelo.

Frontera: acá adentro no se sabe nada de Home Assistant, del SDK de Groq ni de
la red. Entran un id de modelo y un esfuerzo, salen los parámetros de
razonamiento que hay que mandar. Todas las funciones son puras y de nivel
superior, porque el harness de `pruebas/` las carga por AST sin ejecutar los
imports: si alguna pasa a ser método, o a usar un nombre importado que el
cargador no inyecta, se pierden las comprobaciones de golpe y sin aviso.
"""

from __future__ import annotations

from .const import LOGGER

# Cada familia acepta un vocabulario distinto de esfuerzo de razonamiento, y
# la cadena de respaldo cruza de una a otra. Sin traducir, saltar de un Qwen
# (que usa "default"/"none") a un gpt-oss (que exige "low"/"medium"/"high")
# devuelve HTTP 400 y el usuario escucha el error crudo en voz alta.
_EQUIVALENCIAS = {
    "qwen": {"low": "default", "medium": "default", "high": "default",
             "default": "default", "none": "none"},
    # OJO: "default" en Qwen equivale a esfuerzo máximo, pero traducirlo a
    # "high" en gpt-oss es contraproducente acá. En los modelos de razonamiento
    # los tokens de pensamiento SALEN DE max_tokens, así que con un presupuesto
    # chico (500) el razonamiento alto se lo come entero y la respuesta llega
    # VACÍA: el usuario no escucha nada. "medium" deja lugar para contestar.
    "gpt-oss": {"default": "medium", "none": "low",
                "low": "low", "medium": "medium", "high": "high"},
}


def _familia_de(modelo: str) -> str | None:
    """La familia de razonamiento de un modelo, por PREFIJO del id; None si no razona."""
    # Por prefijo y nunca por igualdad: el bug original comparaba contra
    # "qwen/qwen3-32b" (deprecado por Groq en jun 2026), así que
    # qwen/qwen3.8-27b quedaba afuera y se iba sin reasoning_format="hidden".
    # El pensamiento volvía DENTRO de content y el TTS lo leía en voz alta.
    if modelo.startswith("qwen/"):
        return "qwen"
    if modelo.startswith("openai/gpt-oss"):
        return "gpt-oss"
    return None


def _esfuerzo_para(model: str, pedido: str | None) -> str | None:
    """Traduce un esfuerzo al vocabulario de la familia del modelo."""
    familia = _familia_de(model)
    if familia is None:
        # Sin familia conocida no hay a qué traducir: se devuelve tal cual y
        # quien llama decide si corresponde mandarlo (no corresponde).
        return pedido
    tabla = _EQUIVALENCIAS[familia]
    if pedido in tabla:
        return tabla[pedido]
    # Valor desconocido (modelo nuevo, config vieja): mejor el máximo de la
    # familia que un 400 que el usuario escucha como respuesta.
    LOGGER.warning(
        "Esfuerzo de razonamiento %r no válido para %s; uso el de por defecto",
        pedido, model,
    )
    return "default" if familia == "qwen" else "high"


def _vocabulario_de(modelo: str) -> list[str]:
    """Los esfuerzos que acepta la familia del modelo, de menor a mayor; [] si no razona."""
    familia = _familia_de(modelo)
    if familia is None:
        return []
    # Se DERIVA de la tabla en vez de tener su propia lista: mientras el
    # desplegable de la UI y la traducción salgan del mismo lado no pueden
    # divergir, que es lo que pasaba cuando eran dos constantes separadas.
    # La escala es el orden de menor a mayor esfuerzo; cada familia se queda
    # con los peldaños que de verdad usa.
    escala = ["none", "low", "medium", "high", "default"]
    usados = set(_EQUIVALENCIAS[familia].values())
    return [valor for valor in escala if valor in usados]


def _esfuerzo_inicial(modelo: str, guardado: str | None, barato: bool = False) -> str:
    """El valor preseleccionado del desplegable de esfuerzo de una fila de la UI."""
    vocabulario = _vocabulario_de(modelo)
    if not vocabulario:
        # Familia desconocida: la fila se dibuja como texto libre y no hay
        # nada que preseleccionar. Devolver un esfuerzo acá sería ofrecerle al
        # usuario un valor que su modelo va a rechazar con un 400.
        return ""
    if guardado in vocabulario:
        return guardado
    if not barato and not guardado:
        # Sin nada guardado, la fila del principal se queda EN BLANCO, que no es
        # lo mismo que "none" ni que un esfuerzo concreto: es "no configurado",
        # y ahí el modelo se llama con reasoning_format="hidden" y sin
        # reasoning_effort. Devolver un valor acá era inventarle una decisión a
        # quien nunca tocó el campo: en Qwen la rama de abajo da "default", que
        # es el MÁXIMO, así que abrir el formulario y guardar sin tocar nada
        # subía el razonamiento al tope y devolvía el silencio que este proyecto
        # viene arreglando. Un valor guardado que ya no es válido —modelo nuevo,
        # config vieja— sí sigue cayendo en la red de seguridad de abajo: ahí
        # hubo una elección del usuario que respetar como se pueda.
        return ""
    if barato:
        # Filas de enrutador: su max_tokens son 150, así que cualquier
        # pensamiento se come el JSON de tres claves antes de escribirlo y el
        # veredicto cae en la rama de fallo. El mínimo de la familia es lo
        # único que entra.
        return vocabulario[0]
    # Fila del principal: el recomendado sale de traducir "default" con la
    # misma tabla, que ya sabe que en gpt-oss eso es "medium" y no "high"
    # porque el pensamiento alto se come max_tokens antes de contestar.
    # En Qwen sí queda "default", que es su máximo, pero es que Qwen solo
    # tiene dos peldaños: el principal piensa o no piensa. Lo que se dejó de
    # hacer es preseleccionar el máximo por accidente, que era lo que pasaba
    # con reasoning_options[0] —["default", "none"][0]— en TODAS las filas y
    # ante cualquier valor guardado inválido. Las filas de enrutador, que son
    # las que no pueden pagarlo, se van por la rama de arriba.
    return _EQUIVALENCIAS[_familia_de(modelo)]["default"]


def _aplicar_razonamiento(kwargs: dict, model: str, esfuerzo: str | None) -> None:
    """Escribe en kwargs los parámetros de razonamiento del modelo, limpiando los del anterior."""
    # El mismo diccionario se reusa en cada salto de la cadena, así que lo del
    # candidato anterior tiene que desaparecer ANTES de cualquier salida
    # temprana: si no, el reasoning_format de un Qwen viaja pegado a la
    # petición de un llama que lo rechaza con 400.
    kwargs.pop("reasoning_format", None)
    kwargs.pop("reasoning_effort", None)
    kwargs.pop("include_reasoning", None)
    familia = _familia_de(model)
    if familia is None:
        # Familia desconocida: no se manda NADA de razonamiento. Antes acá se
        # colaba reasoning_effort a cualquier modelo, y los que no razonan
        # —llama-3.3-70b-versatile, por ejemplo— contestan HTTP 400 al
        # recibirlo. Eso importa justo ahora que la cadena se ensancha con
        # modelos de otras familias: un 400 en un eslabón de respaldo se
        # escucha como un error crudo en voz alta, que es peor que la
        # respuesta algo peor de un modelo sin pensamiento. Si más adelante
        # entra un razonador nuevo, se agrega su familia a _EQUIVALENCIAS y
        # vuelve a recibir los parámetros.
        return
    # Esfuerzo vacío o None NO es un valor inválido: es "no configurado", y son
    # casos opuestos. Antes los dos caían en la rama de valor desconocido de
    # _esfuerzo_para, que devuelve el MÁXIMO de la familia como red de
    # seguridad contra el 400 — y en Qwen el máximo se llama "default". El
    # `if esfuerzo` cortocircuita ANTES de esa red.
    traducido = _esfuerzo_para(model, esfuerzo) if esfuerzo else None
    if familia == "qwen":
        # reasoning_format va SIEMPRE, aunque no haya esfuerzo: sin él el
        # pensamiento vuelve dentro de content y el TTS lo lee en voz alta.
        # "Sin esfuerzo" no es "sin razonamiento": se le deja al modelo su
        # propio criterio, pero oculto.
        kwargs["reasoning_format"] = "hidden"
        if traducido:
            kwargs["reasoning_effort"] = traducido
    elif familia == "gpt-oss":
        # Va SOLO reasoning_format, nunca acompañado de include_reasoning:
        #
        #     400 - cannot specify both `include_reasoning` and `reasoning_format`
        #
        # Medido contra la API el 2 sep 2026. El código traía los dos porque
        # `include_reasoning=False` por su cuenta no alcanzaba —los gpt-oss
        # devolvían la cadena de pensamiento entera DENTRO de content, en inglés
        # y con la respuesta real pegada al final, y el TTS la leía— así que se
        # agregó `reasoning_format` sin sacar el otro. Con los dos puestos, Groq
        # rechaza la petición entera.
        #
        # Nunca se había visto porque esta rama jamás se ejecutó: la cadena de
        # la instalación era toda Qwen. Los enrutadores fueron el primer gpt-oss
        # que corrió de verdad, y fallaron 10 de 10 sin decir por qué. El que
        # oculta el pensamiento es `reasoning_format`, igual que en Qwen; el
        # otro sobra.
        kwargs["reasoning_format"] = "hidden"
        if traducido:
            kwargs["reasoning_effort"] = traducido


def _esfuerzo_del_candidato(esfuerzo: str | None, es_principal: bool) -> str | None:
    """El principal usa el esfuerzo decidido; los respaldos van sin pensamiento."""
    # Medido en la instalación real: el titular contesta pensando en 120-180
    # tokens, y el suplente quema los 1200 de max_tokens razonando y vuelve
    # VACÍO. Un suplente existe para contestar cuando el titular no puede;
    # gastarle el presupuesto en pensar lo vuelve inútil, y la diferencia no es
    # entre una respuesta mejor y una peor sino entre una respuesta y ninguna.
    # Esta política era el campo reasoning_effort_chain, que ya no existe.
    # Quien llama tiene que calcular es_principal por IDENTIDAD del modelo
    # (nombre == principal) y nunca por su posición en la lista: el principal
    # reaparece más abajo en la rotación cuando está en enfriamiento, y ahí
    # sigue siendo el titular aunque entre último.
    if es_principal:
        return esfuerzo
    return "none"
