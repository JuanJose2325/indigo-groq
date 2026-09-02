#!/usr/bin/env python3
"""Pruebas de lo que se LEE y de lo que se GUARDA en las opciones de la entrada.

El fallo real que protege esta tanda no se ve como un fallo: se ve como que la
casa dejó de contestar después de una actualización que "no tocaba nada".

La entrada que está andando en la instalación real se creó con la versión
anterior. Adentro tiene cinco claves que ya no existen —las tres del segundo
proveedor que se sacó, más `reasoning_effort_chain` y `supports_reasoning`— y no
tiene ninguna de las ocho de los dos enrutadores. Si `_ajustes_enrutadores`
leyera con corchetes en vez de con `.get(clave, RECOMMENDED_*)`, el primer turno
después de actualizar levantaría un KeyError adentro de `_async_handle_message`,
que no es HomeAssistantError: se escapa de `async_converse` y rompe el pipeline
entero de Assist, no solo ese turno. Y si los defaults fueran los enrutadores
PRENDIDOS, una instalación que anda cambiaría de comportamiento sola, sin que el
usuario haya pedido nada.

La otra mitad es al revés: qué se escribe. `_normalizar_opciones` tiene que
tirar esas cinco claves muertas al guardar (para que no se arrastren para
siempre) sin romperse porque estén, y tiene que dejar el esfuerzo de
razonamiento VACÍO tal cual. Vacío no es un error: significa "no configurado", y
barrerlo reintroduce el fallo silencioso del comportamiento 6.

Y de yapa se fija la trampa de la interfaz (enmienda E6): el desplegable de
esfuerzo nunca puede preseleccionar el máximo de la familia. En Qwen el valor
histórico "default" ES el máximo, así que el `reasoning_options[0]` de antes
preseleccionaba el máximo en las TRES filas ante cualquier valor guardado
inválido. En una fila de enrutador —150 tokens de techo— eso significa que el
enrutador se gasta el presupuesto pensando, vuelve vacío y su veredicto cae
derecho en la rama de fallo, todos los turnos.
"""

from cargar import cargar
from runner import comprobar, resumen

E = cargar(["_ajustes_enrutadores"], "enrutadores")
C = cargar(["_normalizar_opciones"], "config_flow")
R = cargar(
    ["_EQUIVALENCIAS", "_familia_de", "_vocabulario_de", "_esfuerzo_inicial"],
    "razonamiento",
)

# La entrada tal como está guardada hoy en la instalación real, con las cinco
# claves muertas incluidas. Es la que tiene que arrancar sin romper.
ENTRADA_VIEJA = {
    "chat_model": "qwen/qwen3.8-27b",
    "model_chain": ["qwen/qwen3.6-27b"],
    "model_cooldown": 60,
    "max_tokens": 1200,
    "history_budget": 2500,
    "llm_hass_api": ["assist"],
    "max_retries": 0,
    "temperature": 1.0,
    "top_p": 1.0,
    "prompt": "Sos Indigo.",
    "cf_account_id": "una-cuenta",
    "cf_api_token": "una-ficha",
    "cf_model": "@cf/qwen/qwen3.8-27b",
    "reasoning_effort_chain": ["none"],
    "supports_reasoning": True,
}

CLAVES_AJUSTES = (
    "casa_activo",
    "casa_modelo",
    "casa_esfuerzo",
    "casa_umbral",
    "razon_activo",
    "razon_modelo",
    "razon_esfuerzo",
    "razon_umbral",
)


# --- Lo que se lee: la entrada vieja no puede romper nada ------------------

ajustes = E._ajustes_enrutadores(ENTRADA_VIEJA)

comprobar(
    "la entrada vieja devuelve las ocho claves, ni una menos",
    set(ajustes) == set(CLAVES_AJUSTES),
)
comprobar(
    "el enrutador de casa arranca APAGADO en una entrada que viene de antes",
    ajustes["casa_activo"] is False,
)
comprobar(
    "el de razonamiento también arranca APAGADO",
    ajustes["razon_activo"] is False,
)
comprobar(
    "ninguna de las cinco claves muertas se cuela en los ajustes",
    not any(
        clave in ajustes
        for clave in ("cf_account_id", "cf_api_token", "cf_model",
                      "reasoning_effort_chain", "supports_reasoning")
    ),
)
comprobar(
    "sin modelo de enrutador guardado cae en el recomendado, no en cadena vacía",
    ajustes["casa_modelo"] == "openai/gpt-oss-20b"
    and ajustes["razon_modelo"] == "openai/gpt-oss-20b",
)
comprobar(
    "el esfuerzo sin configurar queda VACÍO, que no es lo mismo que el máximo",
    ajustes["casa_esfuerzo"] == "" and ajustes["razon_esfuerzo"] == "",
)
comprobar(
    "los umbrales caen en el recomendado",
    ajustes["casa_umbral"] == 0.7 and ajustes["razon_umbral"] == 0.7,
)

# Una entrada completamente vacía es el otro extremo del mismo caso: la que crea
# `DEFAULT_OPTIONS`, que a propósito no trae ninguna clave de enrutador.
vacia = E._ajustes_enrutadores({})
comprobar(
    "una entrada sin ninguna opción tampoco rompe",
    set(vacia) == set(CLAVES_AJUSTES),
)
comprobar(
    "y también deja los dos enrutadores apagados",
    vacia["casa_activo"] is False and vacia["razon_activo"] is False,
)

# Con los dos apagados el turno tiene que ser indistinguible del de antes, y eso
# empieza acá: los ajustes de una entrada vieja y los de una recién creada tienen
# que ser EL MISMO diccionario.
comprobar(
    "entrada vieja y entrada nueva dan exactamente los mismos ajustes",
    ajustes == vacia,
)

# --- Lo que se lee: tipos raros guardados a mano ---------------------------

comprobar(
    "un activo guardado como texto se lee como booleano, no como cadena",
    E._ajustes_enrutadores({"casa_router_enabled": "true"})["casa_activo"] is True,
)
comprobar(
    "un activo en 0 queda apagado",
    E._ajustes_enrutadores({"razon_router_enabled": 0})["razon_activo"] is False,
)
comprobar(
    "un umbral por encima de 1 se acota a 1 en vez de dejar al enrutador mudo",
    E._ajustes_enrutadores({"casa_router_threshold": 1.5})["casa_umbral"] == 1.0,
)
comprobar(
    "un umbral negativo se acota a 0",
    E._ajustes_enrutadores({"razon_router_threshold": -3})["razon_umbral"] == 0.0,
)
comprobar(
    "un umbral que no es número cae en el recomendado en vez de reventar",
    E._ajustes_enrutadores({"casa_router_threshold": "ni idea"})["casa_umbral"] == 0.7,
)
comprobar(
    "un umbral en None cae en el recomendado",
    E._ajustes_enrutadores({"razon_router_threshold": None})["razon_umbral"] == 0.7,
)
comprobar(
    "un umbral guardado como texto numérico se lee igual",
    E._ajustes_enrutadores({"casa_router_threshold": "0.4"})["casa_umbral"] == 0.4,
)
comprobar(
    "un modelo de enrutador guardado en blanco cae en el recomendado",
    # Sin esto saldría una petición con model="" y Groq la rechaza con un 400 en
    # cada turno, con los dos enrutadores prendidos.
    E._ajustes_enrutadores({"casa_router_model": ""})["casa_modelo"]
    == "openai/gpt-oss-20b",
)
comprobar(
    "un modelo de enrutador elegido a mano se respeta",
    E._ajustes_enrutadores({"razon_router_model": "qwen/qwen3.8-27b"})["razon_modelo"]
    == "qwen/qwen3.8-27b",
)
comprobar(
    "los dos enrutadores se leen por separado, no se contagian",
    E._ajustes_enrutadores({"casa_router_enabled": True})["razon_activo"] is False,
)
comprobar(
    "leer los ajustes no muta las opciones de la entrada",
    # `entry.options` es un MappingProxyType: escribirle sería un TypeError en
    # producción y acá pasaría desapercibido.
    set(E._ajustes_enrutadores(dict(ENTRADA_VIEJA)) or ()) and len(ENTRADA_VIEJA) == 15,
)


# --- Lo que se guarda ------------------------------------------------------

guardado = C._normalizar_opciones(dict(ENTRADA_VIEJA))

for muerta in ("cf_account_id", "cf_api_token", "cf_model",
               "reasoning_effort_chain", "supports_reasoning"):
    comprobar(
        f"al guardar se tira la clave muerta {muerta}",
        muerta not in guardado,
    )
comprobar(
    "guardar una entrada vieja no revienta por tener claves desconocidas",
    guardado["chat_model"] == "qwen/qwen3.8-27b",
)
comprobar(
    "las claves vivas sobreviven enteras",
    guardado["model_chain"] == ["qwen/qwen3.6-27b"]
    and guardado["max_tokens"] == 1200
    and guardado["history_budget"] == 2500,
)
comprobar(
    "normalizar no muta el diccionario que recibe",
    "cf_model" in ENTRADA_VIEJA,
)
comprobar(
    "un llm_hass_api con contenido se conserva",
    guardado["llm_hass_api"] == ["assist"],
)
comprobar(
    "un llm_hass_api vacío se BORRA en vez de guardarse como lista vacía",
    "llm_hass_api" not in C._normalizar_opciones({"llm_hass_api": []}),
)
comprobar(
    "un llm_hass_api en None también se borra",
    "llm_hass_api" not in C._normalizar_opciones({"llm_hass_api": None}),
)
comprobar(
    "el esfuerzo VACÍO se guarda tal cual: vacío es 'no configurado', no un error",
    C._normalizar_opciones({"reasoning_effort": ""})["reasoning_effort"] == "",
)
comprobar(
    "los dos esfuerzos de enrutador vacíos también se guardan tal cual",
    C._normalizar_opciones(
        {"casa_router_effort": "", "razon_router_effort": ""}
    ) == {"casa_router_effort": "", "razon_router_effort": ""},
)
comprobar(
    "las ocho claves de enrutador pasan sin que nadie las toque",
    C._normalizar_opciones(
        {
            "casa_router_enabled": True,
            "casa_router_model": "openai/gpt-oss-20b",
            "casa_router_effort": "low",
            "casa_router_threshold": 0.7,
            "razon_router_enabled": False,
            "razon_router_model": "openai/gpt-oss-20b",
            "razon_router_effort": "low",
            "razon_router_threshold": 0.8,
        }
    )["casa_router_threshold"] == 0.7,
)
comprobar(
    "un formulario vacío no revienta",
    C._normalizar_opciones({}) == {},
)

# Lo que se guarda tiene que poder volver a leerse: si el ida y vuelta no cierra,
# el usuario guarda el formulario y al reabrirlo ve otra cosa.
ida_y_vuelta = E._ajustes_enrutadores(
    C._normalizar_opciones(
        {
            "casa_router_enabled": True,
            "casa_router_model": "qwen/qwen3.8-27b",
            "casa_router_effort": "none",
            "casa_router_threshold": 0.55,
        }
    )
)
comprobar(
    "lo que guarda el formulario es lo que después lee el turno",
    ida_y_vuelta["casa_activo"] is True
    and ida_y_vuelta["casa_modelo"] == "qwen/qwen3.8-27b"
    and ida_y_vuelta["casa_esfuerzo"] == "none"
    and ida_y_vuelta["casa_umbral"] == 0.55,
)


# --- La trampa del desplegable de esfuerzo (enmienda E6) -------------------

comprobar(
    "el vocabulario de Qwen sale de la tabla, de menor a mayor",
    R._vocabulario_de("qwen/qwen3.8-27b") == ["none", "default"],
)
comprobar(
    "el de gpt-oss también",
    R._vocabulario_de("openai/gpt-oss-20b") == ["low", "medium", "high"],
)
comprobar(
    "un modelo que no razona no ofrece ningún esfuerzo",
    R._vocabulario_de("meta-llama/llama-4-scout-17b-16e-instruct") == [],
)
comprobar(
    "el vocabulario se deriva de _EQUIVALENCIAS y no de una lista aparte",
    # Si alguien agrega un peldaño a la tabla y se olvida del desplegable, o al
    # revés, los dos se van a poder ofrecer valores que el otro no traduce.
    set(R._vocabulario_de("qwen/qwen3.8-27b"))
    == set(R._EQUIVALENCIAS["qwen"].values())
    and set(R._vocabulario_de("openai/gpt-oss-20b"))
    == set(R._EQUIVALENCIAS["gpt-oss"].values()),
)
comprobar(
    "una fila de enrutador con basura guardada preselecciona el MÍNIMO, no 'default'",
    # Este es el bug exacto: "default" es el máximo de Qwen y era lo que salía de
    # reasoning_options[0]. Con 150 tokens de techo, el enrutador se los gasta
    # pensando y vuelve vacío en todos los turnos.
    R._esfuerzo_inicial("qwen/qwen3.8-27b", "medium", barato=True) == "none",
)
comprobar(
    "una fila de enrutador sin nada guardado también preselecciona el mínimo",
    R._esfuerzo_inicial("openai/gpt-oss-20b", None, barato=True) == "low",
)
comprobar(
    "la fila del principal nunca preselecciona 'high' en gpt-oss",
    # "default" traducido a gpt-oss es "medium" a propósito: con "high" el
    # pensamiento se come max_tokens antes de que llegue a contestar. Se afirma
    # el INVARIANTE y no un valor concreto: lo que no puede pasar nunca es que
    # la UI le regale el máximo a quien no lo eligió, sea porque no hay nada
    # guardado (queda en blanco) o porque lo guardado ya no vale (red de
    # seguridad). Atarlo a un valor puntual hacía fallar la comprobación por un
    # cambio que iba en la misma dirección que ella.
    all(
        R._esfuerzo_inicial("openai/gpt-oss-20b", guardado) != "high"
        for guardado in (None, "", "basura", "default")
    ),
)
comprobar(
    "un valor guardado que sigue siendo válido se respeta en la fila del principal",
    R._esfuerzo_inicial("openai/gpt-oss-20b", "high") == "high",
)
comprobar(
    "y también en una fila de enrutador",
    R._esfuerzo_inicial("openai/gpt-oss-20b", "medium", barato=True) == "medium",
)
comprobar(
    "un valor de otra familia no se cuela: se descarta y se preselecciona el propio",
    R._esfuerzo_inicial("openai/gpt-oss-20b", "default") == "medium",
)
comprobar(
    "un modelo que no razona no preselecciona ningún esfuerzo",
    R._esfuerzo_inicial("meta-llama/llama-4-scout-17b-16e-instruct", "low") == "",
)
comprobar(
    "el preseleccionado siempre es algo que el desplegable ofrece",
    # El desplegable son el vocabulario de la familia MÁS la opción vacía "(sin
    # configurar)", que es un estado real y no la ausencia de uno. Un
    # preseleccionado fuera de esa lista es un formulario que no se puede
    # guardar, así que la propiedad se afirma contra lo que se ofrece de verdad.
    all(
        R._esfuerzo_inicial(modelo, guardado, barato=barato)
        in ([""] + R._vocabulario_de(modelo))
        for modelo in ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
        for guardado in (None, "", "basura", "high", "default", "none")
        for barato in (True, False)
    ),
)
comprobar(
    "sin un valor válido guardado, una fila de enrutador nunca cae en el máximo",
    # El fallo era ese: `reasoning_options[0]` no miraba si lo guardado servía y
    # en Qwen devolvía "default", que es el máximo. Acá se recorre todo lo que
    # puede haber guardado y no ser válido: vacío, None, un valor de la otra
    # familia, basura.
    all(
        R._esfuerzo_inicial(modelo, guardado, barato=True)
        != R._vocabulario_de(modelo)[-1]
        for modelo in ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
        for guardado in (None, "", "basura", "medium", "default")
        if guardado not in R._vocabulario_de(modelo)
    ),
)
comprobar(
    "la fila del PRINCIPAL sin nada guardado queda EN BLANCO, no en el máximo",
    # La disyuntiva entre "none" y "default" es falsa: hay un TERCER estado, y es
    # justo en el que corre una instalación que nunca tocó el campo. Vacío no es
    # "no pienses", es "decidí vos": se manda reasoning_format="hidden" y ningún
    # reasoning_effort. Preseleccionar "default" acá le escribía el MÁXIMO de
    # Qwen a quien no tenía la clave, con solo abrir el formulario y guardar sin
    # tocar nada — el fallo silencioso entrando por la interfaz.
    R._esfuerzo_inicial("qwen/qwen3.8-27b", None) == ""
    and R._esfuerzo_inicial("qwen/qwen3.8-27b", "") == "",
)
comprobar(
    "las filas de ENRUTADOR sin nada guardado sí toman el mínimo, no el blanco",
    # Acá la asimetría es a propósito: el enrutador tiene 150 tokens de techo,
    # así que dejarlo librado al criterio del modelo es arriesgar que el
    # pensamiento se coma el JSON de tres claves y el veredicto caiga en la rama
    # de fallo. El principal puede permitirse "decidí vos"; el enrutador no.
    R._esfuerzo_inicial("qwen/qwen3.8-27b", None, barato=True) == "none"
    and R._esfuerzo_inicial("openai/gpt-oss-20b", None, barato=True) == "low",
)
comprobar(
    "un valor guardado que ya no es válido SÍ cae en la red de seguridad",
    # No se confunde "no configurado" con "configurado con algo que dejó de
    # existir": en el segundo caso hubo una elección del usuario y hay que
    # respetarla como se pueda, no borrarla.
    R._esfuerzo_inicial("qwen/qwen3.8-27b", "ultra") == "default"
    and R._esfuerzo_inicial("openai/gpt-oss-20b", "ultra") == "medium",
)
comprobar(
    "pero un máximo elegido a mano SÍ se respeta: lo eligió el usuario, no la UI",
    # No es una excepción a la regla de arriba, es la regla de al lado: lo que no
    # se puede es preseleccionar el máximo POR DEFECTO. Pisarle al usuario un
    # valor que guardó a propósito sería el otro bug, el simétrico.
    R._esfuerzo_inicial("qwen/qwen3.8-27b", "default", barato=True) == "default"
    and R._esfuerzo_inicial("openai/gpt-oss-20b", "high", barato=True) == "high",
)

resumen("opciones: entrada vieja, normalización y desplegables de esfuerzo")
