#!/usr/bin/env python3
"""Pruebas de qué parámetros de razonamiento se mandan (`_aplicar_razonamiento`).

`_esfuerzo_para` decide QUÉ valor tendría el esfuerzo; esta función decide si se
manda y con qué compañía. La distinción importa: el valor puede ser válido y aun
así no corresponder mandarlo.

Lo que se está protegiendo acá es un 400 que se escucharía en voz alta. La cadena
de respaldo se ensancha con modelos de otras familias, y los que NO razonan
—`llama-3.3-70b-versatile`, por ejemplo— rechazan `reasoning_effort`. Antes, una
familia desconocida caía en un `elif` que se lo mandaba igual.

Y el otro invariante: la función se llama una vez por candidato sobre el MISMO
diccionario, así que tiene que limpiar lo que dejó el modelo anterior. Si no, el
`reasoning_format` de un Qwen viaja pegado a la petición del siguiente.
"""

from cargar import cargar
from runner import comprobar, resumen

M = cargar(["_EQUIVALENCIAS", "_esfuerzo_para", "_aplicar_razonamiento"])

QWEN = "qwen/qwen3.6-27b"
OSS = "openai/gpt-oss-120b"
LLAMA = "llama-3.3-70b-versatile"

CON_RAZONAMIENTO = {"supports_reasoning": True, "reasoning_effort": "default"}
SIN_RAZONAMIENTO = {"supports_reasoning": False, "reasoning_effort": "default"}

CLAVES = ("reasoning_format", "reasoning_effort", "include_reasoning")


def aplicar(model, options, forzar=None, previo=None, principal=True):
    kwargs = dict(previo or {})
    M._aplicar_razonamiento(kwargs, model, options, forzar=forzar,
                            es_principal=principal)
    return {k: v for k, v in kwargs.items() if k in CLAVES}


# 1. EL fallo que se está previniendo: un modelo que no razona no debe recibir
#    NADA de razonamiento, por más que el esfuerzo configurado sea válido.
comprobar(
    "un modelo de familia desconocida no recibe parámetros de razonamiento",
    aplicar(LLAMA, CON_RAZONAMIENTO) == {},
)

# 2. Los que sí razonan los siguen recibiendo, con el formato oculto: sin eso el
#    pensamiento vuelve dentro del texto y el TTS lo lee en voz alta.
comprobar(
    "Qwen recibe el formato oculto",
    aplicar(QWEN, CON_RAZONAMIENTO).get("reasoning_format") == "hidden",
)
comprobar(
    "gpt-oss recibe el formato oculto",
    aplicar(OSS, CON_RAZONAMIENTO).get("reasoning_format") == "hidden",
)
comprobar(
    "gpt-oss recibe un esfuerzo de su propio vocabulario",
    aplicar(OSS, CON_RAZONAMIENTO).get("reasoning_effort") in {"low", "medium", "high"},
)

# 3. `forzar` es lo que usa el reintento cuando el pensamiento se comió el
#    presupuesto entero. Tiene que pisar lo configurado.
comprobar(
    "forzar 'none' pisa el esfuerzo configurado en Qwen",
    aplicar(QWEN, CON_RAZONAMIENTO, forzar="none").get("reasoning_effort") == "none",
)
comprobar(
    "forzar 'none' también aplica a gpt-oss, traducido a su mínimo",
    aplicar(OSS, CON_RAZONAMIENTO, forzar="none").get("reasoning_effort") == "low",
)
comprobar(
    "forzar sobre un modelo que no razona sigue sin mandar nada",
    aplicar(LLAMA, CON_RAZONAMIENTO, forzar="none") == {},
)

# 4. Con el razonamiento apagado en la configuración no se manda nada, ni
#    siquiera a un modelo que sí sabe razonar.
comprobar(
    "supports_reasoning en falso no manda nada a Qwen",
    aplicar(QWEN, SIN_RAZONAMIENTO) == {},
)

# 5. Campo de esfuerzo vacío = "no configurado", NO "valor inválido". Son casos
#    opuestos y durante un tiempo se trataron igual: el vacío caía en la red de
#    seguridad contra el 400, que devuelve el máximo de la familia, y en Qwen el
#    máximo se llama "default". Vaciar el campo para quitarle pensamiento al
#    modelo se lo subía al tope. Se vio en la instalación real, en la línea
#    «Esfuerzo de razonamiento None no válido ...; uso el de por defecto».
for vacio in (None, ""):
    opciones = {"supports_reasoning": True, "reasoning_effort": vacio}
    comprobar(
        f"esfuerzo {vacio!r} no se convierte en el máximo de Qwen",
        "reasoning_effort" not in aplicar(QWEN, opciones),
    )
    comprobar(
        f"esfuerzo {vacio!r} conserva igual el formato oculto en Qwen",
        aplicar(QWEN, opciones).get("reasoning_format") == "hidden",
    )
    comprobar(
        f"esfuerzo {vacio!r} tampoco inventa un esfuerzo para gpt-oss",
        "reasoning_effort" not in aplicar(OSS, opciones),
    )

# 6. Pero un valor de verdad desconocido —modelo nuevo, config vieja— sí tiene
#    que seguir cayendo en la red de seguridad: mejor el máximo que un 400.
comprobar(
    "un valor desconocido sigue traduciéndose a algo válido",
    aplicar(QWEN, {"supports_reasoning": True,
                   "reasoning_effort": "altísimo"}).get("reasoning_effort")
    in {"default", "none"},
)

# 5. Higiene entre candidatos: el mismo diccionario se reusa en cada salto de la
#    cadena, así que lo del modelo anterior tiene que desaparecer. Este es el
#    caso exacto del salto Qwen -> llama.
sucio = {"reasoning_format": "hidden", "reasoning_effort": "default",
         "include_reasoning": False}
comprobar(
    "limpia los restos del candidato anterior al pasar a uno que no razona",
    aplicar(LLAMA, CON_RAZONAMIENTO, previo=sucio) == {},
)
comprobar(
    "limpia include_reasoning al pasar de gpt-oss a Qwen",
    "include_reasoning" not in aplicar(QWEN, CON_RAZONAMIENTO, previo=sucio),
)

# 6. Lo que no es de razonamiento no se toca: la función comparte diccionario
#    con el resto de la petición.
kwargs = {"model": LLAMA, "max_tokens": 1200, "reasoning_effort": "default"}
M._aplicar_razonamiento(kwargs, LLAMA, CON_RAZONAMIENTO)
comprobar(
    "no pisa el resto de la petición",
    kwargs == {"model": LLAMA, "max_tokens": 1200},
)

# 7. Política separada para principal y respaldos. El titular se eligió porque
#    contesta bien y puede permitirse pensar; un suplente entra justo cuando el
#    cupo escasea, y gastarlo razonando es la diferencia entre una respuesta
#    peor y NINGUNA respuesta. Medido: el principal contestaba pensando en
#    120-180 tokens y el respaldo quemaba los 1200 sin llegar a hablar.
DOS_POLITICAS = {"supports_reasoning": True,
                 "reasoning_effort": "default",
                 "reasoning_effort_chain": "none"}

comprobar(
    "el principal usa su propio campo",
    aplicar(QWEN, DOS_POLITICAS).get("reasoning_effort") == "default",
)
comprobar(
    "un respaldo usa el campo de la cadena",
    aplicar(QWEN, DOS_POLITICAS, principal=False).get("reasoning_effort") == "none",
)
comprobar(
    "el respaldo también se traduce a la familia que toque",
    aplicar(OSS, DOS_POLITICAS, principal=False).get("reasoning_effort") == "low",
)

# 8. Sin el campo nuevo configurado —una instalación que viene de una versión
#    anterior— el respaldo NO hereda el esfuerzo del principal: hereda el
#    recomendado, que es no pensar. Heredar sería reproducir el fallo original.
SOLO_VIEJO = {"supports_reasoning": True, "reasoning_effort": "default"}
comprobar(
    "sin el campo nuevo el respaldo no hereda el esfuerzo del principal",
    aplicar(QWEN, SOLO_VIEJO, principal=False).get("reasoning_effort") == "none",
)
comprobar(
    "pero el principal sigue respetando el suyo",
    aplicar(QWEN, SOLO_VIEJO).get("reasoning_effort") == "default",
)

# 9. El reintento forzado pisa las DOS políticas: si volvió vacío, no importa
#    de qué campo salía el esfuerzo.
comprobar(
    "forzar pisa también la política del respaldo",
    aplicar(QWEN, {"supports_reasoning": True,
                   "reasoning_effort_chain": "default"},
            forzar="none", principal=False).get("reasoning_effort") == "none",
)

resumen("parámetros de razonamiento por familia de modelo")
