#!/usr/bin/env python3
"""Pruebas de las transformaciones del historial y de la rotación de la cadena.

Los tres fallos reales que protege este archivo, todos vistos en la instalación
que anda:

1. El límite de Groq cuenta ENTRADA + SALIDA por minuto y el historial entero se
   reenvía en cada turno, así que sin recorte la conversación pasa los 8000 TPM y
   a partir de ahí falla SIEMPRE, no de a ratos. Por eso parecía "arreglarse
   sola" al reiniciar: reiniciar es cuando Home Assistant tira el
   conversation_id. (`_coste_aproximado`, `_recortar_historial`.)

2. Cuando el turno va sin herramientas —que es el camino que ahorra los 4150
   tokens del bloque de Assist sobre los 8000 disponibles— no alcanza con no
   mandar `tools`: un `role="tool"` suelto, o un assistant con `tool_calls` que
   apuntan a herramientas que ya no se declararon, es un 400 seco. Y la limpieza
   NO puede mutar lo que recibe, porque esos dicts salen de `chat_log.content`,
   la lista que HA persiste, cuyos elementos son dataclasses frozen.
   (`_sin_herramientas`, y el orden de la nota 2.18.)

3. Los límites de Groq son POR MODELO, así que cada eslabón de la cadena aporta
   su propia ventana de 8000 TPM. Rotar tiene que pasar ANTES del 429: esperar
   el rechazo cuesta un viaje de red entero y por voz lo que manda es la
   latencia. Y un modelo en enfriamiento no se descarta, se posterga: si se
   descartara, con todos calientes no quedaría ninguno. (`_candidatos`.)

Se ejecuta con `python3 pruebas/test_historial.py` — sin pytest y sin tener
Home Assistant instalado.
"""

import time

from cargar import cargar
from runner import comprobar, resumen

A = cargar(["_aporta_algo"], "mensajes")

H = cargar(
    ["_coste_aproximado", "_recortar_historial", "_sin_herramientas",
     "_ultimos_turnos"],
    "historial",
)
# Módulo aparte: las funciones puras comparten un espacio de nombres POR ARCHIVO
# (punto 3 del docstring de cargar.py), así que la cadena se pide por separado.
C = cargar(["_candidatos"], "cadena")


def usuario(texto):
    return {"role": "user", "content": texto}


def asistente(texto, tool_calls=None):
    m = {"role": "assistant", "content": texto}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def herramienta(texto):
    return {"role": "tool", "tool_call_id": "x", "content": texto}


LLAMADA = {"id": "a", "type": "function",
           "function": {"name": "luz", "arguments": "{}"}}

SISTEMA = {"role": "system", "content": "s" * 350}  # ~104 tokens a 3.5 char/token


# ---------------------------------------------------------------------------
# A. Recorte del historial: _coste_aproximado y _recortar_historial.
# ---------------------------------------------------------------------------

# 1. Con presupuesto de sobra no se toca nada. Recortar de más es tan malo como
#    no recortar: le saca contexto al modelo sin ganar un solo token.
historial = [SISTEMA, usuario("hola"), asistente("qué tal"), usuario("bien")]
comprobar(
    "presupuesto amplio deja el historial intacto",
    H._recortar_historial(historial, 100_000) == historial,
)

# 2. El prompt de sistema NUNCA se descarta, aunque el resto no entre. Y se
#    comprueba por IDENTIDAD, no por igualdad: el objeto tiene que ser el mismo.
recortado = H._recortar_historial([SISTEMA, usuario("x" * 3500)], 150)
comprobar(
    "el sistema sobrevive al recorte",
    recortado[0] is SISTEMA and len(recortado) == 1,
)

# 3. Se conservan los turnos MÁS RECIENTES, no los primeros. Con voz, lo que el
#    usuario acaba de decir es lo único que no se puede perder.
viejos = [usuario(f"pregunta {i}" + "z" * 300) for i in range(10)]
recortado = H._recortar_historial([SISTEMA, *viejos], 400)
comprobar(
    "se queda con la cola, no con la cabeza",
    recortado[-1] is viejos[-1] and viejos[0] not in recortado,
)

# 4. Si ni el prompt de sistema entra, devuelve todo sin tocar: recortar ya no
#    arregla nada y mandar la conversación vacía sería peor que dejarla pasar.
entrada = [SISTEMA, usuario("hola")]
comprobar(
    "sistema más grande que el presupuesto: no recorta",
    H._recortar_historial(entrada, 10) == entrada,
)

# 5. La ventana no puede empezar por un resultado de herramienta huérfano ni por
#    un assistant cuyas tool_calls quedaron afuera: la API contesta 400. El corte
#    se lleva hasta el primer mensaje de usuario, que siempre es corte limpio.
conv = [
    SISTEMA,
    usuario("apagá la luz" + "y" * 400),
    asistente(None, [LLAMADA]),
    herramienta("ok"),
    asistente("Listo"),
]
recortado = H._recortar_historial(conv, 250)
sin_sistema = [m for m in recortado if m["role"] != "system"]
comprobar(
    "no deja tool results huérfanos",
    not sin_sistema or sin_sistema[0]["role"] == "user",
)

# 6. El coste cuenta también los argumentos de las tool_calls, que no viven en
#    `content`. Sin contarlos, un turno con herramientas se subestima entero y el
#    presupuesto se pasa sin que nadie se entere.
comprobar(
    "las tool_calls suman al coste estimado",
    H._coste_aproximado(asistente(None, [LLAMADA]))
    > H._coste_aproximado(asistente(None)),
)

# 7. El recorte lee, no escribe: la lista que recibe es la del turno en curso y
#    quien la armó sigue usándola después.
original = [SISTEMA, usuario("a"), asistente("b")]
copia_previa = list(original)
H._recortar_historial(original, 50)
comprobar(
    "recortar no modifica la lista que recibe",
    original == copia_previa,
)


# ---------------------------------------------------------------------------
# B. Limpieza al ir sin herramientas: _sin_herramientas (comportamiento 10).
# ---------------------------------------------------------------------------

CON_HERRAMIENTAS = [
    SISTEMA,
    usuario("prendé la luz"),
    asistente(None, [LLAMADA]),
    herramienta('{"ok": true}'),
    asistente("Listo, la prendí"),
    usuario("gracias"),
]

limpio = H._sin_herramientas(CON_HERRAMIENTAS)

# 8. Un role="tool" suelto, sin el assistant con tool_calls que lo justifica, es
#    un 400 seco. Es la mitad del fallo que este camino tiene que evitar.
comprobar(
    "no queda ningún mensaje role=tool",
    all(m.get("role") != "tool" for m in limpio),
)

# 9. La otra mitad: tool_calls que apuntan a herramientas que ya no se declararon
#    en la petición. También es 400.
comprobar(
    "no queda ningún assistant con tool_calls",
    all("tool_calls" not in m for m in limpio),
)

# 10. El assistant que solo traía tool_calls se queda sin nada que decir. Si se
#     dejara, viajaría como {"role":"assistant","content":null} en todos los
#     turnos siguientes de la sesión, sumando ruido y tokens para siempre.
comprobar(
    "descarta el assistant que solo traía tool_calls",
    len(limpio) == 4
    and [m["role"] for m in limpio] == ["system", "user", "assistant", "user"],
)

# 11. Lo que sí tiene texto se conserva tal cual, y en su orden. Limpiar no es
#     resumir.
comprobar(
    "conserva el contenido y el orden de lo que queda",
    [m.get("content") for m in limpio]
    == [SISTEMA["content"], "prendé la luz", "Listo, la prendí", "gracias"],
)

# 12. Un assistant con texto Y tool_calls no se tira: pierde las tool_calls pero
#     su respuesta al usuario sigue siendo contexto válido.
mixto = H._sin_herramientas([asistente("Ya voy", [LLAMADA])])
comprobar(
    "el assistant con texto y tool_calls conserva el texto",
    len(mixto) == 1 and mixto[0]["content"] == "Ya voy"
    and "tool_calls" not in mixto[0],
)

# 13. Un content de solo espacios es lo mismo que vacío: se descarta. Sale de un
#     modelo que razonó y no llegó a escribir nada útil.
comprobar(
    "el assistant con contenido en blanco también se descarta",
    H._sin_herramientas([asistente("   ")]) == [],
)

# 14. NO muta la lista de entrada. Esos dicts salen de `chat_log.content`, que es
#     la lista que Home Assistant PERSISTE y cuyos elementos son dataclasses
#     frozen: tocarlos ahí es corromper la sesión guardada, no un turno.
comprobar(
    "no muta la lista que recibe",
    len(CON_HERRAMIENTAS) == 6
    and [m["role"] for m in CON_HERRAMIENTAS]
    == ["system", "user", "assistant", "tool", "assistant", "user"],
)

# 15. Y tampoco muta los dicts de adentro: el assistant original conserva sus
#     tool_calls. Un `del mensaje["tool_calls"]` pasaría las comprobaciones de
#     arriba y rompería igual.
comprobar(
    "no muta los dicts que recibe",
    CON_HERRAMIENTAS[2].get("tool_calls") == [LLAMADA],
)

# 16. Los dicts que salen son objetos NUEVOS, no los mismos por referencia. Es lo
#     que garantiza que el próximo que escriba encima no toque lo persistido.
comprobar(
    "devuelve dicts nuevos, no los de la entrada",
    all(nuevo is not viejo
        for nuevo in limpio for viejo in CON_HERRAMIENTAS),
)

# 17. Historial vacío: primer turno de una sesión sin herramientas. No tiene que
#     reventar ni inventar mensajes.
comprobar(
    "historial vacío devuelve lista vacía",
    H._sin_herramientas([]) == [],
)


# ---------------------------------------------------------------------------
# C. Orden entre las dos transformaciones: limpiar y DESPUÉS recortar (nota 2.18).
# ---------------------------------------------------------------------------

# Conversación calibrada a mano contra CHARS_PER_TOKEN = 3.5: el sistema son 104
# tokens, cada turno de texto 32, el tool result 89 y el "Listo" 5.
LARGA_LLAMADA = {"id": "a", "type": "function",
                 "function": {"name": "luz",
                              "arguments": '{"entity_id":"light.cuarto"}'}}
U1 = usuario("A" * 100)
A1 = asistente("B" * 100)
U2 = usuario("C" * 100)
A2 = asistente(None, [LARGA_LLAMADA])
T1 = herramienta("R" * 300)
A3 = asistente("Listo")
TURNO = [SISTEMA, U1, A1, U2, A2, T1, A3]
PRESUPUESTO = 234  # 104 del sistema + 130 para el resto

correcto = H._recortar_historial(H._sin_herramientas(TURNO), PRESUPUESTO)
invertido = H._sin_herramientas(H._recortar_historial(TURNO, PRESUPUESTO))

# 18. Al revés, el recorte calibra el presupuesto contra mensajes que están por
#     desaparecer: se gasta los 89 tokens del tool result y los de las tool_calls
#     en algo que la limpieza va a tirar dos líneas más abajo.
comprobar(
    "limpiar antes de recortar conserva más turnos útiles",
    len(correcto) > len(invertido),
)

# 19. Y no es que conserve "un poco menos": acá el orden invertido se queda sin
#     conversación. La ventana le arrancaba en el assistant de las tool_calls,
#     la regla del corte limpio la vació hasta el system, y recién después la
#     limpieza pasó por una lista que ya no tenía nada.
comprobar(
    "el orden invertido se come la conversación entera",
    [m["role"] for m in invertido] == ["system"],
)

# 20. Con el orden correcto sobreviven los cuatro turnos de texto: el
#     presupuesto se gastó solo en lo que iba a viajar.
comprobar(
    "con el orden correcto sobreviven los turnos de texto",
    [m.get("content") for m in correcto]
    == [SISTEMA["content"], "A" * 100, "B" * 100, "C" * 100, "Listo"],
)

# 21. Y el resultado sigue siendo un JSON válido para la API: arranca en un user
#     después del system y no quedó ni un rastro de herramientas.
sin_sistema = [m for m in correcto if m["role"] != "system"]
comprobar(
    "el orden correcto deja una lista que la API acepta",
    sin_sistema[0]["role"] == "user"
    and all(m["role"] != "tool" and "tool_calls" not in m for m in correcto),
)


# ---------------------------------------------------------------------------
# D. Contexto del enrutador de casa: _ultimos_turnos (nota 2.19).
# ---------------------------------------------------------------------------

# El caso real que obliga a mandar los DOS lados: sin la respuesta de la IA, ese
# "apagala" no tiene sustantivo y el enrutador clasifica a ciegas.
CASO_REAL = [
    SISTEMA,
    usuario("¿tengo luces en el cuarto?"),
    asistente("Sí, una"),
    usuario("apagala"),
]
contexto = H._ultimos_turnos(CASO_REAL, 3)

# 22. La respuesta del asistente TIENE que estar. Es la única que dice "una", que
#     es lo que resuelve el pronombre. El plan viejo mandaba solo los turnos del
#     usuario y por eso no alcanzaba.
comprobar(
    "el contexto trae la respuesta del asistente",
    "asistente: Sí, una" in contexto,
)

# 23. Y también el turno del usuario que la provocó: los dos lados o ninguno.
comprobar(
    "el contexto trae el turno previo del usuario",
    "usuario: ¿tengo luces en el cuarto?" in contexto,
)

# 24. El último user es la consulta que se está clasificando: va aparte en el
#     prompt, y repetida acá el modelo la leería como contexto previo.
comprobar(
    "descarta el último turno del usuario",
    "apagala" not in contexto,
)

# 25. En la primera conversión del turno el system todavía es el prompt del turno
#     ANTERIOR, no está vacío: si se colara, el enrutador recibiría el prompt
#     entero de Assist como si fuera conversación.
comprobar(
    "el prompt de sistema no se cuela en el contexto",
    "s" * 350 not in contexto,
)

# 26. Los resultados de herramienta son JSON crudo: ruido puro para clasificar
#     una frase en castellano, y encima caros.
con_tool = [
    usuario("prendé la luz"),
    asistente(None, [LLAMADA]),
    herramienta('{"ok":true,"entity":"light.cuarto"}'),
    asistente("Listo"),
    usuario("y la otra"),
]
contexto_tool = H._ultimos_turnos(con_tool, 3)
comprobar(
    "los resultados de herramienta no entran al contexto",
    "ok" not in contexto_tool and "light.cuarto" not in contexto_tool,
)

# 27. El assistant que solo traía tool_calls no tiene texto que aportar: una
#     línea "asistente: " vacía solo confunde al clasificador.
comprobar(
    "el assistant sin texto no genera línea",
    "asistente: \n" not in contexto_tool + "\n"
    and contexto_tool.count("asistente:") == 1,
)

# 28. Cada línea se corta a 200 caracteres: el sustantivo que resuelve el
#     pronombre entra de sobra, y el enrutador tiene un techo de 150 tokens de
#     salida. Inflarle la entrada solo le suma latencia al turno.
largo = H._ultimos_turnos(
    [usuario("L" * 500), asistente("M" * 500), usuario("ahora")], 3
)
comprobar(
    "recorta cada turno a 200 caracteres",
    all(len(linea.split(": ", 1)[1]) == 200 for linea in largo.split("\n")),
)

# 29. `cantidad` cuenta turnos, o sea pares: 2 turnos son como mucho 4 líneas.
muchos = [SISTEMA]
for i in range(6):
    muchos.append(usuario(f"pregunta {i}"))
    muchos.append(asistente(f"respuesta {i}"))
muchos.append(usuario("la de ahora"))
comprobar(
    "cantidad cuenta pares usuario/asistente",
    len(H._ultimos_turnos(muchos, 2).split("\n")) == 4,
)

# 30. Y son los ÚLTIMOS pares, no los primeros: el contexto que resuelve un
#     pronombre siempre está pegado a la consulta.
comprobar(
    "se queda con los turnos más recientes",
    "respuesta 5" in H._ultimos_turnos(muchos, 2)
    and "respuesta 0" not in H._ultimos_turnos(muchos, 2),
)

# 31. Cantidad cero es "sin contexto": el usuario puede querer que el enrutador
#     clasifique la frase sola, y ahí no se manda nada.
comprobar(
    "cantidad cero devuelve cadena vacía",
    H._ultimos_turnos(CASO_REAL, 0) == "",
)

# 32. Primer turno de la sesión: no hay historial y el único user es la consulta
#     que se clasifica. Tiene que devolver "" y no una línea suelta.
comprobar(
    "el primer turno no genera contexto",
    H._ultimos_turnos([SISTEMA, usuario("prendé la luz")], 3) == "",
)

# 33. Historial vacío: mismo camino, sin reventar.
comprobar(
    "historial vacío devuelve cadena vacía",
    H._ultimos_turnos([], 3) == "",
)


# ---------------------------------------------------------------------------
# E. Rotación de la cadena de modelos: _candidatos.
# ---------------------------------------------------------------------------

AHORA = time.monotonic()
PRINCIPAL = "qwen/qwen3.8-27b"
CADENA = ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"]

# 34. Sin datos de uso manda la preferencia del usuario. Depende del centinela
#     -1e9: un modelo nunca usado tiene que contar siempre como frío.
comprobar(
    "sin historial de uso respeta el orden configurado",
    C._candidatos(PRINCIPAL, CADENA, {}, 60) == [PRINCIPAL, *CADENA],
)

# 35. Rotación preventiva: el principal recién usado baja al final ANTES de que
#     Groq conteste 429. Esperar el rechazo cuesta un viaje de red entero.
orden = C._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA}, 60)
comprobar(
    "el modelo caliente baja al final",
    orden[0] == CADENA[0] and orden[-1] == PRINCIPAL,
)

# 36. Pero no desaparece: se posterga. Un caliente sigue sirviendo de última red.
comprobar(
    "el modelo caliente sigue estando",
    set(orden) == {PRINCIPAL, *CADENA},
)

# 37. Con TODOS calientes se devuelven igual, en el orden de preferencia. Una
#     lista vacía acá sería dejar al usuario sin respuesta en el peor momento.
calientes = {m: AHORA for m in [PRINCIPAL, *CADENA]}
comprobar(
    "todos calientes: no se pierde ninguno",
    C._candidatos(PRINCIPAL, CADENA, calientes, 60) == [PRINCIPAL, *CADENA],
)

# 38. Un uso de hace 120 segundos con enfriamiento de 60 ya no cuenta: la ventana
#     de tokens por minuto de ese modelo hace rato que se liberó.
comprobar(
    "pasado el enfriamiento vuelve a ser preferido",
    C._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA - 120}, 60)[0]
    == PRINCIPAL,
)

# 39. Enfriamiento 0 apaga la rotación preventiva. Load-bearing: la comparación
#     es >= y no >, si no con dos turnos seguidos y un reloj de resolución gruesa
#     este caso se cae.
comprobar(
    "enfriamiento 0 deja el orden configurado",
    C._candidatos(PRINCIPAL, CADENA, calientes, 0) == [PRINCIPAL, *CADENA],
)

# 40. El principal repetido dentro de la cadena no se prueba dos veces: sería
#     gastar un viaje de red en un modelo que acaba de fallar.
comprobar(
    "quita duplicados conservando la primera aparición",
    C._candidatos(PRINCIPAL, [PRINCIPAL, CADENA[0]], {}, 60)
    == [PRINCIPAL, CADENA[0]],
)

# 41. Cadena vacía: configuración mínima válida, queda solo el principal.
comprobar(
    "cadena vacía deja solo el principal",
    C._candidatos(PRINCIPAL, [], {}, 60) == [PRINCIPAL],
)

# 42. Un campo de texto en blanco en la UI llega como "": pedirle a Groq un
#     modelo llamado "" es un 400.
comprobar(
    "ignora entradas vacías de la cadena",
    C._candidatos(PRINCIPAL, ["", CADENA[0]], {}, 60)
    == [PRINCIPAL, CADENA[0]],
)

# 43. La partición fríos/calientes es ESTABLE: dentro de cada grupo se conserva
#     la preferencia que configuró el usuario. Reordenar ahí sería decidir por él.
orden = C._candidatos(PRINCIPAL, CADENA, {PRINCIPAL: AHORA, CADENA[0]: AHORA}, 60)
comprobar(
    "los calientes mantienen su orden relativo al final",
    orden == [CADENA[1], PRINCIPAL, CADENA[0]],
)

# ---------------------------------------------------------------------------
# _aporta_algo -- no reenviar turnos que no dicen nada
#
# Cuando la cadena se agota y se acepta el vacio queda persistido un
# AssistantContent con content=None. Desde ese momento viajaba como
# {"role": "assistant", "content": null} en TODOS los turnos siguientes de la
# sesion: gastaba cupo del minuto para no decir nada, y hay APIs que rechazan un
# mensaje de asistente sin contenido ni llamadas.
# ---------------------------------------------------------------------------

# 1. El caso que motivo la funcion: el turno vacio que dejo una cadena agotada.
comprobar(
    "un turno sin texto ni llamadas no se reenvia",
    A._aporta_algo(None, None) is False,
)
comprobar(
    "texto vacio tampoco",
    A._aporta_algo("", []) is False,
)
comprobar(
    "solo espacios en blanco tampoco",
    A._aporta_algo("   \n\t ", None) is False,
)

# 2. Lo normal si viaja.
comprobar(
    "un turno con texto se reenvia",
    A._aporta_algo("Prendi la luz.", None) is True,
)

# 3. LA GUARDA QUE IMPORTA: con tool_calls se reenvia SIEMPRE, aunque no haya
#    texto. El ToolResultContent que viene detras se empareja por tool_call_id;
#    si se descartara el mensaje que pidio la herramienta, el resultado quedaria
#    huerfano y eso es un 400 seguro. Es el mismo error que _sin_herramientas
#    evita por el otro lado.
comprobar(
    "con tool_calls se reenvia aunque no haya texto",
    A._aporta_algo(None, [{"id": "x"}]) is True,
)
comprobar(
    "y tambien con texto vacio",
    A._aporta_algo("", [{"id": "x"}]) is True,
)

resumen("historial, limpieza y cadena")
