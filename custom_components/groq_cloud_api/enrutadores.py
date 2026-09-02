"""Los dos enrutadores que deciden, antes de armar la petición, qué se manda.

Frontera de este módulo: acá viven los dos prompts, la llamada en paralelo a
Groq y la interpretación de lo que devuelven. Todo lo que DECIDE —parseo,
veredictos y la tabla de fallos— es puro y se carga por AST desde las pruebas;
lo único que toca la red son las dos corrutinas del final.

Existe por un número medido: con `llm_hass_api: ["assist"]` la entrada base de
cada pregunta es de 4.455 tokens, y sin herramientas es de ~280. Ese bloque de
domótica cuesta ~4.150 tokens, el 52 % del techo de 8.000 TPM del plan gratuito
de Groq, y se paga igual cuando la pregunta es sobre pulpos. El enrutador de
casa lo saca cuando no hace falta; el de razonamiento hace lo propio con los
tokens de pensamiento, que salen de la misma bolsa que `max_tokens`.

Los dos enrutadores corren contra su PROPIA ventana de cupo (otro modelo), así
que no le comen tokens al principal, y por eso tampoco entran en el
enfriamiento de la cadena: no son eslabones, son otra ventana.
"""

from __future__ import annotations

import asyncio
import json
import time

from .const import (
    CONF_CASA_ROUTER_EFFORT,
    CONF_CASA_ROUTER_ENABLED,
    CONF_CASA_ROUTER_MODEL,
    CONF_CASA_ROUTER_THRESHOLD,
    CONF_RAZON_ROUTER_EFFORT,
    CONF_RAZON_ROUTER_ENABLED,
    CONF_RAZON_ROUTER_MODEL,
    CONF_RAZON_ROUTER_THRESHOLD,
    LOGGER,
    RECOMMENDED_ROUTER_EFFORT,
    RECOMMENDED_ROUTER_ENABLED,
    RECOMMENDED_ROUTER_MODEL,
    RECOMMENDED_ROUTER_THRESHOLD,
    ROUTER_MAX_TOKENS,
    TIEMPO_MAXIMO_ENRUTADOR,
)
from .razonamiento import _aplicar_razonamiento

# --------------------------------------------------------------------------
# PURO: prompts, parseo y veredictos. Nada de acá toca la red ni Home Assistant.
# --------------------------------------------------------------------------

# La pregunta NO es "¿es una orden para la casa?". Al pasar `None` a
# `async_provide_llm_data` no se van solo las herramientas: se va también el
# `api_prompt`, que es el volcado YAML de todas las entidades expuestas con su
# área. Sin ese bloque, "¿tengo luces en el dormitorio?" —que no es una orden—
# se contesta mal. Por eso el prompt pregunta por PERTENENCIA al dominio de la
# casa, que es más ancho que el de las órdenes.
#
# Y la regla de desempate va explícita Y JUSTIFICADA: un modelo obedece mucho
# mejor una regla que trae su motivo que una orden suelta.
_PROMPT_CASA = """Sos un clasificador binario. Decidís si una consulta de voz necesita el contexto y las herramientas de la casa inteligente, o si se puede responder sin ellos.

No respondas la consulta. Solo clasificala.

TIENE QUE VER CON LA CASA (needs_home = true):
- Órdenes sobre dispositivos: prender, apagar, subir, bajar, poner, cambiar.
- Preguntas por el estado actual: si algo está encendido, la temperatura, el modo.
- Preguntas sobre qué dispositivos hay, dónde están o cómo se llaman.
- Cualquier cosa que mencione una de estas entidades de la casa:
{ENTIDADES}
- Listas de tareas y de la compra: agregar, quitar, consultar qué hay.
- Multimedia: reproducir, pausar, volumen, siguiente.
- Referencias a algo de la casa nombrado antes en la conversación ("apagala",
  "ponela", "la otra también", "subile un poco").
- Rutinas, escenas, automatizaciones, temporizadores y alarmas.

NO TIENE QUE VER CON LA CASA (needs_home = false):
- Conocimiento general, definiciones, datos curiosos, historia, ciencia.
- Matemática, programación, lógica.
- Charla, saludos, agradecimientos.
- Traducciones, redacción, texto creativo.
- Consejos y opiniones sin relación con la casa.
- La hora y el clima de afuera.

REGLA QUE MANDA SOBRE TODAS LAS DEMÁS:
Ante CUALQUIER duda, respondé true. Los dos errores no cuestan lo mismo:
equivocarte hacia true solo gasta tokens, mientras que equivocarte hacia false
deja al usuario sin poder controlar su casa. Si dudás, es true.

SALIDA:
Respondé EXCLUSIVAMENTE un objeto JSON válido. Sin markdown, sin cercas de
código, sin explicaciones antes ni después.

{"needs_home": true|false, "confidence": 0.0-1.0, "reason": "menos de 8 palabras"}

Turnos anteriores de la conversación (contexto, puede estar vacío):
{CONTEXTO}

Consulta a clasificar:
{CONSULTA}"""

# El de razonamiento tiene el trabajo opuesto: acá el error caro es razonar de
# más. Los tokens de pensamiento salen de `max_tokens`, así que con un techo
# chico un modelo que piensa demasiado devuelve la respuesta VACÍA y el usuario
# escucha silencio. Medido en la instalación real: el principal contesta
# pensando en 120-180 tokens, y el respaldo quema los 1200 enteros razonando y
# vuelve sin una palabra.
#
# La distinción fina que hay que sostener es anuncio contra tarea: "necesito
# ayuda en matemática" no es un problema que resolver, es el preámbulo; "resolvé
# esta sumatoria paso a paso" sí lo es.
_PROMPT_RAZONAMIENTO = """Sos un clasificador binario. Decidís si una consulta de voz necesita que el modelo piense paso a paso antes de contestar, o si se contesta de una.

No respondas la consulta. Solo clasificala.

NECESITA PENSAR (reasoning_required = true):
- Problemas de matemática con cuentas, despejes, demostraciones o varios pasos.
- Programación: escribir, depurar o explicar código no trivial.
- Acertijos, deducción, planificación con restricciones, comparaciones con
  varios criterios a la vez.
- Cualquier consulta donde equivocarse en un paso intermedio arruina la
  respuesta entera.

NO NECESITA PENSAR (reasoning_required = false):
- ANUNCIOS Y PREÁMBULOS. "Necesito ayuda con matemática", "te quiero preguntar
  algo de física", "tengo una duda de programación" no son problemas: son el
  aviso de que viene uno. Se contestan con "dale, decime".
- Datos, definiciones y hechos que se saben o no se saben.
- Charla, saludos, agradecimientos, opiniones.
- Órdenes para la casa y preguntas por el estado de la casa.
- Redacción, traducción y texto creativo.

REGLA QUE MANDA SOBRE TODAS LAS DEMÁS:
Ante CUALQUIER duda, respondé false. Los dos errores no cuestan lo mismo: no
pensar cuando convenía da una respuesta más pobre, pero pensar cuando no hacía
falta se come el presupuesto de salida entero y devuelve una respuesta VACÍA,
que el usuario escucha como silencio. Si dudás, es false.

SALIDA:
Respondé EXCLUSIVAMENTE un objeto JSON válido. Sin markdown, sin cercas de
código, sin explicaciones antes ni después.

{"reasoning_required": true|false, "confidence": 0.0-1.0, "category": "math|coding|logic|general|creative", "brief_reason": "menos de 8 palabras"}

Consulta a clasificar:
{CONSULTA}"""

# APAGADO NO ES LO MISMO QUE FALLO, y en el de razonamiento son opuestos. Estas
# dos constantes están en el nivel superior justamente para que una prueba pueda
# clavar esa asimetría: apagado significa "comportamiento de siempre" (casa
# conserva las herramientas, razonamiento SÍ piensa con el esfuerzo
# configurado), mientras que fallo significa "el camino barato y seguro" (casa
# conserva igual, razonamiento NO piensa).
VEREDICTO_CASA_APAGADO = (True, 1.0, "enrutador de casa apagado")
VEREDICTO_RAZONAMIENTO_APAGADO = (True, 1.0, "general", "enrutador de razonamiento apagado")


def _lista_entidades(nombres: list[str], tope: int) -> str:
    """Los nombres de las entidades expuestas, uno por línea y como mucho `tope`."""
    # Van al prompt para que "agregá pan a la lista" se reconozca como domótica
    # sin que el enrutador tenga que adivinar que existe una lista de la compra.
    # Con las cinco entidades expuestas de esta casa son unos 30 tokens; el tope
    # está para que una instalación con doscientas entidades no convierta al
    # enrutador en un gasto más caro que el bloque que viene a evitar.
    vistos = []
    for nombre in nombres or []:
        limpio = str(nombre).strip()
        if not limpio or limpio in vistos:
            continue
        vistos.append(limpio)
        if len(vistos) >= tope:
            break
    return "\n".join(vistos)


def _prompt_casa(entidades: str, contexto: str, consulta: str) -> str:
    """Arma el prompt del enrutador de casa con las entidades, los turnos previos y la consulta."""
    # Se sustituye con `replace` y no con `format`: el prompt lleva adentro el
    # ejemplo de salida JSON, y sus llaves harían reventar a `format` con un
    # KeyError en el camino de la respuesta al usuario.
    texto = _PROMPT_CASA
    texto = texto.replace("{ENTIDADES}", (entidades or "").strip() or "(ninguna)")
    texto = texto.replace("{CONTEXTO}", (contexto or "").strip() or "(sin turnos previos)")
    return texto.replace("{CONSULTA}", (consulta or "").strip())


def _prompt_razonamiento(consulta: str) -> str:
    """Arma el prompt del enrutador de razonamiento con la consulta."""
    # Este no recibe contexto a propósito: si necesita pensar o no lo dice la
    # consulta sola, y cada token que se le suma se paga en latencia por voz.
    return _PROMPT_RAZONAMIENTO.replace("{CONSULTA}", (consulta or "").strip())


def _texto_del_enrutador(resultado: object) -> str | None:
    """El texto que devolvió un enrutador; None si falló, si vino vacío o si se truncó."""
    # Acepta indistintamente lo que salga de asyncio.gather(return_exceptions=True):
    # una respuesta, una excepción o None. Una excepción se traga y se convierte
    # en None SIN re-levantarla: el fallo de un enrutador no puede tumbar al otro
    # ni al turno entero. Quien traduce ese None en una decisión es el veredicto,
    # que es el único que sabe hacia qué lado se falla.
    if resultado is None or isinstance(resultado, BaseException):
        return None
    eleccion = (getattr(resultado, "choices", None) or [None])[0]
    if eleccion is None:
        return None
    # Truncado es fallo acá aunque en la respuesta principal no lo sea: el
    # enrutador tiene 150 tokens de techo para escupir un JSON de tres claves,
    # así que si se cortó por longitud lo que llegó es JSON a medio cerrar.
    if getattr(eleccion, "finish_reason", None) == "length":
        return None
    mensaje = getattr(eleccion, "message", None)
    contenido = getattr(mensaje, "content", None)
    # Todo por getattr y sin isinstance contra clases del SDK, salvo este de
    # str: un `content` que llega como lista de bloques —lo que devuelven otros
    # proveedores y alguna versión del SDK— hace reventar al `.strip()` con un
    # AttributeError, y acá arriba no hay ningún except: esa excepción sube por
    # _decidir_enrutadores y se lleva puesto el turno entero, que es justo lo
    # que la tabla de fallos existe para evitar. Lo que no es texto se trata
    # como fallo y cada tabla lo resuelve hacia su lado seguro.
    if not isinstance(contenido, str):
        return None
    return contenido.strip() or None


def _extraer_json(texto: str | None) -> dict | None:
    """El primer objeto JSON del texto, tolerando cercas ```json y prosa alrededor; None si no hay."""
    # Un clasificador al que se le pide "solo JSON" igual devuelve de vez en
    # cuando ```json ... ``` o una frase de cortesía adelante. Rechazar eso
    # mandaría el veredicto a la rama de fallo por un detalle de formato, así
    # que se busca el primer objeto balanceado y se ignora el resto.
    if not isinstance(texto, str):
        return None
    profundidad = 0
    inicio = -1
    en_cadena = False
    escapado = False
    for posicion, caracter in enumerate(texto):
        if en_cadena:
            if escapado:
                escapado = False
            elif caracter == "\\":
                escapado = True
            elif caracter == '"':
                en_cadena = False
            continue
        if caracter == '"':
            en_cadena = True
        elif caracter == "{":
            if profundidad == 0:
                inicio = posicion
            profundidad += 1
        elif caracter == "}":
            if profundidad == 0:
                continue
            profundidad -= 1
            if profundidad == 0 and inicio >= 0:
                try:
                    datos = json.loads(texto[inicio:posicion + 1])
                except ValueError:
                    # No era JSON válido: puede haber otro objeto más adelante
                    # (típico cuando el modelo escribe un ejemplo y después la
                    # respuesta de verdad), así que se sigue buscando.
                    inicio = -1
                    continue
                if isinstance(datos, dict):
                    return datos
                inicio = -1
    return None


def _veredicto_casa(resultado: object, umbral: float) -> tuple:
    """Si esta consulta necesita el contexto y las herramientas de la casa: (necesita, confianza, motivo)."""
    # ANTE FALLO CONSERVA LAS HERRAMIENTAS. Los dos errores son asimétricos: un
    # falso positivo gasta 4.150 tokens de más, un falso negativo deja al
    # usuario sin poder prender la luz, que es la función primaria de la casa.
    # Dicho de otro modo: este enrutador solo puede ahorrar, nunca romper. Para
    # romper el control de la casa tiene que estar seguro Y equivocado.
    texto = _texto_del_enrutador(resultado)
    if texto is None:
        return (True, 0.0, "el enrutador de casa no contestó")
    datos = _extraer_json(texto)
    if datos is None:
        return (True, 0.0, "el enrutador de casa no devolvió JSON")
    necesita = datos.get("needs_home")
    if not isinstance(necesita, bool):
        return (True, 0.0, "needs_home ausente o no booleano")
    confianza = datos.get("confidence")
    # bool es subclase de int en Python: sin descartarlo, `"confidence": true`
    # pasaría como 1.0 y le daría al enrutador una certeza que nunca declaró.
    if isinstance(confianza, bool) or not isinstance(confianza, (int, float)):
        return (True, 0.0, "confianza ausente o de tipo equivocado")
    confianza = float(confianza)
    # Escrito como rango y no como `< 0.0 or > 1.0` por NaN: json.loads acepta
    # el literal NaN sin chistar, y TODA comparación con NaN da False, así que
    # la forma negada lo dejaba pasar. Con la confianza en NaN el `< umbral` de
    # más abajo también da False y las herramientas se apagaban con una certeza
    # que no existe: es el único valor que derrota a las dos tablas a la vez.
    if not 0.0 <= confianza <= 1.0:
        return (True, 0.0, "confianza fuera de [0,1]")
    motivo = datos.get("reason")
    motivo = motivo.strip()[:80] if isinstance(motivo, str) and motivo.strip() else "sin motivo"
    if necesita:
        return (True, confianza, motivo)
    # La poca confianza se resuelve para el mismo lado que el fallo: conservar.
    if confianza < umbral:
        return (True, confianza, "poca confianza para quitar: " + motivo)
    return (False, confianza, motivo)


def _veredicto_razonamiento(resultado: object, umbral: float) -> tuple:
    """Si esta consulta necesita pensar: (razona, confianza, categoría, motivo)."""
    # ANTE FALLO NO RAZONA, que es la tabla OPUESTA a la del enrutador de casa y
    # es a propósito: razonar de más devuelve respuestas vacías, y una respuesta
    # vacía por voz es silencio. Ojo: esto es el FALLO. El enrutador APAGADO sí
    # razona, con el esfuerzo configurado (VEREDICTO_RAZONAMIENTO_APAGADO).
    texto = _texto_del_enrutador(resultado)
    if texto is None:
        return (False, 0.0, "general", "el enrutador de razonamiento no contestó")
    datos = _extraer_json(texto)
    if datos is None:
        return (False, 0.0, "general", "el enrutador de razonamiento no devolvió JSON")
    razona = datos.get("reasoning_required")
    if not isinstance(razona, bool):
        return (False, 0.0, "general", "reasoning_required ausente o no booleano")
    confianza = datos.get("confidence")
    # Mismo descarte de bool que en el veredicto de casa, por el mismo motivo.
    if isinstance(confianza, bool) or not isinstance(confianza, (int, float)):
        return (False, 0.0, "general", "confianza ausente o de tipo equivocado")
    confianza = float(confianza)
    # Mismo rango cerrado que en el veredicto de casa, y por el mismo NaN: acá
    # dejarlo pasar mandaba a pensar con certeza inventada, que es como se
    # llegaba a la respuesta vacía.
    if not 0.0 <= confianza <= 1.0:
        return (False, 0.0, "general", "confianza fuera de [0,1]")
    # La categoría es informativa: sirve para auditar el log, no para decidir.
    # Una categoría inventada por el modelo se normaliza a "general" y NO
    # invalida el veredicto, que ya venía respaldado por needs/confidence.
    categoria = datos.get("category")
    if categoria not in ("math", "coding", "logic", "general", "creative"):
        categoria = "general"
    motivo = datos.get("brief_reason")
    motivo = motivo.strip()[:80] if isinstance(motivo, str) and motivo.strip() else "sin motivo"
    if not razona:
        return (False, confianza, categoria, motivo)
    # Poca confianza cae para el mismo lado que el fallo: no pensar.
    if confianza < umbral:
        return (False, confianza, categoria, "poca confianza para pensar: " + motivo)
    return (True, confianza, categoria, motivo)


def _ajustes_enrutadores(options: dict) -> dict:
    """Lee de las opciones los ajustes de los dos enrutadores, con defaults para las claves nuevas."""
    # Todo con options.get(clave, RECOMMENDED_*) para que una entrada que viene
    # de la versión anterior —con cf_account_id, cf_api_token, cf_model,
    # reasoning_effort_chain y supports_reasoning adentro— siga andando sin
    # tocar nada: las claves viejas no se leen, y las nuevas, al faltar, dejan
    # los dos enrutadores APAGADOS. Actualizar la integración no puede cambiarle
    # el comportamiento a una instalación que está andando.
    ajustes = {}
    for prefijo, clave_activo, clave_modelo, clave_esfuerzo, clave_umbral in (
        (
            "casa",
            CONF_CASA_ROUTER_ENABLED,
            CONF_CASA_ROUTER_MODEL,
            CONF_CASA_ROUTER_EFFORT,
            CONF_CASA_ROUTER_THRESHOLD,
        ),
        (
            "razon",
            CONF_RAZON_ROUTER_ENABLED,
            CONF_RAZON_ROUTER_MODEL,
            CONF_RAZON_ROUTER_EFFORT,
            CONF_RAZON_ROUTER_THRESHOLD,
        ),
    ):
        umbral = options.get(clave_umbral, RECOMMENDED_ROUTER_THRESHOLD)
        try:
            umbral = float(umbral)
        except (TypeError, ValueError):
            umbral = RECOMMENDED_ROUTER_THRESHOLD
        # Se acota en vez de rechazarse: un umbral de 1.5 guardado a mano
        # dejaría al enrutador sin poder decidir nunca, y eso se ve como "el
        # enrutador no hace nada" en lugar de como un error de configuración.
        umbral = min(1.0, max(0.0, umbral))
        ajustes[prefijo + "_activo"] = bool(
            options.get(clave_activo, RECOMMENDED_ROUTER_ENABLED)
        )
        # El `or` cubre el modelo guardado como cadena vacía: sin él saldría una
        # petición con model="" que Groq rechaza con un 400 en cada turno.
        ajustes[prefijo + "_modelo"] = (
            options.get(clave_modelo, RECOMMENDED_ROUTER_MODEL)
            or RECOMMENDED_ROUTER_MODEL
        )
        # El esfuerzo SÍ conserva la cadena vacía: acá vacío significa "no
        # configurado", que no es lo mismo que un valor inválido y no es lo
        # mismo que el máximo.
        ajustes[prefijo + "_esfuerzo"] = options.get(
            clave_esfuerzo, RECOMMENDED_ROUTER_EFFORT
        )
        ajustes[prefijo + "_umbral"] = umbral
    return ajustes


def _linea_decision(enrutador: str, modelo: str, veredicto: bool,
                    confianza: float, motivo: str, ms: int) -> str:
    """La línea de auditoría de una decisión de enrutador, con modelo, veredicto, confianza, motivo y ms."""
    # Los cinco datos van juntos en UNA línea porque con esto se audita si el
    # enrutador se está equivocando (~/simular-assist/cupo.sh) en vez de
    # confiar en él a ciegas. La imprime el llamador con LOGGER.warning: el
    # usuario tiene `logger: default: warning` en configuration.yaml, así que
    # en INFO esto no lo vería nunca.
    return (
        f"enrutador {enrutador}: modelo={modelo} "
        f"veredicto={'sí' if veredicto else 'no'} "
        f"confianza={float(confianza):.2f} motivo=\"{motivo}\" ms={int(ms)}"
    )


# --------------------------------------------------------------------------
# ASYNC: lo único que toca la red. Nada de acá se carga por AST.
# --------------------------------------------------------------------------


async def _consultar_enrutador(cliente: object, modelo: str, esfuerzo: str | None,
                               prompt: str, tiempo_maximo: float) -> object:
    """Una llamada a un enrutador, con su propio timeout; devuelve la respuesta cruda."""
    kwargs = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ROUTER_MAX_TOKENS,
        # Un clasificador no tiene que ser creativo: con temperatura 0 la misma
        # consulta da el mismo veredicto, que es lo que hace auditable el log.
        "temperature": 0,
    }
    # Sin esto, un enrutador de familia qwen o gpt-oss devuelve el pensamiento
    # DENTRO de content, el JSON no parsea y el veredicto se va derecho a la
    # rama de fallo. Con 150 tokens de techo, además, el pensamiento se come el
    # presupuesto entero y la respuesta vuelve truncada.
    _aplicar_razonamiento(kwargs, modelo, esfuerzo)
    # Por voz, un enrutador colgado es peor que un enrutador equivocado: la
    # decisión equivocada cuesta tokens, la colgada cuesta el turno entero. El
    # timeout se levanta como excepción y gather la convierte en un fallo, que
    # cada tabla resuelve hacia su lado seguro.
    async with asyncio.timeout(tiempo_maximo):
        return await cliente.chat.completions.create(**kwargs)


async def _decidir_enrutadores(cliente: object, ajustes: dict, consulta: str,
                               entidades: list[str], contexto: str) -> tuple:
    """Corre los dos enrutadores EN PARALELO y devuelve los dos veredictos ya interpretados."""
    # En paralelo y no en fila porque los dos son latencia pura delante de la
    # respuesta: encadenados serían ~600 ms antes de que el asistente empiece a
    # trabajar, juntos son ~300. Y con return_exceptions=True el fallo de uno no
    # se lleva puesto al otro.
    #
    # Ninguno de los dos toca `ultimo_uso`: no son eslabones de la cadena, corren
    # contra otra ventana de cupo. Si el usuario elige de enrutador un modelo que
    # también está en la cadena principal se van a comer cupo entre ellos, y eso
    # se avisa en el formulario, no se resuelve acá.

    async def _cronometrado(corutina: object, marcas: dict, clave: str) -> object:
        """Corre la corrutina anotando cuánto tardó, sin tocar lo que devuelve ni lo que levanta."""
        # El `finally` sin `except` es deliberado: mide igual cuando revienta,
        # pero deja pasar la excepción intacta para que la recoja el gather.
        inicio = time.monotonic()
        try:
            return await corutina
        finally:
            marcas[clave] = int((time.monotonic() - inicio) * 1000)

    # Tope de nombres que viajan en el prompt del enrutador de casa. Con las
    # cinco entidades expuestas de esta instalación sobra; el número está para
    # que una casa grande no vuelva al enrutador más caro que las herramientas.
    lista = _lista_entidades(entidades, 40)

    marcas: dict = {}
    faenas: dict = {}
    if ajustes.get("casa_activo"):
        faenas["casa"] = _cronometrado(
            _consultar_enrutador(
                cliente,
                ajustes["casa_modelo"],
                ajustes["casa_esfuerzo"],
                _prompt_casa(lista, contexto, consulta),
                TIEMPO_MAXIMO_ENRUTADOR,
            ),
            marcas,
            "casa",
        )
    if ajustes.get("razon_activo"):
        faenas["razon"] = _cronometrado(
            _consultar_enrutador(
                cliente,
                ajustes["razon_modelo"],
                ajustes["razon_esfuerzo"],
                _prompt_razonamiento(consulta),
                TIEMPO_MAXIMO_ENRUTADOR,
            ),
            marcas,
            "razon",
        )

    claves = list(faenas)
    crudos = []
    if claves:
        crudos = await asyncio.gather(
            *(faenas[clave] for clave in claves), return_exceptions=True
        )
    resultados = dict(zip(claves, crudos))

    if "casa" in resultados:
        casa = _veredicto_casa(resultados["casa"], ajustes["casa_umbral"])
        LOGGER.warning(
            _linea_decision(
                "casa", ajustes["casa_modelo"], casa[0], casa[1], casa[2],
                marcas.get("casa", 0),
            )
        )
    else:
        # Apagado no es fallo, aunque acá coincidan: conserva las herramientas
        # porque ese es el comportamiento de siempre.
        casa = VEREDICTO_CASA_APAGADO

    if "razon" in resultados:
        razon = _veredicto_razonamiento(resultados["razon"], ajustes["razon_umbral"])
        LOGGER.warning(
            _linea_decision(
                "razonamiento", ajustes["razon_modelo"], razon[0], razon[1],
                f"[{razon[2]}] {razon[3]}", marcas.get("razon", 0),
            )
        )
    else:
        # Acá apagado y fallo NO coinciden: apagado SÍ razona, con el esfuerzo
        # que el usuario configuró. Quitarle el pensamiento a una instalación
        # que anda, solo por actualizar, sería cambiarle el comportamiento.
        razon = VEREDICTO_RAZONAMIENTO_APAGADO

    return (casa, razon)
