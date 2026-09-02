# Plan: enrutador de herramientas

Estado: **propuesta, sin implementar.** Escrito el 1 sep 2026 para revisar antes de tocar código.

## El problema, con los números medidos

| | entrada base | preguntas por minuto |
|---|---|---|
| con `llm_hass_api: ["assist"]` | 4.455 tokens | 1 |
| sin herramientas | ~280 tokens | 5 |

Las herramientas de domótica cuestan **~4.150 tokens en cada pregunta**, sea sobre la
casa o sobre pulpos. Son el 52 % del techo de 8.000 TPM del plan gratuito. La mayoría
de las consultas no son de domótica, así que ese costo se paga casi siempre en vano.

## La idea

Un modelo chico y rápido decide, antes de armar la petición, si esta consulta necesita
el bloque de domótica. Si no lo necesita, no se manda.

## Dónde se engancha

En `_async_handle_message`, justo antes de pedir los datos:

```python
usar = await self._decidir_herramientas(user_input, chat_log, options)
await chat_log.async_provide_llm_data(
    user_input.as_llm_context(DOMAIN),
    options.get(CONF_LLM_HASS_API) if usar else None,
    options.get(CONF_PROMPT),
    user_input.extra_system_prompt,
)
```

Verificado en el código de Home Assistant: pasar `None` deja `llm_api` en None (sin
herramientas ni contexto estático), y el `SystemContent` **se reemplaza en el índice 0
en cada llamada**, así que la decisión puede cambiar turno a turno dentro de la misma
conversación sin romper nada.

## El principio que gobierna el diseño: los errores no cuestan lo mismo

- **Falso negativo** — quita las herramientas cuando hacían falta. El asistente no
  puede prender la luz. Se rompe la función primaria de la casa.
- **Falso positivo** — las deja cuando no hacían falta. Se gastan 4.150 tokens de más.

Son asimétricos, así que **todo camino incierto conserva las herramientas**:

| situación | decisión |
|---|---|
| enrutador desactivado | conserva |
| la llamada al enrutador falla (429, timeout, red) | conserva |
| devuelve algo que no es JSON válido | conserva |
| confianza por debajo del umbral | conserva |
| dice, con confianza, que no toca la casa | **quita** |

Dicho de otro modo: el enrutador solo puede *ahorrar*, nunca *romper*. Para romper el
control de la casa tiene que estar seguro y equivocado, que es el caso más raro.

## Dos cosas que el enrutador tiene que ver, y no son obvias

**1. La pregunta no es "¿es una orden para la casa?".** Al quitar el API no se van solo
las herramientas: también se va el contexto estático con la lista de dispositivos.
Entonces «¿tengo luces en el dormitorio?» no es una orden, pero sin ese contexto se
contesta mal. La pregunta correcta es más amplia: *¿esto tiene algo que ver con la casa?*

**2. Necesita algo de contexto previo.** «Apagala» es ambiguo suelto, pero después de
«prendé la luz del dormitorio» es evidentemente domótica. Se le pasa el turno anterior
del usuario (unos 20 tokens) para que las referencias no lo confundan.

Además se le inyectan los nombres de las entidades expuestas —hoy son cinco, unos 30
tokens— para que reconozca «agregá pan a la lista» como domótica sin adivinar.

## El prompt

```
Eres un clasificador binario. Decidís si una consulta de voz necesita acceso a
las herramientas y al contexto de la casa inteligente, o si se puede responder
sin ellos.

NECESITA LA CASA (needs_home = true):
- Órdenes sobre dispositivos: prender, apagar, subir, bajar, poner, cambiar.
- Preguntas por el estado actual: si algo está encendido, la temperatura, el modo.
- Preguntas sobre qué dispositivos existen o dónde están.
- Cualquier cosa sobre estas entidades de la casa:
{ENTIDADES}
- Listas de tareas y de la compra: agregar, quitar, consultar qué hay.
- Multimedia: reproducir, pausar, volumen, siguiente.
- Referencias a algo mencionado antes que fuera de la casa («apagala», «ponela»).

NO LA NECESITA (needs_home = false):
- Conocimiento general, definiciones, datos curiosos, historia, ciencia.
- Matemática, programación, lógica.
- Charla, saludos, agradecimientos.
- Traducciones, redacción, texto creativo.
- Consejos y opiniones sin relación con la casa.

REGLA QUE MANDA SOBRE TODAS LAS DEMÁS:
Ante CUALQUIER duda, respondé true. Equivocarte hacia true solo gasta tokens;
equivocarte hacia false deja al usuario sin poder controlar su casa. No son
igual de graves. Si dudás, es true.

SALIDA:
Respondé EXCLUSIVAMENTE un objeto JSON válido. Sin markdown, sin ```json, sin
explicaciones antes ni después.

{"needs_home": true/false, "confidence": 0.0-1.0, "reason": "menos de 8 palabras"}

Turno anterior del usuario (contexto, puede estar vacío):
"{TURNO_ANTERIOR}"

Consulta a clasificar:
"{CONSULTA}"
```

Diferencias a propósito con tu enrutador de razonamiento: la regla de desempate es
explícita y está justificada (los modelos obedecen mejor una regla con motivo que una
orden suelta), y el contexto previo entra como campo separado para que no se confunda
con la consulta a clasificar.

## Opciones de configuración nuevas

| opción | por defecto | para qué |
|---|---|---|
| `router_enabled` | **false** | Apagado por defecto: una instalación que actualiza no cambia de comportamiento sin que el dueño lo decida. |
| `router_model` | `openai/gpt-oss-20b` | El más rápido de Groq y con **su propia ventana** de 8.000 TPM, así que no le come cupo al principal. |
| `router_threshold` | `0.7` | Por debajo de esta confianza se conservan las herramientas. |

## Costo

Una llamada extra de ~250 tokens de entrada y ~30 de salida a un modelo de 1000 t/s:
**unos 300 ms**. Sale de una ventana de cupo distinta a la del modelo principal.

A cambio, en las consultas que no son de domótica —la mayoría— la petición baja de
~4.455 a ~280 tokens de entrada.

## El detalle técnico que hay que cubrir

Si un turno usó herramientas y el siguiente va sin ellas, el historial arrastra
mensajes `role: "tool"` y `tool_calls` sin herramientas declaradas, lo que puede dar un
HTTP 400. Al ir sin herramientas hay que limpiar el historial: quitar los mensajes de
rol `tool` y despojar de `tool_calls` a los mensajes del asistente.

## Qué se va a poder ver en el log

Una línea por decisión, con el modelo, el veredicto, la confianza, el motivo y los
milisegundos que tardó. Así se puede auditar con `~/simular-assist/cupo.sh` si el
enrutador se está equivocando, en vez de tener que confiar en él a ciegas.

## Pruebas previstas

- La tabla de incertidumbre entera: cada fila de arriba, comprobando que conserva.
- JSON inválido, JSON sin las claves, confianza fuera de rango, respuesta vacía.
- Limpieza del historial: que no queden `tool` huérfanos ni `tool_calls` sueltos.
- Que la decisión no se filtre entre turnos (cada consulta se decide sola).

## Lo que NO hace este plan

No toca el techo de 8.000 TPM: lo esquiva. Con el enrutador andando, una ráfaga de
preguntas de domótica seguidas va a seguir chocando contra el mismo muro, porque esas
sí pagan los 4.150 tokens. Para eso la única salida sigue siendo el Developer Tier.
