# Groq Cloud API — fork con cadena de respaldo y enrutadores

[![Abrir en HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=JuanJose2325&repository=indigo-groq&category=integration)

Agente de conversación para Home Assistant que habla con
[Groq](https://groq.com). Es un fork de
[HunorLaczko/ha-groq-cloud-api](https://github.com/HunorLaczko/ha-groq-cloud-api)
con los cambios que hicieron falta para que un asistente de voz doméstico
aguante el uso diario dentro del plan gratuito de Groq.

## Por qué existe este fork

El plan gratuito de Groq no es escaso en volumen, es escaso en **ráfagas**: el
límite de tokens por minuto se cuenta sumando entrada más salida, y el historial
entero de la conversación se reenvía en cada turno. Con la API de Assist metiendo
además unos 4150 tokens de esquemas de herramientas en cada petición —el 52 % del
techo de 8000 por minuto—, una conversación normal supera el límite en pocos
turnos, y a partir de ahí falla **siempre**, no de a ratos. Parecía arreglarse
sola al reiniciar Home Assistant, que es justo cuando se descarta el
`conversation_id`.

Por voz eso se nota mucho más que por texto: no hay indicador de carga, así que
un silencio largo es indistinguible de un cuelgue y el usuario repite el comando,
encadenando dos peticiones contra el mismo cupo.

Lo que agrega el fork:

| Qué | Para qué |
|---|---|
| **Recorte del historial** por presupuesto de tokens | Corta la conversación por turnos enteros para que la petición no crezca sin techo. Nunca deja resultados de herramienta huérfanos (eso es un 400). |
| **Cadena de modelos de respaldo** | Los límites de Groq son *por modelo*: cada modelo de la cadena aporta su propia ventana de tokens por minuto. |
| **Rotación preventiva por enfriamiento** | Cambia de modelo *antes* de que Groq devuelva un 429, para no pagar el viaje de red del rechazo. Un modelo en enfriamiento pasa al final de la lista, nunca se descarta. |
| **Enrutador de casa** (opcional) | Un modelo chico decide si la consulta tiene que ver con la casa. Cuando dice que no, el turno va sin el bloque de herramientas y se ahorran esos 4150 tokens. |
| **Enrutador de razonamiento** (opcional) | Otro modelo chico decide si la consulta necesita pensar. Razonar de gusto devolvía respuestas vacías, porque los tokens de pensamiento salen del mismo presupuesto que la respuesta. |
| **Encogido ante un 413** | Un rechazo por tamaño no se arregla rotando de modelo: al siguiente le llega la misma petición. Se baja el techo a la mitad, hasta un piso de 400, y se reintenta el mismo modelo. |
| **Reintento sin pensamiento ante una respuesta vacía** | Si el modelo se gastó el presupuesto razonando y volvió mudo, se le repite la pregunta con el pensamiento apagado antes de abandonarlo. |
| **Traducción del esfuerzo de razonamiento** | Cada familia usa un vocabulario distinto (`default`/`none` en Qwen, `low`/`medium`/`high` en gpt-oss). Sin traducir, saltar de una a otra devuelve un 400 que el TTS terminaba leyendo en voz alta. |
| **`reasoning_format: hidden`** en Qwen y gpt-oss | Sin esto, los modelos devolvían la cadena de pensamiento entera *dentro* de la respuesta, en inglés, y el TTS la leía. La familia se detecta por prefijo del id, no por igualdad de nombre. |
| **Registro de consumo real** | Entrada, salida, tokens cacheados y `finish_reason` en cada petición, en `warning`. Sin eso hay que adivinar si el recorte está bien calibrado. |
| Traducción al español | Los campos nuevos aparecían con su nombre crudo en la interfaz. |

## Instalación

### Con HACS (recomendado)

Botón de arriba, o bien: HACS → menú de los tres puntos → **Repositorios
personalizados** → pegar la URL del repositorio, categoría **Integration** →
Descargar → reiniciar Home Assistant.

### A mano

Copiar `custom_components/groq_cloud_api/` dentro de la carpeta
`custom_components/` de la configuración de Home Assistant y reiniciar.

Después: **Ajustes → Dispositivos y servicios → Añadir integración → Groq Cloud
API**, y pegar la clave de API de [console.groq.com](https://console.groq.com).

> ⚠️ **Ojo con el dominio.** Esta integración conserva el dominio
> `groq_cloud_api` del proyecto original. Eso quiere decir que si tenés instalada
> también la original, HACS puede pisar una con la otra. Renombrar el dominio lo
> evitaría, pero obliga a reconfigurar la integración desde cero (se pierden las
> opciones y hay que volver a elegir el agente en el pipeline de Assist).

## Opciones

Además de las del proyecto original (instrucciones, modelo, tokens máximos,
temperatura, top_p, control de Home Assistant):

- **Esfuerzo de razonamiento** — cuánto piensa el modelo principal *cuando*
  piensa. Dejarlo vacío es un valor legítimo: significa «no configurado», y ahí
  se manda `reasoning_format: hidden` sin ningún esfuerzo. Vacío **no** es lo
  mismo que el máximo.
- **Presupuesto del historial (tokens)** — cuánto historial se reenvía. Tiene que
  dejar sitio para los tokens máximos de respuesta *y*, si hay razonamiento
  activo, para los tokens de pensamiento, que salen de la misma bolsa.
- **Modelos de respaldo** — lista, en orden de preferencia. Los respaldos van
  siempre sin pensamiento: medido acá, el principal contesta pensando en 120-180
  tokens y el suplente quema los 1200 razonando y vuelve vacío.
- **Enfriamiento por modelo (segundos)** — 0 desactiva la rotación preventiva.
- **Enrutador de casa: activar, modelo, esfuerzo y umbral de confianza** —
  apagado por defecto.
- **Enrutador de razonamiento: activar, modelo, esfuerzo y umbral** — apagado por
  defecto.

Cada desplegable de esfuerzo se arma con la familia del modelo de **su** fila, y
Home Assistant no vuelve a dibujar el formulario cuando cambiás un desplegable:
si cambiaste el modelo, guardá y reabrí para ver los valores que le corresponden.

### Los dos enrutadores

Los dos vienen **apagados**, y con los dos apagados el comportamiento es el mismo
que sin ellos: actualizar la integración no le cambia nada a una instalación que
está andando.

Cuando se prenden, corren **en paralelo** contra su propio modelo (o sea, contra
otra ventana de cupo) antes de armar la petición principal, con un techo de 150
tokens y 5 segundos cada uno.

Sus tablas de fallo son **opuestas a propósito**:

- **Casa: ante cualquier fallo conserva las herramientas.** Equivocarse hacia
  «sí» solo gasta tokens; equivocarse hacia «no» deja al usuario sin poder
  prender la luz. Solo puede ahorrar, nunca romper.
- **Razonamiento: ante cualquier fallo no razona.** Pensar de más se come el
  presupuesto de salida y devuelve la respuesta vacía, que por voz se escucha
  como silencio.

Y **apagado no es lo mismo que fallo**: el enrutador de razonamiento apagado
*sí* piensa, con el esfuerzo configurado.

Cada decisión deja una línea en `warning` con el modelo, el veredicto, la
confianza, el motivo y los milisegundos que tardó, para poder auditar si el
enrutador se está equivocando en vez de confiar en él a ciegas.

### Configuración de referencia

La que está funcionando en la instalación de la que salió este fork, sobre el
plan gratuito de Groq:

| Opción | Valor |
|---|---|
| Modelo | `qwen/qwen3.8-27b` |
| Modelos de respaldo | `qwen/qwen3.6-27b` |
| Tokens máximos | 1200 |
| Presupuesto del historial | 2500 |
| Enfriamiento | 60 s |
| Esfuerzo de razonamiento | `default` |
| Enrutadores | apagados |

### Cómo calibrar el presupuesto del historial

El registro de consumo va en `warning`, así que se ve sin tocar la configuración
del `logger`. Mirar la línea `Groq OK`: trae entrada, salida, total y cuántos
tokens vinieron de caché. Si el total se acerca al límite por minuto del modelo,
bajar el presupuesto. Si aparece `RESPUESTA VACÍA`, el razonamiento se comió los
tokens máximos antes de llegar a contestar: subir los tokens máximos, bajar el
esfuerzo, o prender el enrutador de razonamiento.

## Privacidad

Esto manda lo que se le dice al asistente —y el estado de las entidades que
tengas expuestas a Assist— a un servicio en la nube. Groq es el único proveedor
al que se conecta la integración, y también es a quien se le consultan los dos
enrutadores.

Cuando el enrutador de casa dice que la consulta no tiene que ver con la casa, el
turno viaja **sin** el volcado de entidades expuestas: prenderlo manda menos
datos, no más. Lo que sí ve el enrutador de casa son los **nombres** de esas
entidades, y nada más: ni sus estados, ni sus áreas.

La clave de API no se guarda en este repositorio: se carga desde la interfaz de
Home Assistant.

## Pruebas

```bash
pruebas/todas.sh
```

329 comprobaciones sobre las funciones puras que concentran los fallos que más
caro salieron: el recorte y la limpieza del historial, la rotación de la cadena,
la traducción del esfuerzo de razonamiento, la lectura de respuestas y errores,
la máquina de estados del bucle de candidatos, las dos tablas de fallo de los
enrutadores y la compatibilidad con la configuración vieja.

No hace falta pytest ni tener Home Assistant instalado: `pruebas/cargar.py` lee
los módulos de la integración, se queda con los nodos del árbol de sintaxis que
le interesan y los ejecuta con las pocas dependencias sustituidas. Corren contra
el código real, así que si alguien lo edita de más, se rompen. Si una función
pura se renombra o deja de ser de nivel superior, el cargador levanta un
`AssertionError` en vez de saltearse la comprobación en silencio.

## Créditos

Fork de [HunorLaczko/ha-groq-cloud-api](https://github.com/HunorLaczko/ha-groq-cloud-api).
Se conservan el dominio, la estructura y las correcciones originales.
