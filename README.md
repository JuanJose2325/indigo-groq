# Groq Cloud API — fork con cadena de respaldo

[![Abrir en HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=TU-USUARIO&repository=indigo-groq&category=integration)

> **Antes de publicar:** en el enlace de arriba cambiá `TU-USUARIO` por tu usuario
> de GitHub y `indigo-groq` por el nombre del repositorio, si le ponés otro. Ese
> botón no funciona hasta que el repositorio exista en GitHub.

Agente de conversación para Home Assistant que habla con
[Groq](https://groq.com). Es un fork de
[HunorLaczko/ha-groq-cloud-api](https://github.com/HunorLaczko/ha-groq-cloud-api)
con los cambios que hicieron falta para que un asistente de voz doméstico
aguante el uso diario dentro del plan gratuito de Groq.

## Por qué existe este fork

El plan gratuito de Groq no es escaso en volumen, es escaso en **ráfagas**: el
límite de tokens por minuto se cuenta sumando entrada más salida, y el historial
entero de la conversación se reenvía en cada turno. Con la API de Assist metiendo
además unos 4400 tokens de esquemas de herramientas en cada petición, una
conversación normal supera el límite en pocos turnos — y a partir de ahí falla
**siempre**, no de a ratos. Parecía arreglarse sola al reiniciar Home Assistant,
que es justo cuando se descarta el `conversation_id`.

Por voz eso se nota mucho más que por texto: no hay indicador de carga, así que
un silencio largo es indistinguible de un cuelgue y el usuario repite el comando,
encadenando dos peticiones.

Lo que agrega el fork:

| Qué | Para qué |
|---|---|
| **Recorte del historial** por presupuesto de tokens | Corta la conversación por turnos enteros para que la petición no crezca sin techo. Nunca deja resultados de herramienta huérfanos (eso es un 400). |
| **Cadena de modelos de respaldo** | Los límites de Groq son *por modelo*: cada modelo de la cadena aporta su propia ventana de tokens por minuto. |
| **Rotación preventiva por enfriamiento** | Cambia de modelo *antes* de que Groq devuelva un 429, para no pagar el viaje de red del rechazo. Un modelo en enfriamiento pasa al final de la lista, nunca se descarta. |
| **Respaldo en Cloudflare Workers AI** | Red de última instancia. Ahí corre el mismo `qwen3.8-27b` y no hay límite de tokens por minuto, así que la calidad se mantiene; solo es más lento. |
| **Traducción del esfuerzo de razonamiento** | Cada familia de modelos usa un vocabulario distinto (`default`/`none` en Qwen, `low`/`medium`/`high` en gpt-oss). Sin traducir, saltar de una a otra devuelve un 400 que el TTS terminaba leyendo en voz alta. |
| **`reasoning_format: hidden`** en Qwen y gpt-oss | Sin esto, los modelos devolvían la cadena de pensamiento entera *dentro* de la respuesta, en inglés, y el TTS la leía. |
| **Registro de consumo real** | Entrada, salida, tokens cacheados y `finish_reason` en cada petición. Sin eso hay que adivinar si el recorte está bien calibrado. |
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
temperatura, top_p, control de Home Assistant, esfuerzo de razonamiento):

- **Presupuesto del historial (tokens)** — cuánto historial se reenvía. Tiene que
  dejar sitio para los tokens máximos de respuesta *y*, si hay razonamiento
  activo, para los tokens de pensamiento, que salen de la misma bolsa.
- **Modelos de respaldo** — lista, en orden de preferencia.
- **Enfriamiento por modelo (segundos)** — 0 desactiva la rotación preventiva.
- **Cloudflare: ID de cuenta, token y modelo** — vacíos por defecto (desactivado).

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
| Esfuerzo de razonamiento | `default` (con «admite razonamiento» activado) |
| Cloudflare | `@cf/qwen/qwen3.8-27b` |

Con eso: mediana de aproximadamente 1 s en Groq y de 3,4 s cuando cae a
Cloudflare, con picos ocasionales cerca de 20 s.

### Cómo calibrar el presupuesto del historial

Poner el registro en `debug` para `custom_components.groq_cloud_api` y mirar la
línea `Groq OK`: trae entrada, salida, total y cuántos tokens vinieron de caché.
Si el total se acerca al límite por minuto del modelo, bajar el presupuesto. Si
aparece `RESPUESTA VACÍA por truncado`, el razonamiento se comió los tokens
máximos antes de llegar a contestar: subir los tokens máximos o bajar el
esfuerzo.

## Privacidad

Esto manda lo que se le dice al asistente —y el estado de las entidades que
tengas expuestas a Assist— a un servicio en la nube.

- **Groq** es obligatorio: es el proveedor principal. Revisá sus condiciones
  antes de usarlo, sobre todo en el plan gratuito.
- **Cloudflare Workers AI** es **opcional y viene desactivado**. Solo se activa
  si cargás a mano el ID de cuenta y el token. Si preferís que tus
  conversaciones no salgan hacia un segundo proveedor, dejá esos campos vacíos:
  la integración funciona igual, solo que sin red de última instancia.

Ninguna credencial se guarda en este repositorio: las dos se cargan desde la
interfaz de Home Assistant.

## Pruebas

```bash
pruebas/todas.sh
```

28 comprobaciones sobre las tres funciones puras que concentran los fallos que
más caro salieron: el recorte del historial, la rotación de la cadena y la
traducción del esfuerzo. No hace falta pytest ni tener Home Assistant instalado
— las pruebas leen `conversation.py` y extraen esas funciones del árbol de
sintaxis, así que corren contra el código real: si alguien lo edita de más, se
rompen.

## Herramientas

`herramientas/probar-cloudflare.py` mide Cloudflare Workers AI por fuera de Home
Assistant: latencia hasta el primer byte y su dispersión, si el modelo llama
herramientas correctamente, si pega el caché y cuántas «neuronas» cuesta cada
petición. Lee `CF_ACCOUNT_ID` y `CF_API_TOKEN` del entorno.

```bash
CF_ACCOUNT_ID=... CF_API_TOKEN=... python3 herramientas/probar-cloudflare.py
```

## Créditos

Fork de [HunorLaczko/ha-groq-cloud-api](https://github.com/HunorLaczko/ha-groq-cloud-api).
Se conservan el dominio, la estructura y las correcciones originales.
