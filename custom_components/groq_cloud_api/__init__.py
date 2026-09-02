"""Alta y baja de la entrada de la integración de Groq.

Frontera de este módulo: crear el cliente que va a usar todo el resto y montar
la plataforma de conversación. Nada de lógica de turno, nada de decisiones: eso
vive en `conversation.py` y en los módulos de decisión.
"""

from __future__ import annotations

from dataclasses import dataclass

import groq

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import CONF_MAX_RETRIES, DOMAIN, LOGGER

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class Clientes:
    """Los clientes que necesita la entrada; hoy uno solo.

    Groq multiplexa por el parámetro `model` de cada petición, así que el
    modelo principal, los eslabones de la cadena de respaldo y los dos
    enrutadores comparten esta misma conexión: tres modelos distintos no
    necesitan tres clientes. Se conserva el dataclass en vez de guardar el
    cliente pelado para no cambiarle la forma a `runtime_data`.
    """

    groq: groq.AsyncClient


type GroqConfigEntry = ConfigEntry[Clientes]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """No hace nada: la integración se configura solo por entrada."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Crea el cliente de Groq y monta la plataforma de conversación."""
    LOGGER.debug("Configurando la entrada %s", entry.entry_id)

    entry.runtime_data = Clientes(
        groq=groq.AsyncGroq(
            api_key=entry.data[CONF_API_KEY],
            # El httpx compartido de Home Assistant, no uno propio: así el pool
            # de conexiones y el cierre ordenado los maneja el core.
            http_client=get_async_client(hass),
            # Los reintentos del SDK van en 0 por defecto a propósito. Acá el
            # reintento que sirve es rotar de modelo —cada uno tiene su propia
            # ventana de 8000 tokens por minuto—, no repetirle al mismo, que
            # solo agrega latencia sobre una ventana que sigue ocupada.
            max_retries=entry.options.get(CONF_MAX_RETRIES, 0),
        )
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: GroqConfigEntry) -> None:
    """Recarga la entrada entera cuando cambian las opciones.

    Recargar es más caro que releer, pero deja una sola forma de que el estado
    quede viejo: ninguna. El cliente se recrea con el `max_retries` nuevo, la
    entidad vuelve a calcular si soporta CONTROL y el registro de enfriamiento
    arranca vacío, que es exactamente lo que se quiere después de cambiarle la
    cadena de modelos.
    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Descarga la plataforma de conversación."""
    LOGGER.debug("Descargando la entrada %s", entry.entry_id)

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
