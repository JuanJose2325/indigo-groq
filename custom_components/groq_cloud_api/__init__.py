"""Groq Cloud API integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import groq

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_CF_ACCOUNT,
    CONF_CF_MODEL,
    CONF_CF_TOKEN,
    CONF_MAX_RETRIES,
    DOMAIN,
    LOGGER,
)

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

@dataclass
class Clientes:
    """Los proveedores disponibles para atender una petición.

    Cloudflare Workers AI expone un endpoint compatible con OpenAI, así que el
    mismo SDK sirve cambiándole la base_url. Sirve de válvula de desborde: no
    tiene límite de tokens por minuto, así que cuando Groq rechaza por cupo se
    puede seguir con EL MISMO modelo en vez de degradar a uno más tonto.
    """

    groq: groq.AsyncClient
    http: Any
    cf_cuenta: str | None = None
    cf_ficha: str | None = None

    @property
    def hay_cf(self) -> bool:
        """Si falta cualquiera de los dos datos, el respaldo no existe."""
        return bool(self.cf_cuenta and self.cf_ficha)


type GroqConfigEntry = ConfigEntry[Clientes]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Groq Cloud API."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Set up Groq Cloud API from a config entry."""
    LOGGER.debug("Setting up %s", entry)

    client = groq.AsyncGroq(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        max_retries=entry.options.get(CONF_MAX_RETRIES, 0),
    )

    cuenta = (entry.options.get(CONF_CF_ACCOUNT) or "").strip()
    ficha = (entry.options.get(CONF_CF_TOKEN) or "").strip()
    if cuenta and ficha:
        LOGGER.debug("Respaldo de Cloudflare activo con el modelo %s",
                     entry.options.get(CONF_CF_MODEL))

    entry.runtime_data = Clientes(
        groq=client,
        http=get_async_client(hass),
        cf_cuenta=cuenta or None,
        cf_ficha=ficha or None,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: GroqConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Unload a config entry."""
    LOGGER.debug("Unloading %s", entry)

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
