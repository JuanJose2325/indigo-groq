"""La UI de la integración: alta de la entrada y formulario de opciones.

Frontera de este módulo: todo lo que el usuario ve y toca en Home Assistant.
Acá no hay ni una decisión de tiempo de ejecución; lo único que se decide es qué
se le ofrece al usuario y qué se guarda en `options`. La interpretación de esas
opciones vive en `enrutadores.py` y en `conversation.py`.

La normalización de lo que se guarda (`_normalizar_opciones`) es una función PURA
y de nivel superior a propósito: el arnés de `pruebas/` la carga leyendo el AST de
este archivo, sin ejecutar ni un import, y así la compatibilidad con las entradas
viejas queda con prueba en vez de con buena voluntad.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import groq
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
)

from .const import (
    CONF_CASA_ROUTER_EFFORT,
    CONF_CASA_ROUTER_ENABLED,
    CONF_CASA_ROUTER_MODEL,
    CONF_CASA_ROUTER_THRESHOLD,
    CONF_CHAT_MODEL,
    CONF_HISTORY_BUDGET,
    CONF_MAX_RETRIES,
    CONF_MAX_TOKENS,
    CONF_MODEL_CHAIN,
    CONF_MODEL_COOLDOWN,
    CONF_PROMPT,
    CONF_RAZON_ROUTER_EFFORT,
    CONF_RAZON_ROUTER_ENABLED,
    CONF_RAZON_ROUTER_MODEL,
    CONF_RAZON_ROUTER_THRESHOLD,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_NAME,
    DEFAULT_OPTIONS,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HISTORY_BUDGET,
    RECOMMENDED_MAX_RETRIES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MODEL_COOLDOWN,
    RECOMMENDED_ROUTER_ENABLED,
    RECOMMENDED_ROUTER_MODEL,
    RECOMMENDED_ROUTER_THRESHOLD,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .razonamiento import _esfuerzo_inicial, _vocabulario_de

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)


async def async_fetch_models(api_key: str, hass: HomeAssistant) -> list[str]:
    """Los ids de modelos que ofrece la API, ordenados; cliente efímero sobre el httpx de HA."""
    client = groq.AsyncGroq(api_key=api_key, http_client=get_async_client(hass))
    response = await client.models.list()
    model_ids = sorted([model.id for model in response.data if model.id])
    LOGGER.debug("Modelos disponibles: %s", model_ids)
    return model_ids


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Valida la clave de API pidiendo la lista de modelos; PermissionDenied -> UnauthorizedError."""
    try:
        await async_fetch_models(data[CONF_API_KEY], hass)
    except groq.PermissionDeniedError:
        # Una clave que existe pero no tiene el permiso de listar modelos no es
        # una clave inválida: distinguirlo cambia el mensaje que ve el usuario.
        raise UnauthorizedError


class GroqConfigFlow(ConfigFlow, domain=DOMAIN):
    """Alta de la entrada: solo pide la clave de API."""

    VERSION = 1
    # Sube a 3 porque cambian las claves de options: mueren las cinco viejas
    # (las tres cf_* del segundo proveedor que se sacó, reasoning_effort_chain y
    # supports_reasoning) y nacen las ocho de los dos enrutadores.
    MINOR_VERSION = 3

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Pide la clave de API, la valida y crea la entrada con DEFAULT_OPTIONS."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        errors: dict[str, str] = {}

        self._async_abort_entries_match(user_input)
        try:
            await validate_input(self.hass, user_input)
        except groq.APIConnectionError:
            errors["base"] = "cannot_connect"
        except groq.AuthenticationError:
            errors["base"] = "invalid_auth"
        except UnauthorizedError:
            errors["base"] = "unauthorized"
        except Exception:
            LOGGER.exception("Excepción inesperada validando la clave de API")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title=DEFAULT_NAME,
                data=user_input,
                options=DEFAULT_OPTIONS,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Devuelve el flujo de opciones (sin argumentos, estilo HA moderno)."""
        return GroqOptionsFlow()


class GroqOptionsFlow(OptionsFlow):
    """Formulario de opciones con las tres filas de modelo + esfuerzo."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Muestra el formulario o guarda las opciones ya normalizadas."""
        if user_input is not None:
            return self.async_create_entry(
                title="", data=_normalizar_opciones(user_input)
            )

        api_key = self.config_entry.data.get(CONF_API_KEY)
        try:
            available_models = await async_fetch_models(api_key, self.hass)
        except groq.GroqError:
            # Que la API no conteste no puede dejar al usuario sin poder tocar
            # sus opciones: el formulario se muestra igual, degradado a texto
            # libre en los campos de modelo.
            LOGGER.warning(
                "No pude traer la lista de modelos de Groq; los campos de modelo "
                "van como texto libre"
            )
            available_models = []

        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                self._build_options_schema(options, available_models)
            ),
        )

    def _build_options_schema(
        self,
        options: dict[str, Any] | MappingProxyType[str, Any],
        available_models: list[str],
    ) -> dict:
        """Arma el schema; cada desplegable de esfuerzo sale de la familia del modelo de SU fila."""
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        model_options: list[SelectOptionDict] = [
            SelectOptionDict(label=model, value=model) for model in available_models
        ]

        # Las tres filas modelo + esfuerzo. Cada una resuelve su modelo primero
        # porque el vocabulario del desplegable de esfuerzo depende de la FAMILIA
        # de ESE modelo, no de la del principal: la cadena y los enrutadores
        # pueden mezclar familias, y ofrecerle "medium" a un Qwen es un 400 seguro.
        modelo_principal = self._modelo_de_la_fila(
            options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL), available_models
        )
        modelo_casa = self._modelo_de_la_fila(
            options.get(CONF_CASA_ROUTER_MODEL, RECOMMENDED_ROUTER_MODEL),
            available_models,
        )
        modelo_razon = self._modelo_de_la_fila(
            options.get(CONF_RAZON_ROUTER_MODEL, RECOMMENDED_ROUTER_MODEL),
            available_models,
        )

        # El principal es el único que piensa de verdad, así que su preselección
        # es el esfuerzo medio de la familia (barato=False). Los enrutadores
        # tienen 150 tokens de techo y solo devuelven un JSON de tres claves: con
        # el esfuerzo alto se les va el presupuesto pensando y vuelven vacíos,
        # que en su tabla de fallos significa "conservar herramientas" y "no
        # razonar". Por eso barato=True.
        selector_esf_principal, esf_principal = self._selector_de_esfuerzo(
            modelo_principal, options.get(CONF_REASONING_EFFORT), barato=False
        )
        selector_esf_casa, esf_casa = self._selector_de_esfuerzo(
            modelo_casa, options.get(CONF_CASA_ROUTER_EFFORT), barato=True
        )
        selector_esf_razon, esf_razon = self._selector_de_esfuerzo(
            modelo_razon, options.get(CONF_RAZON_ROUTER_EFFORT), barato=True
        )

        selector_modelo = self._selector_de_modelo(model_options)
        # La cadena de respaldo es el mismo listado pero con selección múltiple y
        # ordenada: los límites de Groq son POR MODELO (8000 tokens por minuto
        # cada uno), así que cada eslabón suma su propia ventana.
        if model_options:
            selector_cadena: Any = SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            )
        else:
            selector_cadena = TextSelector()

        # Umbral de confianza de los dos enrutadores. El paso de 0,05 es para que
        # se pueda mover con el deslizador sin pelearse con los decimales.
        selector_umbral = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))

        return {
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": options.get(
                        CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                    )
                },
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                description={
                    "suggested_value": (
                        [v]
                        if isinstance(v := options.get(CONF_LLM_HASS_API), str)
                        else v
                    )
                },
            ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
            vol.Required(
                CONF_CHAT_MODEL,
                default=modelo_principal,
            ): selector_modelo,
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": esf_principal},
                default=esf_principal,
            ): selector_esf_principal,
            vol.Required(
                CONF_CASA_ROUTER_ENABLED,
                default=options.get(
                    CONF_CASA_ROUTER_ENABLED, RECOMMENDED_ROUTER_ENABLED
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_CASA_ROUTER_MODEL,
                default=modelo_casa,
            ): selector_modelo,
            vol.Optional(
                CONF_CASA_ROUTER_EFFORT,
                description={"suggested_value": esf_casa},
                default=esf_casa,
            ): selector_esf_casa,
            vol.Required(
                CONF_CASA_ROUTER_THRESHOLD,
                default=options.get(
                    CONF_CASA_ROUTER_THRESHOLD, RECOMMENDED_ROUTER_THRESHOLD
                ),
            ): selector_umbral,
            vol.Required(
                CONF_RAZON_ROUTER_ENABLED,
                default=options.get(
                    CONF_RAZON_ROUTER_ENABLED, RECOMMENDED_ROUTER_ENABLED
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_RAZON_ROUTER_MODEL,
                default=modelo_razon,
            ): selector_modelo,
            vol.Optional(
                CONF_RAZON_ROUTER_EFFORT,
                description={"suggested_value": esf_razon},
                default=esf_razon,
            ): selector_esf_razon,
            vol.Required(
                CONF_RAZON_ROUTER_THRESHOLD,
                default=options.get(
                    CONF_RAZON_ROUTER_THRESHOLD, RECOMMENDED_ROUTER_THRESHOLD
                ),
            ): selector_umbral,
            vol.Optional(
                CONF_MODEL_CHAIN,
                description={"suggested_value": options.get(CONF_MODEL_CHAIN, [])},
            ): selector_cadena,
            vol.Required(
                CONF_MODEL_COOLDOWN,
                default=options.get(CONF_MODEL_COOLDOWN, RECOMMENDED_MODEL_COOLDOWN),
            ): int,
            vol.Required(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            ): int,
            vol.Required(
                CONF_HISTORY_BUDGET,
                default=options.get(CONF_HISTORY_BUDGET, RECOMMENDED_HISTORY_BUDGET),
            ): int,
            vol.Required(
                CONF_TOP_P,
                default=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_MAX_RETRIES,
                default=options.get(CONF_MAX_RETRIES, RECOMMENDED_MAX_RETRIES),
            ): int,
        }

    def _modelo_de_la_fila(
        self, preferido: str, available_models: list[str]
    ) -> str:
        """El modelo que se preselecciona en una fila, garantizando que esté en el desplegable."""
        # Un valor por defecto que no está entre las opciones deja el desplegable
        # en blanco y el usuario no entiende por qué. Si el modelo guardado ya no
        # existe en la cuenta (Groq depreca modelos seguido: qwen/qwen3-32b murió
        # en junio de 2026), se cae al primero disponible y se avisa.
        if not available_models or preferido in available_models:
            return preferido
        LOGGER.warning(
            "El modelo '%s' ya no está disponible; preselecciono '%s'",
            preferido,
            available_models[0],
        )
        return available_models[0]

    def _selector_de_modelo(self, model_options: list[SelectOptionDict]) -> Any:
        """Desplegable de modelos, o texto libre si la API no contestó la lista."""
        if not model_options:
            return str
        return SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
            )
        )

    def _selector_de_esfuerzo(
        self, modelo: str, guardado: Any, barato: bool
    ) -> tuple[Any, str]:
        """El selector de esfuerzo de una fila y su valor preseleccionado."""
        vocabulario = _vocabulario_de(modelo)
        if vocabulario:
            # OJO con el preseleccionado: nunca puede ser vocabulario[0] a ciegas.
            # En Qwen el orden de la familia empieza en "none" pero el valor
            # histórico "default" es el MÁXIMO, y preseleccionar el máximo era la
            # misma trampa del campo vacío: el modelo quema los 1200 tokens de
            # max_tokens pensando y vuelve sin nada que decir. `_esfuerzo_inicial`
            # devuelve el guardado si sigue siendo válido y, si no, el recomendado
            # de la familia; jamás el máximo.
            #
            # Y "sin configurar" TIENE que ser elegible, no solo alcanzable
            # borrando la clave a mano. Es un tercer estado real —ni "none" ni un
            # esfuerzo concreto— en el que se manda reasoning_format="hidden" sin
            # reasoning_effort y decide el modelo; es el estado en el que corre
            # una instalación que nunca tocó el campo. Sin esta opción en el
            # desplegable, abrir el formulario y guardar sin tocar nada le
            # escribía "default" —el máximo de Qwen— a quien no tenía la clave:
            # el fallo silencioso de vuelta, entrando por la interfaz.
            opciones = [SelectOptionDict(label="(sin configurar)", value="")]
            opciones += [
                SelectOptionDict(label=valor, value=valor) for valor in vocabulario
            ]
            return (
                SelectSelector(
                    SelectSelectorConfig(
                        options=opciones,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                _esfuerzo_inicial(modelo, guardado, barato=barato),
            )
        # Familia desconocida: texto libre y se respeta lo que el usuario haya
        # escrito. Acá no se pasa por `_esfuerzo_inicial` porque devolvería ""
        # y le borraría el valor cada vez que abre y cierra el formulario.
        return TextSelector(), guardado or ""


def _normalizar_opciones(user_input: dict) -> dict:
    """Limpia lo que se va a guardar: borra las claves muertas y las vacías que significan 'sin valor'."""
    # Las claves van como literales y no como constantes importadas por dos
    # motivos. Uno: las cinco ya no tienen constante —las tres cf_* eran del
    # segundo proveedor que se sacó, y reasoning_effort_chain y
    # supports_reasoning se reemplazaron por la familia del modelo—, se borraron
    # de const.py y solo sobreviven acá como basura que hay que ignorar sin
    # romper. Dos: esta función la carga el arnés leyendo el AST de este archivo,
    # sin ejecutar los imports, así que cada nombre de afuera sería un NameError
    # en tiempo de prueba. Sin nombres externos, la comprobación no se puede caer
    # en silencio.
    muertas = (
        "cf_account_id",
        "cf_api_token",
        "cf_model",
        "reasoning_effort_chain",
        "supports_reasoning",
    )
    limpio = {}
    for clave, valor in user_input.items():
        if clave in muertas:
            continue
        limpio[clave] = valor

    # `llm_hass_api` vacío se BORRA en vez de guardarse como lista vacía. Lo que
    # se le pasa a `async_provide_llm_data` no es solo la lista de herramientas:
    # ahí también viaja el api_prompt con el volcado YAML de todas las entidades
    # expuestas y su área. Una lista vacía guardada es indistinguible de "sin
    # API" al leerla, pero ensucia las options; se saca y listo.
    if not limpio.get("llm_hass_api"):
        limpio.pop("llm_hass_api", None)

    # Los tres campos de esfuerzo se dejan TAL CUAL, incluso vacíos. Vacío no es
    # un valor inválido: significa "no configurado", y en ese caso el modelo se
    # llama con reasoning_format="hidden" y sin reasoning_effort, que es distinto
    # de mandarle el máximo. Barrerlos acá reintroduciría el fallo silencioso.
    return limpio


class UnauthorizedError(HomeAssistantError):
    """La clave de API es válida pero no tiene los permisos necesarios."""
