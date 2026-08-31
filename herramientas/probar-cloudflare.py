#!/usr/bin/env python3
"""Mide si Cloudflare Workers AI sirve para Indigo.

Responde las cuatro preguntas que hoy no tienen respuesta pública:

  1. ¿Cuánto tarda en empezar a hablar (TTFB) y cuánta VARIANZA tiene?
     La varianza importa más que la mediana: un promedio lindo con picos de
     10 s se siente peor que algo constante y mediocre.
  2. ¿Funciona el tool calling en el endpoint compatible con OpenAI?
     Es EL riesgo del proyecto: sin herramientas no hay domótica.
  3. ¿El caché de prompt pega de verdad? Se mira `cached_tokens`.
  4. ¿Cuántas neurons cuesta cada petición? De ahí sale cuántas entran en las
     10.000 diarias.

Uso:
    export CF_ACCOUNT_ID=...
    export CF_API_TOKEN=...
    python3 probar.py                    # todas las pruebas
    python3 probar.py --modelo @cf/...   # otro modelo
    python3 probar.py --repeticiones 10  # más muestras para la varianza

El token necesita permiso de lectura de Workers AI. Nada de esto escribe en la
cuenta: son solo inferencias.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time, urllib.error, urllib.request

MODELO = "@cf/qwen/qwen3.8-27b"
TIEMPO_MAXIMO = 120

# Frases parecidas a las que Juan le dice a Indigo, no textos de laboratorio.
FRASES = [
    "Hola, ¿todo bien?",
    "Decime en una frase qué es la inflación.",
    "Recomendame una receta fácil con pan, tomate y queso.",
    "¿Cuál es la diferencia entre un perfume floral y uno cítrico?",
]

# Una herramienta con la misma forma que las que inyecta Home Assistant.
HERRAMIENTAS = [{
    "type": "function",
    "function": {
        "name": "HassTurnOn",
        "description": "Enciende un dispositivo o aparato de la casa.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Nombre del dispositivo, por ejemplo 'luz'"},
                "area": {"type": "string",
                         "description": "Habitación, por ejemplo 'dormitorio'"},
            },
            "required": ["name"],
        },
    },
}]

SISTEMA = (
    "Eres un asistente de voz. Respondes en español rioplatense, breve y en "
    "prosa, sin listas ni markdown. Cuando el usuario pide encender algo de la "
    "casa, usas la herramienta correspondiente."
)


def _url(cuenta: str) -> str:
    return (f"https://api.cloudflare.com/client/v4/accounts/{cuenta}"
            f"/ai/v1/chat/completions")


def pedir(cuenta, token, cuerpo, sesion=None, stream=False):
    """Una petición. Devuelve (ttfb_s, total_s, texto, uso, error)."""
    cabeceras = {"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"}
    if sesion:
        # Enruta a la misma instancia: es lo que hace que el caché pegue.
        cabeceras["x-session-affinity"] = sesion
    datos = json.dumps({**cuerpo, "stream": stream}).encode()
    req = urllib.request.Request(_url(cuenta), data=datos, headers=cabeceras)

    t0 = time.monotonic()
    ttfb = None
    partes: list[str] = []
    uso = None
    try:
        with urllib.request.urlopen(req, timeout=TIEMPO_MAXIMO) as r:
            if not stream:
                cuerpo_resp = json.loads(r.read().decode())
                ttfb = time.monotonic() - t0
                elec = (cuerpo_resp.get("choices") or [{}])[0]
                msg = elec.get("message") or {}
                texto = msg.get("content") or ""
                if msg.get("tool_calls"):
                    texto += " [tool_calls: " + json.dumps(
                        msg["tool_calls"], ensure_ascii=False) + "]"
                return ttfb, time.monotonic() - t0, texto, \
                    cuerpo_resp.get("usage"), None
            for linea in r:
                linea = linea.decode().strip()
                if not linea.startswith("data:"):
                    continue
                carga = linea[5:].strip()
                if carga == "[DONE]":
                    break
                try:
                    trozo = json.loads(carga)
                except json.JSONDecodeError:
                    continue
                if trozo.get("usage"):
                    uso = trozo["usage"]
                delta = ((trozo.get("choices") or [{}])[0].get("delta") or {})
                pedazo = delta.get("content") or ""
                if pedazo:
                    if ttfb is None:
                        ttfb = time.monotonic() - t0
                    partes.append(pedazo)
    except urllib.error.HTTPError as e:
        return None, time.monotonic() - t0, "", None, \
            f"HTTP {e.code}: {e.read().decode()[:400]}"
    except Exception as e:  # noqa: BLE001
        return None, time.monotonic() - t0, "", None, f"{type(e).__name__}: {e}"
    return ttfb, time.monotonic() - t0, "".join(partes), uso, None


def titulo(t):
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default=MODELO)
    p.add_argument("--repeticiones", type=int, default=5)
    args = p.parse_args()

    cuenta = os.environ.get("CF_ACCOUNT_ID")
    token = os.environ.get("CF_API_TOKEN")
    if not cuenta or not token:
        print("Falta CF_ACCOUNT_ID o CF_API_TOKEN en el entorno.\n"
              "  export CF_ACCOUNT_ID=...\n  export CF_API_TOKEN=...")
        return 2

    print(f"Modelo: {args.modelo}")
    base = {"model": args.modelo, "max_tokens": 500}

    # --- 1. LATENCIA Y VARIANZA -------------------------------------------
    titulo("1. Latencia (streaming) — lo que importa es la dispersión")
    ttfbs: list[float] = []
    for i in range(args.repeticiones):
        frase = FRASES[i % len(FRASES)]
        ttfb, total, texto, uso, err = pedir(
            cuenta, token,
            {**base, "messages": [{"role": "system", "content": SISTEMA},
                                  {"role": "user", "content": frase}]},
            stream=True)
        if err:
            print(f"  {i + 1}. ERROR {err}")
            continue
        if ttfb is None:
            print(f"  {i + 1}. sin contenido (¿todo el presupuesto en pensar?)")
            continue
        ttfbs.append(ttfb)
        tps = (len(texto) / 4) / max(total - ttfb, 1e-6)
        print(f"  {i + 1}. TTFB {ttfb:5.2f} s | total {total:5.2f} s | "
              f"~{tps:5.1f} tok/s | {len(texto):4d} car. | {frase[:34]}")

    if ttfbs:
        print(f"\n  mediana {statistics.median(ttfbs):.2f} s | "
              f"mín {min(ttfbs):.2f} | MÁX {max(ttfbs):.2f} | "
              f"dispersión {max(ttfbs) - min(ttfbs):.2f} s")
        print("  Referencia: el TTS de cara-tts empieza a sonar a ~0,45 s.")
        print("  Con TTFB > 2 s la conversación deja de sentirse fluida.")

    # --- 2. TOOL CALLING (el riesgo del proyecto) -------------------------
    titulo("2. Tool calling — sin esto no hay domótica")
    ttfb, total, texto, uso, err = pedir(
        cuenta, token,
        {**base, "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": "Prendé la luz del dormitorio."}],
         "tools": HERRAMIENTAS, "tool_choice": "auto"})
    if err:
        print(f"  ✗ FALLA: {err}")
        print("  → Si rechaza el parámetro 'tools', Cloudflare queda descartado.")
    else:
        print(f"  respuesta en {total:.2f} s:\n    {texto[:400]}")
        print("  ✓ ACEPTA 'tools'" if "tool_calls" in texto
              else "  ⚠ aceptó el parámetro pero NO llamó la herramienta "
                   "(contestó en prosa)")

    # --- 3. CACHÉ ---------------------------------------------------------
    titulo("3. Caché de prompt — dos idénticas con la misma sesión")
    relleno = ("Contexto de la casa. " * 400)  # prefijo largo y estable
    cuerpo = {**base, "messages": [
        {"role": "system", "content": SISTEMA + "\n" + relleno},
        {"role": "user", "content": "Decime solo la palabra listo."}]}
    for intento in (1, 2):
        _, total, _, uso, err = pedir(cuenta, token, cuerpo, sesion="indigo-1")
        if err:
            print(f"  {intento}. ERROR {err}")
            continue
        det = (uso or {}).get("prompt_tokens_details") or {}
        neuronas = (uso or {}).get("neurons")
        print(f"  {intento}. entrada {(uso or {}).get('prompt_tokens')} | "
              f"cacheados {det.get('cached_tokens')} | "
              f"neurons {neuronas} | {total:.2f} s")
        if neuronas:
            print(f"      → a este costo entran {int(10000 / neuronas)} "
                  f"peticiones en las 10.000 diarias")
    print("  Si en la 2ª 'cacheados' sigue en 0 o None, el caché no está pegando.")
    print("  ⚠ OJO: este prompt ronda los 2.500 tokens. El de Home Assistant\n"
          "  ronda los 4.500, así que el costo real es AÚN MAYOR que este.")

    # --- 4. COSTO ---------------------------------------------------------
    titulo("4. Uso declarado (para estimar neurons)")
    _, _, _, uso, err = pedir(
        cuenta, token,
        {**base, "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": FRASES[1]}]})
    print(f"  {json.dumps(uso, ensure_ascii=False, indent=2) if uso else err}")
    print("\n  El consumo real de neurons se ve en el panel de Cloudflare,\n"
          "  en Workers AI. Comparalo contra las 10.000 diarias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
