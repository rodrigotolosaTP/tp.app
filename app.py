import io
import re
from dataclasses import dataclass
from datetime import date
import pandas as pd
import streamlit as st
import requests
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pdfplumber
from typing import Sequence

# === Graficador CSV libre (del módulo adjunto) ===
from contexto_semestral_25 import (
    leer_csv_con_fallback,
    intentar_parsear_datetime,
    intentar_a_numero,
    construir_x_compuesto,
    calcular_y_limites,
    agrupar_topk_ancho,
    filtrar_tc,          
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import unicodedata, re

import tempfile, os
import pdfplumber
# ---------- BCRA API helpers ----------
import json
import numpy as np
import pandas as pd
import requests

BCRA_BASE_URL = "https://api.estadisticasbcra.com"

@st.cache_data(show_spinner=False, ttl=1800)
def bcra_fetch(endpoint: str, token: str) -> list[dict]:
    """
    Llama a api.estadisticasbcra.com/<endpoint> con el token BEARER.
    Devuelve la lista JSON tal cual (esperado: [{'d': 'YYYY-MM-DD', 'v': <valor>}, ...])
    """
    ep = endpoint.strip()
    if not ep.startswith("/"):
        ep = "/" + ep
    url = BCRA_BASE_URL + ep
    headers = {"Authorization": f"BEARER {token}"}
    r = requests.get(url, headers=headers, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} — {r.text[:300]}")
    try:
        return r.json()
    except json.JSONDecodeError:
        raise RuntimeError("La respuesta no es JSON válido.")

def bcra_json_to_df(items: list[dict], label: str) -> pd.DataFrame:
    """
    Convierte [{'d':'YYYY-MM-DD','v':valor},...] a DataFrame con columnas ['Fecha', <label>]
    """
    df = pd.DataFrame(items)
    if not {"d", "v"}.issubset(df.columns):
        raise ValueError("Formato inesperado en la respuesta (faltan 'd' o 'v').")
    df["Fecha"] = pd.to_datetime(df["d"], errors="coerce")
    df[label] = pd.to_numeric(df["v"], errors="coerce")
    return df[["Fecha", label]].dropna()

def bcra_merge_series(series_map: dict[str, str], token: str) -> pd.DataFrame:
    """
    series_map: {nombre_serie: endpoint}
    Descarga y hace merge outer por 'Fecha' de todas las series pedidas.
    """
    merged = None
    for name, ep in series_map.items():
        data = bcra_fetch(ep, token)
        df = bcra_json_to_df(data, name)
        merged = df if merged is None else merged.merge(df, on="Fecha", how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("Fecha").reset_index(drop=True)

import re

_NUMPAIR = re.compile(r"""
^\s*
([0-9.\s]+(?:,\d{2})?)   # número 1 con miles y decimal opcional
\s+                      # separador (espacios/tab)
([0-9.\s]+(?:,\d{2})?)   # número 2
\s*$""", re.X)

def split_joined_numbers_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si alguna columna tiene muchos valores con 'dos números pegados',
    la divide en <col>_1 y <col>_2 ya normalizados.
    """
    target = None
    best_ratio = 0.0
    for c in df.columns[::-1]:  # pruebo de derecha a izquierda
        s = df[c].astype(str)
        ratio = s.str.match(_NUMPAIR).mean()
        if ratio > 0.4 and ratio > best_ratio:  # umbral heurístico
            target, best_ratio = c, ratio

    if target is None:
        return df

    s = df[target].astype(str)
    n1 = s.str.replace(_NUMPAIR, r"\1", regex=True)
    n2 = s.str.replace(_NUMPAIR, r"\2", regex=True)
    df[target + "_1"] = normalizar_numeros_columna(n1)
    df[target + "_2"] = normalizar_numeros_columna(n2)
    return df.drop(columns=[target])

def _to_points(page_w, page_h, top_pct, left_pct, bottom_pct, right_pct):
    """Convierte porcentajes (0-100) de la página a puntos (y0, x0, y1, x1)."""
    return [
        page_h * (top_pct/100.0),
        page_w * (left_pct/100.0),
        page_h * (bottom_pct/100.0),
        page_w * (right_pct/100.0),
    ]

def _cols_rel_to_abs(area, rel_pcts):
    """De porcentajes relativos al ancho del área -> lista de x en puntos para 'columns' de Tabula."""
    if not rel_pcts: 
        return None
    x0, x1 = area[1], area[3]
    w = x1 - x0
    return [x0 + (p/100.0)*w for p in rel_pcts]

def extraer_tabla_split_2columnas(pdf_bytes, page, *,
                                  split_x_pct=50.0,
                                  top_pct=12.0, bottom_pct=96.0,
                                  left_cols_pct=None, right_cols_pct=None,
                                  flavor_stream=True):
    """
    Divide la página en 2 áreas: izquierda (texto) y derecha (números).
    Usa Tabula en modo stream (por defecto) y permite pasar 'columns' relativos.
    Devuelve [df_izq, df_der] (las hojas pueden variar según PDF).
    """
    import tabula
    # Guardamos a archivo temporal porque tabula-java lee desde ruta
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as p:
            pg = p.pages[page-1]
            W, H = pg.width, pg.height

        # Definimos áreas izquierda/derecha en puntos
        area_left  = _to_points(W, H, top_pct, 2.0, bottom_pct, split_x_pct)
        area_right = _to_points(W, H, top_pct, split_x_pct, bottom_pct, 98.0)

        cols_left  = _cols_rel_to_abs(area_left,  left_cols_pct  or [])
        cols_right = _cols_rel_to_abs(area_right, right_cols_pct or [])

        dfs = []
        for area, cols in [(area_left, cols_left), (area_right, cols_right)]:
            part = tabula.read_pdf(
                tmp_path,
                pages=page,
                area=area,
                stream=flavor_stream, lattice=not flavor_stream,
                guess=False,
                multiple_tables=False,
                columns=cols,
                pandas_options={"dtype": str}
            )
            if part:
                dfs.append(part[0])
            else:
                dfs.append(pd.DataFrame())
        return dfs  # [df_izq, df_der]
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

def normalizar_numeros_columna(s: pd.Series) -> pd.Series:
    """
    Convierte strings tipo '1.234.567,89' o '1,234,567.89' a float.
    Si falla, NaN.
    """
    def _clean(x):
        if pd.isna(x): return None
        t = str(x).strip()
        # retirar espacios y separadores invisibles
        t = t.replace("\u00a0", "").replace("\u202f", "")
        # si hay coma como decimal y punto como miles
        if t.count(",") == 1 and t.count(".") >= 1 and t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        # si el decimal ya es punto y hay comas de miles
        elif t.count(".") == 1 and t.count(",") >= 1 and t.rfind(".") > t.rfind(","):
            t = t.replace(",", "")
        else:
            # fallback: eliminar miles comunes
            if t.count(".") > 1 and "," not in t:
                t = t.replace(".", "")
            if t.count(",") > 1 and "." not in t:
                t = t.replace(",", "")
            t = t.replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None
    return s.apply(_clean)

def _norm_txt(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

def auto_paginas_por_keywords(pdf_bytes: bytes, keywords: list[str], max_por_keyword: int = 1) -> list[int]:
    """Devuelve páginas (1-based) donde aparezcan palabras clave (sin tildes, insensitive)."""
    import pdfplumber
    kw = [_norm_txt(k) for k in keywords if k.strip()]
    pags = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            nt = _norm_txt(t)
            if any(k in nt for k in kw):
                pags.append(i)
    # limitar repeticiones por keyword (heurística simple)
    return sorted(list(dict.fromkeys(pags)))[: max_por_keyword * max(1, len(kw))]

def postprocesar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia tabla: encabeza con primera fila si aplica, columnas vacías, números."""
    out = df.copy()

    # si la primera fila no tiene NaN, usarla de encabezado
    if out.shape[0] > 1 and out.iloc[0].isna().sum() == 0:
        out.columns = out.iloc[0].astype(str).str.strip()
        out = out.iloc[1:].reset_index(drop=True)

    # drop columnas 100% vacías o 'Unnamed'
    out = out.loc[:, [c for c in out.columns if not (str(c).lower().startswith("unnamed"))]]
    out = out.dropna(axis=1, how="all")

    # normalización numérica amigable (miles/puntos/comas y paréntesis negativos)
    def _to_num(x):
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return pd.NA
        s = s.replace(".", "").replace(" ", "")
        s = s.replace(",", ".")
        # paréntesis contables
        if re.fullmatch(r"\(.*\)", s):
            try:
                return -float(s.strip("()"))
            except Exception:
                return pd.NA
        try:
            return float(s)
        except Exception:
            return x  # dejar como texto si no es número

    for c in out.columns:
        # solo intentar si la mayoría parece número
        col = out[c].astype(str)
        frac_num = col.str.replace(r"[.\s]", "", regex=True).str.replace(",", ".", regex=False)\
                      .str.fullmatch(r"-?\(?\d+(\.\d+)?\)?", na=False).mean()
        if frac_num > 0.6:
            out[c] = out[c].map(_to_num)

    return out


def _to_dt_like(s: pd.Series) -> pd.Series:
    """Convierte una serie a datetime mensual cuando se pueda (Period, datetime o texto)."""
    if pd.api.types.is_period_dtype(s):
        return s.dt.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    # último intento sobre texto
    return pd.to_datetime(s.astype(str), errors="coerce")

def _parse_bound(txt: str | None) -> pd.Timestamp | None:
    """Acepta 'YYYY-MM' o 'YYYY-MM-DD' y devuelve Timestamp; vacío -> None."""
    if not txt:
        return None
    txt = txt.strip()
    if not txt:
        return None
    # si viene 'YYYY-MM' le agregamos día
    if len(txt) == 7 and txt[4] == "-":
        txt = txt + "-01"
    try:
        return pd.to_datetime(txt, errors="coerce")
    except Exception:
        return None

# === Helpers para detectar CSVs de IPC y normalizar nombres ===
_IPC_REGION_CANDIDATES   = ["región", "region", "zona", "área geográfica", "area geografica", "provincia", "gba"]
_IPC_CONCEPTO_CANDIDATES = ["descripción", "descripcion", "clasificador", "división", "division",
                            "grupo", "rubro", "coicop", "título", "titulo"]
_IPC_FECHA_CANDIDATES    = ["periodo", "índice_tiempo", "indice_tiempo", "fecha", "mes"]
_IPC_VALOR_CANDIDATES    = ["ipc", "índice_ipc", "indice_ipc", "nivel", "valor", "v_m_ipc", "v_i_a_ipc"]

def _norm_txt(s: str) -> str:
    # bajar a ascii “casero” (sin dependencias)
    table = str.maketrans("ÁÉÍÓÚáéíóúÑñ", "AEIOUaeiouNn")
    return str(s).translate(table).lower().strip()

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_norm = {_norm_txt(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_txt(cand)
        # match exacto o contenido
        if key in cols_norm:
            return cols_norm[key]
        # fallback: “contiene”
        for k, orig in cols_norm.items():
            if key in k:
                return orig
    return None

def guess_ipc_columns(df: pd.DataFrame):
    """Devuelve (col_region, col_concepto, col_fecha, col_valor) si parecen existir."""
    col_region   = _pick_col(df, _IPC_REGION_CANDIDATES)
    col_concepto = _pick_col(df, _IPC_CONCEPTO_CANDIDATES)
    col_fecha    = _pick_col(df, _IPC_FECHA_CANDIDATES)
    col_valor    = _pick_col(df, _IPC_VALOR_CANDIDATES)
    return col_region, col_concepto, col_fecha, col_valor

def paginas_tienen_texto(pdf_bytes: bytes, paginas: Sequence[int]) -> bool:
    """True si al menos una de las páginas tiene palabras 'reales' (no imagen)."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for p in paginas:
                if 1 <= p <= len(pdf.pages):
                    words = pdf.pages[p-1].extract_words()
                    if words and len(words) > 3:
                        return True
        return False
    except Exception:
        return False

def leer_tablas_tabula(pdf_bytes: bytes, pages_str: str, lattice: bool, stream: bool):
    """Envuelve tabula.read_pdf con parámetros razonables."""
    import tabula
    dfs = tabula.read_pdf(
        io.BytesIO(pdf_bytes),
        pages=pages_str,
        multiple_tables=True,
        lattice=lattice,
        stream=stream,
        guess=True,
        pandas_options={"dtype": "string"}
    )
    # filtrar vacías
    dfs = [d for d in dfs if d is not None and not d.empty]
    return dfs

def leer_tablas_pdfplumber(pdf_bytes: bytes, paginas: Sequence[int]):
    """Extrae tablas simples con pdfplumber."""
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in paginas:
            if 1 <= p <= len(pdf.pages):
                page = pdf.pages[p-1]
                for tbl in page.extract_tables():
                    df = pd.DataFrame(tbl)
                    # limpia filas/cols completamente vacías
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    if not df.empty:
                        out.append(df)
    return out


def _titulo_sugerido(etiqueta: str | None, value_col: str) -> str:
    """
    Devuelve un título por defecto según la etiqueta y si el valor es ajustado u original.
    """
    sufijo = "ajustados" if value_col == "monto_ajustado_base" else "originales"
    etq = (etiqueta or "Transacciones")

    # Frases lindas según etiqueta
    mapping = {
        "Servicio prestado":      f"Facturación por servicios ({sufijo})",
        "Servicio recibido":      f"Gastos por servicios ({sufijo})",
        "Ventas":                 f"Facturación ({sufijo})",
        "Compras":                f"Compras ({sufijo})",
        "Exportaciones":          f"Exportaciones ({sufijo})",
        "Importaciones":          f"Importaciones ({sufijo})",
        "Intereses financieros":  f"Intereses financieros ({sufijo})",
        "Transacciones":          f"Montos ({sufijo})",
    }
    return mapping.get(etq, f"Montos — {etq} ({sufijo})")

def _scale_y_miles_si_corresponde(ax, serie, threshold=1_000_000):
    """Si el máximo supera 'threshold', muestra el eje Y en miles."""
    maxv = float(pd.Series(serie).max())
    if maxv >= threshold:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: f"{x/1_000:,.0f}".replace(",", "."))
        )
        ax.set_ylabel("Monto (miles)")
    else:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}".replace(",", ".")))
        ax.set_ylabel("Monto")

# Motores opcionales de extracción de tablas de PDF
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    import tabula
    _HAS_TABULA = True
except Exception:
    _HAS_TABULA = False

# =========================
# Constantes
# =========================
API_SERIES_BASE = "https://apis.datos.gob.ar/series/api/series"

# Presets de series IPC (INDEC / Datos Argentina)
SERIES_PRESETS = {
    "Nivel general — Nacional (base dic-2016)": "101.1_I2NG_2016_M_22",
    "Núcleo — Nacional (base dic-2016)":        "103.1_I2N_2016_M_15",
    "Bienes — Nacional (base dic-2016)":        "102.1_I2B_ABRI_M_15",
    "Servicios — Nacional (base dic-2016)":     "102.1_I2S_ABRI_M_18",
    "Regulados — Nacional (base dic-2016)":     "103.1_I2R_2016_M_18",
    "Estacionales — Nacional (base dic-2016)":  "103.1_I2E_2016_M_21",
}

# =========================
# Utilidades core (motor)
# =========================

# ---------- PDF → Tablas ----------
def _find_pages_with_keywords(pdf_bytes: bytes, keywords: list[str]) -> list[int]:
    """Devuelve páginas (1-based) donde aparecen palabras clave."""
    if not _HAS_PDFPLUMBER:
        return []
    pgs = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = (page.extract_text() or "").lower()
            if any(k.lower() in txt for k in keywords):
                pgs.append(i)
    return pgs

def _tabula_extract(pdf_bytes: bytes, pages: list[int] | str = "all",
                    max_pages: int = 6, java_mem: str = "2g") -> list[pd.DataFrame]:
    """Extrae tablas con tabula, limitando cantidad de páginas y asignando memoria a Java."""
    if not _HAS_TABULA:
        return []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        # Normalizamos el argumento pages y limitamos cantidad
        if isinstance(pages, list) and pages:
            pages = sorted(set(int(p) for p in pages if int(p) > 0))[:max_pages]
            pages_arg = ",".join(str(p) for p in pages)
        else:
            # si viene "all", mejor no: lo acotamos a las 10 primeras por seguridad
            pages_arg = "1-10" if pages == "all" else pages

        common = dict(
            pages=pages_arg,
            multiple_tables=True,
            pandas_options={"dtype": str},
            java_options=[f"-Xmx{java_mem}"],  # memoria para JVM (e.g., "1g", "2g")
        )
        try:
            dfs = tabula.read_pdf(tmp.name, lattice=True, **common) or []
            dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and d.shape[0] >= 2 and d.shape[1] >= 2]
            if dfs:
                return dfs
            dfs = tabula.read_pdf(tmp.name, stream=True, **common) or []
            return [d for d in dfs if isinstance(d, pd.DataFrame) and d.shape[0] >= 2 and d.shape[1] >= 2]
        except Exception:
            return []

def _pdfplumber_extract(pdf_bytes: bytes, pages: list[int] | None = None) -> list[pd.DataFrame]:
    """Extrae tablas con pdfplumber (menos preciso, pero sin Java)."""
    if not _HAS_PDFPLUMBER:
        return []
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        idxs = range(len(pdf.pages)) if not pages else [p-1 for p in pages if 1 <= p <= len(pdf.pages)]
        for i in idxs:
            page = pdf.pages[i]
            # Intento con heurística de líneas
            ts = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            }) or []
            if not ts:
                # Segundo intento “texto”
                ts = page.extract_tables() or []
            for t in ts:
                df = pd.DataFrame(t)
                if df.shape[0] >= 2 and df.shape[1] >= 2:
                    # Si la primera fila parece encabezado, promuévela
                    header = df.iloc[0].fillna("")
                    if (header.astype(str) != "").any():
                        df.columns = [str(c).strip() if c is not None else "" for c in header]
                        df = df.iloc[1:].reset_index(drop=True)
                    df.columns = [str(c).strip() or f"col{i}" for i, c in enumerate(df.columns)]
                    out.append(df.reset_index(drop=True))
    return out

def extract_pdf_tables(pdf_bytes: bytes, pages: list[int] | None) -> list[pd.DataFrame]:
    """Estrategia: Tabula → (si vacío) pdfplumber."""
    dfs = _tabula_extract(pdf_bytes, pages="all" if pages is None else pages)
    if not dfs:
        dfs = _pdfplumber_extract(pdf_bytes, pages=pages)
    return dfs

def dataframes_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Escribe varios DataFrames a un Excel (una hoja por estado) con ancho de columnas cómodo."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            sheet = name[:31]  # límite Excel
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            # Ajuste simple de ancho
            for col_idx, col in enumerate(df.columns):
                try:
                    maxw = max([len(str(col))] + df[col].astype(str).str.len().clip(upper=80).tolist())
                except Exception:
                    maxw = 12
                ws.set_column(col_idx, col_idx, min(max(12, maxw + 2), 50))
    return output.getvalue()

# ---------- Gráficos ----------
def make_chart_png(
    df: pd.DataFrame,
    chart_type: str = "line",              # "line" | "bar"
    value_col: str = "monto_ajustado_base",
    filtro_tipo: str | None = None,        # etiqueta a filtrar o None
    freq: str | None = None,               # None | "Q" | "A"
    titulo: str = "Facturación por servicios",
    threshold_miles: int = 1_000_000,      # umbral para pasar a “(miles)”
    bar_width_days: int = 25               # ancho de barras en días (para fechas)
) -> bytes:
    df_plot = df.copy()
    if filtro_tipo:
        df_plot = df_plot[df_plot.get("tipo").astype(str) == str(filtro_tipo)]

    # Agregación temporal
    serie = (df_plot
             .groupby("periodo", as_index=True)[value_col]
             .sum()
             .sort_index())

    if freq in ("Q", "A"):
        # asegurar índice datetime para resample
        s = pd.Series(serie.values, index=pd.to_datetime(serie.index))
        serie = s.resample(freq).sum()

    x = pd.to_datetime(serie.index)
    y = serie.values

    fig, ax = plt.subplots(figsize=(10, 4))

    if chart_type == "bar":
        # ancho de barra expresado en días (con fechas queda prolijo)
        ax.bar(x, y, width=bar_width_days)
    else:
        ax.plot(x, y, marker="o", linewidth=2)

    # Eje X lindo con Mes-Año
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45, ha="right")

    # Escala eje Y (miles si supera el umbral)
    _scale_y_miles_si_corresponde(ax, y, threshold=threshold_miles)

    ax.set_xlabel("Fecha")
    ax.set_title(titulo)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _norm_api_date(s: str | None) -> str | None:
    """Normaliza fechas aceptadas por la API: YYYY, YYYY-MM o YYYY-MM-DD."""
    if not s:
        return None
    return str(s).strip()

def cargar_ipc_api(
    serie_id: str = "101.1_I2NG_2016_M_22",
    start_date: str | None = None,
    end_date: str | None = None,
    fallback_without_dates: bool = True,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con (periodo, ipc) desde la API.
    Maneja errores 4xx/5xx con mensajes claros.
    """
    sd = _norm_api_date(start_date)
    ed = _norm_api_date(end_date)

    # Validación simple de rango (si ambos existen)
    try:
        if sd and ed:
            sd_dt = pd.to_datetime(sd + "-01" if len(sd) == 7 else sd)
            ed_dt = pd.to_datetime(ed + "-01" if len(ed) == 7 else ed)
            if sd_dt > ed_dt:
                raise ValueError(f"start_date ({sd}) no puede ser > end_date ({ed}).")
    except Exception:
        pass

    params = {
        "ids": serie_id,
        "format": "csv",
        "header": "ids",
        "sort": "asc",
        "limit": 5000,   # límite máximo permitido por la API
    }
    if sd: params["start_date"] = sd
    if ed: params["end_date"]   = ed

    try:
        resp = requests.get(API_SERIES_BASE, params=params, timeout=15)
        if not resp.ok:
            detalle = resp.text.strip()
            msg = f"API INDEC devolvió {resp.status_code} ({resp.reason})."
            if detalle:
                msg += f" Detalle: {detalle[:400]}"
            # Fallback sin fechas si se habilitó
            if fallback_without_dates and (sd or ed):
                resp2 = requests.get(API_SERIES_BASE, params={
                    "ids": serie_id, "format": "csv", "header": "ids", "sort": "asc", "limit": 5000
                }, timeout=15)
                if resp2.ok:
                    resp = resp2
                else:
                    raise RuntimeError(msg)
            else:
                raise RuntimeError(msg)

        # Parsear CSV en memoria
        df_raw = pd.read_csv(io.StringIO(resp.text))
        fecha_col = df_raw.columns[0]   # suele ser 'indice_tiempo'
        ipc_col   = serie_id            # la serie viene con este nombre

        df_raw[fecha_col] = pd.to_datetime(df_raw[fecha_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[fecha_col])
        df_raw["periodo"] = df_raw[fecha_col].dt.to_period("M").dt.to_timestamp()

        df_ipc = df_raw.rename(columns={ipc_col: "ipc"})[["periodo", "ipc"]].copy()
        df_ipc["ipc"] = pd.to_numeric(df_ipc["ipc"], errors="coerce")
        if df_ipc["ipc"].isna().all():
            raise RuntimeError("La API devolvió datos sin valores numéricos para 'ipc'. Verificá el ID de serie.")
        return df_ipc

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo contactar la API del INDEC: {e}")

def _parse_fecha_col(fecha_series: pd.Series, *,
                     dayfirst: bool = True,
                     fmt: str | None = None) -> pd.Series:
    """
    Parseo robusto de fechas:
      - formato explícito si se provee
      - auto-parse (dayfirst)
      - seriales de Excel (1899-12-30 base)
    """
    s = fecha_series.astype("string").str.strip()

    def _excel_serial_to_dt(x):
        try:
            val = float(str(x).replace(",", "."))
            if val <= 0 or val > 100000:
                return pd.NaT
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(val), "D")
        except Exception:
            return pd.NaT

    is_numeric_like = s.str.replace(",", ".", regex=False).str.fullmatch(r"-?\d+(\.\d+)?", na=False)
    excel_guess = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_numeric_like.mean() > 0.6:
        excel_guess = s.where(is_numeric_like).apply(_excel_serial_to_dt)

    if fmt:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
    else:
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)

    parsed = parsed.fillna(excel_guess)

    mask = parsed.isna()
    if mask.any():
        s2 = s[mask].str.replace(".", "/", regex=False)
        parsed2 = pd.to_datetime(s2, errors="coerce", dayfirst=dayfirst)
        parsed.loc[mask] = parsed2

    return parsed

def _to_period_m(fecha_series: pd.Series) -> pd.Series:
    return fecha_series.dt.to_period("M").dt.to_timestamp()

@dataclass
class ServiciosConfig:
    nombre: str
    fecha_col: str
    monto_col: str
    etiqueta: str | None = None  # opcional

def normalizar_servicios(df: pd.DataFrame, cfg: ServiciosConfig,
                         *, fecha_fmt: str | None = None, dayfirst: bool = True,
                         drop_bad_dates: bool = False, collect_errors: bool = True) -> pd.DataFrame:
    faltan = [c for c in (cfg.fecha_col, cfg.monto_col) if c not in df.columns]
    if faltan:
        raise ValueError(f"[{cfg.nombre}] Faltan columnas: {faltan}")

    out = df.copy()
    out["_fecha_dt"] = _parse_fecha_col(out[cfg.fecha_col], dayfirst=dayfirst, fmt=fecha_fmt)

    if out["_fecha_dt"].isna().any():
        malos = (out.loc[out["_fecha_dt"].isna(), cfg.fecha_col]
                   .astype(str).str.strip().value_counts().head(10))
        msg = (f"[{cfg.nombre}] Hay fechas no parseables en '{cfg.fecha_col}'. "
               f"Ejemplos: {list(malos.index)}")
        if drop_bad_dates:
            if collect_errors:
                st.warning(msg + " → Se descartarán esas filas.")
            out = out.loc[~out["_fecha_dt"].isna()].copy()
        else:
            raise ValueError(msg)

    out["_monto_float"] = (
        out[cfg.monto_col].astype(str)
          .str.replace(".", "", regex=False)
          .str.replace(",", ".", regex=False)
          .astype(float)
    )
    out["periodo"] = out["_fecha_dt"].dt.to_period("M").dt.to_timestamp()
    if cfg.etiqueta:
        out["tipo"] = cfg.etiqueta
    return out

def cargar_ipc_largo(df: pd.DataFrame, fecha_col="fecha", indice_col="ipc") -> pd.DataFrame:
    if fecha_col not in df.columns or indice_col not in df.columns:
        raise ValueError(f"IPC largo debe tener columnas '{fecha_col}' y '{indice_col}'.")
    tmp = df.copy()
    tmp[fecha_col] = _parse_fecha_col(tmp[fecha_col])
    if tmp[fecha_col].isna().any():
        raise ValueError("IPC largo: fechas no parseables.")
    tmp["periodo"] = _to_period_m(tmp[fecha_col])
    tmp = (tmp.groupby("periodo", as_index=False)[indice_col]
               .mean()
               .rename(columns={indice_col: "ipc"})
               .sort_values("periodo"))
    return tmp[["periodo","ipc"]]

def cargar_ipc_ancho(df: pd.DataFrame, ancho_tiene_header: bool, mapa_mes: dict | None) -> pd.DataFrame:
    tmp = df.copy()
    if not ancho_tiene_header:
        if tmp.shape[1] == 1 and tmp.shape[0] >= 2:
            labels = str(tmp.iloc[0,0]).split(";")
            valores = str(tmp.iloc[1,0]).split(";")
            largo = pd.DataFrame({
                "label": [s.strip().lower() for s in labels],
                "ipc": pd.to_numeric([v.strip().replace(".","").replace(",",".") for v in valores], errors="coerce")
            })
        else:
            labels = list(tmp.iloc[0].astype(str).str.strip().str.lower().values)
            vals = tmp.iloc[1].astype(str).str.strip().str.replace(".","",regex=False).str.replace(",",".",regex=False)
            largo = pd.DataFrame({"label": labels, "ipc": pd.to_numeric(vals, errors="coerce")})
    else:
        numeric_counts = tmp.apply(lambda r: pd.to_numeric(r, errors="coerce").notna().sum(), axis=1)
        fila_idx = int(numeric_counts.reset_index(drop=True).idxmax())
        fila_vals = pd.to_numeric(
            tmp.iloc[fila_idx].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False),
            errors="coerce"
        )
        labels = list(tmp.columns.astype(str).str.strip().str.lower())
        largo = pd.DataFrame({"label": labels, "ipc": fila_vals.values})

    largo = largo.dropna(subset=["ipc"]).copy()

    def norm_label(s: str) -> str:
        s = s.replace("sept","sep")
        if mapa_mes:
            for abrev, nmes in mapa_mes.items():
                s = s.replace(abrev.lower(), str(nmes))
        return s

    largo["label_norm"] = largo["label"].apply(norm_label)

    fecha = pd.to_datetime(largo["label_norm"], format="%m-%y", errors="coerce")
    for fmt in ("%m-%Y","%Y-%m","%m/%Y","%Y/%m"):
        mask = fecha.isna()
        if mask.any():
            fecha.loc[mask] = pd.to_datetime(largo.loc[mask,"label_norm"], format=fmt, errors="coerce")
    mask = fecha.isna()
    if mask.any():
        fecha.loc[mask] = pd.to_datetime(largo.loc[mask,"label_norm"], errors="coerce", dayfirst=True)

    if fecha.isna().any():
        malos = largo.loc[fecha.isna(),"label"].tolist()
        raise ValueError(f"No pude parsear rótulos IPC: {malos}")

    largo["periodo"] = fecha.dt.to_period("M").dt.to_timestamp()
    largo = (largo.groupby("periodo", as_index=False)["ipc"].mean().sort_values("periodo"))
    return largo[["periodo","ipc"]]

def ajustar_a_base(df_servicios: pd.DataFrame, df_ipc: pd.DataFrame, base: str) -> pd.DataFrame:
    base_ts = pd.to_datetime(base).to_period("M").to_timestamp()

    ipc_base = df_ipc[df_ipc["periodo"] <= base_ts].tail(1)
    if ipc_base.empty:
        raise ValueError("No hay IPC anterior o igual a la base elegida.")
    valor_base = float(ipc_base["ipc"].iloc[0])

    merged = df_servicios.merge(df_ipc, on="periodo", how="left", validate="many_to_one")
    if merged["ipc"].isna().any():
        falt = merged[merged["ipc"].isna()]["periodo"].unique()
        raise ValueError(f"Falta IPC para periodos: {falt}")

    merged["deflactor"] = valor_base / merged["ipc"]
    merged["monto_ajustado_base"] = merged["_monto_float"] * merged["deflactor"]
    return merged


# =================================================================================================================================================

# =========================
# UI — Streamlit (bloque unificado)
# =========================

st.set_page_config(page_title="Transfer pricing tools", page_icon="📈", layout="centered")
st.title("Herramientas para analisis de precios en transferencia de bienes y servicios")

st.markdown(
    "- Ajuste por inflación de transacciones, exportar resultados, previsualizar y gráficos.\n"
    "- Herramientas para graficar rapidamente topicos del contexto macroeconómico.\n"
    "- Lectura de balances en PDFs para el armado de estados de resultados y de situación patrimonial.\n"
    "- Confección automatica de analisis de comparación de precios con comparables internos.\n"
)

# --- mapeo separadores común ---
sep_map = {";": ";", ",": ",", "\\t (tab)": "\t"}

# =================================================================================================================================================

with st.expander("Ajustar por inflación montos de transacciones", expanded=True):

    # ---------------------------
    # 1) Subir CSV(s) transacciones
    # ---------------------------
    st.subheader("1) Subir CSV de transacciones")
    files_serv = st.file_uploader("Elegí 1 o más CSV de transacciones", type=["csv"], accept_multiple_files=True, key="files_tx")
    colt1, colt2 = st.columns(2)
    with colt1:
        sep_serv = st.selectbox("Separador (transacciones)", [";", ",", "\\t (tab)"], index=0, key="sep_tx")
    with colt2:
        encoding_serv = st.selectbox("Encoding (transacciones)", ["latin1", "utf-8", "utf-8-sig", "cp1252"], index=0, key="enc_tx")

    # leer primer CSV para inferir columnas
    columnas_disponibles, fecha_col, monto_col = None, None, None
    if files_serv:
        try:
            _sample = pd.read_csv(
                io.BytesIO(files_serv[0].getvalue()),
                sep=sep_map[sep_serv], encoding=encoding_serv, engine="python", nrows=5
            )
            columnas_disponibles = list(_sample.columns)
            st.session_state.setdefault("fecha_col", columnas_disponibles[0])
            st.session_state.setdefault("monto_col", columnas_disponibles[-1])

            colc1, colc2 = st.columns(2)
            with colc1:
                fecha_col = st.selectbox(
                    "Columna de FECHA", columnas_disponibles,
                    index=columnas_disponibles.index(st.session_state["fecha_col"]) if st.session_state["fecha_col"] in columnas_disponibles else 0,
                    key="fecha_col"
                )
            with colc2:
                monto_col = st.selectbox(
                    "Columna de MONTO", columnas_disponibles,
                    index=columnas_disponibles.index(st.session_state["monto_col"]) if st.session_state["monto_col"] in columnas_disponibles else len(columnas_disponibles)-1,
                    key="monto_col"
                )
        except Exception as e:
            st.error(f"No pude leer columnas del primer CSV: {e}")

    st.divider()

    # ---------------------------
    # 2) Elegir fuente de IPC
    # ---------------------------
    st.subheader("2) Elegir fuente de IPC")
    fuente_ipc = st.radio("Fuente del IPC", ["API (INDEC)", "CSV"], index=0, horizontal=True)

    file_ipc = None  # define por adelantado
    if fuente_ipc == "API (INDEC)":
        st.markdown("Usaremos la API de **Datos Argentina / INDEC** (IPC mensual).")

        # presets + manual
        preset_opciones = list(SERIES_PRESETS.keys()) + ["(Ingresar ID manualmente)"]
        sel = st.selectbox("Serie IPC (preset)", preset_opciones, index=0)
        if sel == "(Ingresar ID manualmente)":
            serie_id = st.text_input(
                "ID de serie (pegá el ID exacto)",
                value=st.session_state.get("serie_id_manual", "101.1_I2NG_2016_M_22"),
                key="serie_id_manual",
                help="Ej.: 101.1_I2NG_2016_M_22 (Nivel general — Nacional, base dic-2016)"
            )
        else:
            serie_id = SERIES_PRESETS[sel]
            st.caption(f"ID seleccionado: `{serie_id}`")

        st.session_state["serie_id"] = serie_id  # persistir

        # rango opcional
        colr1, colr2 = st.columns(2)
        with colr1:
            start_api = st.text_input("start_date (opcional)", value=st.session_state.get("start_api", ""), key="start_api")
        with colr2:
            end_api   = st.text_input("end_date (opcional)",   value=st.session_state.get("end_api", ""),   key="end_api")

        st.info(f"Serie IPC activa: `{st.session_state['serie_id']}`")

    else:
        st.write("Subí un CSV de IPC en **formato largo** (fecha, ipc) o **ancho** (meses/valores).")
        file_ipc = st.file_uploader("CSV de IPC", type=["csv"], accept_multiple_files=False)
        colci1, colci2 = st.columns(2)
        with colci1:
            ipc_formato = st.radio("Formato del IPC", ["Largo (fecha,ipc)", "Ancho (meses en columnas)"], index=1, horizontal=True)
            sep_ipc = st.selectbox("Separador IPC", [";", ",", "\\t (tab)"], index=0)
        with colci2:
            ipc_tiene_header = st.checkbox("IPC ancho con encabezado", value=False, help="Dejalo apagado si tu CSV viene como 2 filas (meses/valores).")
            encoding_ipc = st.selectbox("Encoding IPC", ["latin1", "utf-8", "utf-8-sig", "cp1252"], index=0)

    st.divider()

    # ---------------------------
    # 3) Parámetros
    # ---------------------------
    st.subheader("3) Parámetros")
    colp1, colp2 = st.columns(2)
    with colp1:
        base_period = st.date_input("Base de referencia (llevar valores a):", value=date(2024,12,1))
        etiqueta_global = st.selectbox(
            "Etiqueta opcional para todas las transacciones",
            [None, "Servicio prestado", "Servicio recibido", "Importaciones", "Compras", "Exportaciones", "Ventas", "Intereses financieros"],
            index=0
        )
    with colp2:
        st.markdown("**Parseo de fecha (transacciones):**")
        r1, r2, r3 = st.columns([1,1,1.2])
        with r1: ui_dayfirst = st.checkbox("Día primero (dd/mm/aaaa)", value=True)
        with r2: ui_drop_bad = st.checkbox("Descartar filas con fecha inválida", value=False)
        with r3: ui_fmt = st.text_input("Formato explícito (opcional)", value="", placeholder="%d/%m/%Y o %d/%m/%Y %H:%M:%S")
    fecha_fmt = ui_fmt.strip() or None

    # Autocompletar start/end para API según transacciones
    if files_serv and fecha_col and monto_col:
        try:
            mins = []
            for f in files_serv:
                df_tmp = pd.read_csv(
                    io.BytesIO(f.getvalue()),
                    sep=sep_map[sep_serv], encoding=encoding_serv, engine="python",
                    usecols=[fecha_col]
                )
                dt = _parse_fecha_col(df_tmp[fecha_col], dayfirst=ui_dayfirst, fmt=fecha_fmt)
                per = dt.dt.to_period("M").dt.to_timestamp()
                mins.append(per.min())

            min_tx  = pd.Series(mins).min()
            base_ts = pd.to_datetime(base_period).to_period("M").to_timestamp()
            sd_auto = min_tx.strftime("%Y-%m") if pd.notna(min_tx) else ""
            ed_auto = base_ts.strftime("%Y-%m")
            st.session_state.setdefault("start_api", sd_auto)
            st.session_state.setdefault("end_api",   ed_auto)
        except Exception:
            pass

    st.divider()

    # ---------------------------
    # 4) Ajustar + Exportar
    # ---------------------------
    st.subheader("4) Ajustar y Exportar")
    btn_disabled = (
        not files_serv
        or (fuente_ipc == "CSV" and not file_ipc)
        or (fuente_ipc == "API (INDEC)" and not st.session_state.get("serie_id", "").strip())
        or not (fecha_col and monto_col)
    )

    if st.button("⚙️ Ajustar y Exportar", type="primary", disabled=btn_disabled):
        try:
            # 1) Leer transacciones
            servicios_list = []
            for f in files_serv:
                df = pd.read_csv(io.BytesIO(f.getvalue()),
                                 sep=sep_map[sep_serv], encoding=encoding_serv, engine="python")
                cfg = ServiciosConfig(nombre=f.name, fecha_col=fecha_col, monto_col=monto_col, etiqueta=etiqueta_global)
                servicios_list.append(
                    normalizar_servicios(df, cfg, fecha_fmt=fecha_fmt, dayfirst=ui_dayfirst, drop_bad_dates=ui_drop_bad)
                )
            df_serv = pd.concat(servicios_list, ignore_index=True)

            # 2) IPC
            min_tx = df_serv["periodo"].min()
            base_ts = pd.to_datetime(base_period).to_period("M").to_timestamp()
            sd_auto = min_tx.strftime("%Y-%m") if pd.notna(min_tx) else None
            ed_auto = base_ts.strftime("%Y-%m")

            if fuente_ipc == "API (INDEC)":
                sd = (st.session_state.get("start_api","").strip() or sd_auto)
                ed = (st.session_state.get("end_api","").strip()   or ed_auto)
                serie_sel = st.session_state.get("serie_id", "101.1_I2NG_2016_M_22")
                df_ipc = cargar_ipc_api(serie_id=serie_sel, start_date=sd, end_date=ed)
                st.info(f"IPC desde la API — serie: `{serie_sel}` — rango: {sd or 'inicio'} → {ed or 'último disponible'}")
            else:
                ipc_bytes = file_ipc.getvalue()
                ipc_df_raw = pd.read_csv(
                    io.BytesIO(ipc_bytes), sep=sep_map[sep_ipc], encoding=encoding_ipc, engine="python",
                    header=None if (ipc_formato=="Ancho (meses en columnas)" and not ipc_tiene_header) else "infer"
                )
                if ipc_formato == "Largo (fecha,ipc)":
                    if "fecha" in ipc_df_raw.columns and "ipc" in ipc_df_raw.columns:
                        df_ipc = cargar_ipc_largo(ipc_df_raw, "fecha", "ipc")
                    else:
                        ipc_df_raw.columns = ["fecha","ipc"] + [f"col{i}" for i in range(2, ipc_df_raw.shape[1])]
                        df_ipc = cargar_ipc_largo(ipc_df_raw[["fecha","ipc"]], "fecha", "ipc")
                else:
                    df_ipc = cargar_ipc_ancho(
                        ipc_df_raw, ancho_tiene_header=ipc_tiene_header,
                        mapa_mes={"ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
                                  "jul":"07","ago":"08","sep":"09","sept":"09","oct":"10","nov":"11","dic":"12"}
                    )

            st.caption("Preview IPC obtenido (primeras/últimas filas):")
            cp1, cp2 = st.columns(2)
            with cp1: st.dataframe(df_ipc.head(5), use_container_width=True)
            with cp2: st.dataframe(df_ipc.tail(5), use_container_width=True)

            # 3) Ajuste
            base_str = pd.to_datetime(base_period).strftime("%Y-%m-01")
            df_aj = ajustar_a_base(df_serv, df_ipc, base_str)

            # 4) Exportables
            csv_bytes = df_aj.to_csv(index=False).encode("utf-8-sig")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                cols_pref = [c for c in df_aj.columns if c not in ["_fecha_dt","_monto_float"]]
                df_aj[cols_pref].to_excel(writer, sheet_name="Ajuste", index=False)
            xlsx_bytes = output.getvalue()

            base_tag = pd.to_datetime(base_period).strftime("%Y-%m")
            st.success("¡Listo! Datos ajustados generados.")
            st.download_button("⬇️ Descargar CSV (ajustado)", data=csv_bytes,
                               file_name=f"transacciones_ajustadas_{base_tag}.csv", mime="text/csv")
            st.download_button("⬇️ Descargar Excel (ajustado)", data=xlsx_bytes,
                               file_name=f"transacciones_ajustadas_{base_tag}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.write("Previa (primeras filas):")
            st.dataframe(df_aj.head(20), use_container_width=True)

            # dejar disponible para el bloque de gráficos
            st.session_state["df_aj"] = df_aj

        except Exception as e:
            st.error(str(e))

    st.divider()

    # ---------------------------
    # 5) Gráficos (dinámica de montos)
    # ---------------------------
    st.subheader("5) Gráficos (dinámica de montos)")
    if "df_aj" not in st.session_state:
        st.info("Primero generá el ajuste para habilitar los gráficos.")
    else:
        df_prev = st.session_state["df_aj"]
        g1, g2 = st.columns(2)
        with g1:
            chart_type_h = st.selectbox("Tipo de gráfico", ["Línea", "Barras"], index=0)
            value_col_h = st.selectbox(
                "Dato a graficar",
                ["monto_ajustado_base", "_monto_float"],
                format_func=lambda v: "Monto ajustado (base)" if v=="monto_ajustado_base" else "Monto original"
            )
        with g2:
            filtro_tipo_h = None
            if "tipo" in df_prev.columns and df_prev["tipo"].notna().any():
                opts = ["(Todos)"] + sorted([str(t) for t in df_prev["tipo"].dropna().unique()])
                pick = st.selectbox("Filtrar por etiqueta", opts, index=0)
                filtro_tipo_h = None if pick == "(Todos)" else pick
            freq_lbl = st.selectbox("Agregación", ["Mensual", "Trimestral", "Anual"], index=0)
            freq_map = {"Mensual": None, "Trimestral": "Q", "Anual": "A"}

        # título dinámico por etiqueta
        default_title = "Facturación por servicios"
        if filtro_tipo_h:
            default_title = f"{default_title} — {filtro_tipo_h}"
        titulo_h = st.text_input("Título", value=default_title)

        if st.button("Generar gráfico"):
            png_bytes = make_chart_png(
                df_prev,
                chart_type="line" if chart_type_h == "Línea" else "bar",
                value_col=value_col_h,
                filtro_tipo=filtro_tipo_h,
                freq=freq_map[freq_lbl],
                titulo=titulo_h
            )
            st.image(png_bytes, caption=titulo_h, use_container_width=True)
            st.download_button("⬇️ Descargar PNG", data=png_bytes, file_name="grafico.png", mime="image/png")

# =================================================================================================================================================

# =========================================================
# 5) Graficar CSV libre (línea/barras, top-K, MEP/CCL)
# =========================================================
with st.expander("5) Graficar tópicos para el armado del contexto macro", expanded=False):
    colu1, colu2 = st.columns(2)
    with colu1:
        csv_libre = st.file_uploader("Subí un CSV para graficar", type=["csv"], key="csv_libre")
    with colu2:
        sep_libre = st.selectbox("Separador", [";", ",", "\\t (tab)"], index=0, key="sep_libre")
        enc_libre = st.selectbox("Encoding", ["utf-8", "latin1", "utf-8-sig", "cp1252"], index=1, key="enc_libre")

    if csv_libre is not None:
        # Leer a disco temporal para reusar la lógica robusta del módulo
        tmp_path = "._tmp_grafico.csv"
        with open(tmp_path, "wb") as f:
            f.write(csv_libre.getvalue())

        df_g = leer_csv_con_fallback(
            tmp_path,
            sep = {";":";", ",":",", "\\t (tab)":"\t"}[sep_libre],
            enc_pref = enc_libre
        )
        st.caption("Vista previa")
        st.dataframe(df_g.head(20), use_container_width=True)

        # --- Detección “IPC-like” y filtros rápidos ---
        col_region, col_concepto, col_fecha_auto, col_valor_auto = guess_ipc_columns(df_g)

        df_src = df_g.copy()           # <- ESTE df_src será el que usen los gráficos
        default_x = None
        default_y = []

        if col_region or col_concepto:
            st.info("Se detectaron columnas típicas de IPC; podés filtrar por **Región** y **Concepto/Clasificador**.")
            # filtros
            if col_region:
                opts_r = sorted(df_g[col_region].dropna().astype(str).unique().tolist())
                sel_r  = st.multiselect(f"Región ({col_region})", opts_r, key="ipc_reg")
                if sel_r:
                    df_src = df_src[df_src[col_region].astype(str).isin(sel_r)]
            if col_concepto:
                opts_c = sorted(df_g[col_concepto].dropna().astype(str).unique().tolist())
                sel_c  = st.multiselect(f"Concepto/Clasificador ({col_concepto})", opts_c, key="ipc_conc")
                if sel_c:
                    df_src = df_src[df_src[col_concepto].astype(str).isin(sel_c)]

            # sugerencias default para X/Y
            if col_fecha_auto in df_src.columns:
                default_x = col_fecha_auto
            num_cols = df_src.select_dtypes(include="number").columns.tolist()
            if col_valor_auto in num_cols:
                default_y = [col_valor_auto]

        tabs = st.tabs([
            "Línea / Barras (ancho o largo)",
            "Top-K (formato ancho)",
            "Tipos de cambio (MEP/CCL)",
            "BCRA API (series)"
        ])


        # -----------------------------
        # TAB 1: Línea / Barras genérico
        # -----------------------------
        with tabs[0]:
            form1 = st.form("form_line_bar")
            c1, c2, c3 = form1.columns([1,1,1])
            formato = c1.radio("Formato", ["Ancho (columnas=series)", "Largo (una col. por valor + hue opcional)"], index=0, horizontal=False)
            tipo_chart = c2.radio("Gráfico", ["Línea", "Barras"], index=0, horizontal=True)
            y_pad = c3.slider("Margen eje Y", 0.0, 0.5, 0.10, 0.01, help="Espacio arriba/abajo del máximo y mínimo")

            # >>> USAR df_src en lugar de df_g
            cols = list(df_src.columns)

            # Defaults de rango (si hay una X por defecto con pinta de fecha)
            min_txt_default = ""
            max_txt_default = ""
            if default_x and default_x in df_src.columns:
                dtc = _to_dt_like(df_src[default_x])
                if dtc.notna().any():
                    try:
                        min_txt_default = dtc.min().strftime("%Y-%m")
                        max_txt_default = dtc.max().strftime("%Y-%m")
                    except Exception:
                        pass

            # X: 1 o 2 columnas (Año + Trimestre) — con default inteligente si es IPC
            x_default = [default_x] if (default_x and default_x in cols) else [cols[0]]
            x_cols = form1.multiselect("Columna(s) X (1 o 2 para Año+Trimestre)", options=cols, default=x_default)

            # Si X es 1 columna y viene de 'periodo/fecha', sugiero parsear fecha
            parse_default = (len(x_cols) == 1 and default_x and default_x == x_cols[0])
            x_is_date = form1.checkbox("Intentar parsear X como fecha (si es 1 columna)", value=parse_default)

            # Rango de período (solo aplica cuando X es 1 columna)
            c_desde, c_hasta = form1.columns(2)
            desde_txt = c_desde.text_input("Desde (YYYY-MM o YYYY-MM-DD)", value=min_txt_default)
            hasta_txt = c_hasta.text_input("Hasta (YYYY-MM o YYYY-MM-DD)", value=max_txt_default)

            # Y: una o varias series — con default IPC si lo detectamos
            y_default = default_y if default_y else [c for c in cols[1:3] if c in cols]
            y_cols = form1.multiselect("Columnas Y (series a graficar)", options=cols, default=y_default)

            hue_col = None
            if formato.startswith("Largo"):
                hue_col = form1.selectbox("Columna de categoría (hue) [opcional]", ["(ninguna)"] + cols, index=0)
                if hue_col == "(ninguna)":
                    hue_col = None

            import re
            cols = list(df_g.columns)
            es_csv_ipc = any(
                re.search(r"(indice_?ipc|v_(m|i_a)_?ipc)", c, re.IGNORECASE) for c in cols
            )

            # Si arriba de las pestañas agregaste filtros de Región / Descripción,
            # suelen estar en estas variables (ajustá los nombres si usaste otros):
            region_txt   = ", ".join(sel_r) if "sel_r" in locals() and sel_r else ""
            concepto_txt = ", ".join(sel_c) if "sel_c" in locals() and sel_c else ""

            titulo_default = ""
            if es_csv_ipc:
                titulo_default = "Serie IPC"
                if concepto_txt:
                    titulo_default = concepto_txt
                if region_txt:
                    titulo_default = f"{titulo_default} — {region_txt}"

            # ----- título dinámico según región/descripcion -----
            import re

            cols = list(df_g.columns)
            es_csv_ipc = any(re.search(r"(indice_?ipc|v_(m|i_a)_?ipc)", c, re.IGNORECASE) for c in cols)

            region_txt   = ", ".join(sel_r) if "sel_r" in locals() and sel_r else ""
            concepto_txt = ", ".join(sel_c) if "sel_c" in locals() and sel_c else ""

            titulo_default = ""
            if es_csv_ipc:
                titulo_default = "Serie IPC"
                if concepto_txt:
                    titulo_default = concepto_txt
                if region_txt:
                    titulo_default = f"{titulo_default} — {region_txt}"

            # si cambió la "semilla" (región, concepto), refrescamos el título por defecto
            seed = (region_txt, concepto_txt)
            if st.session_state.get("_titulo_seed") != seed:
                st.session_state["_titulo_seed"] = seed
                st.session_state["titulo_csv_libre"] = titulo_default  # sobrescribe solo cuando cambia la semilla

            # widget (ahora se alimenta desde session_state)
            titulo = form1.text_input("Título",
                                    value=st.session_state.get("titulo_csv_libre", titulo_default),
                                    key="titulo_csv_libre")

            titulo = form1.text_input("Título", value=titulo_default)

            submit1 = form1.form_submit_button("Generar gráfico")

            if submit1:
                if not x_cols or not y_cols:
                    st.error("Elegí al menos una columna X y una Y.")
                else:
                    df_plot = df_src.copy()

                    # ---------- helpers (soportan YYYYMM / YYYY-MM / YYYY-MM-DD) ----------
                    def _parse_bound_txt(txt: str, end=False):
                        txt = (txt or "").strip()
                        if not txt:
                            return None
                        # YYYYMM (6 dígitos)
                        if len(txt) == 6 and txt.isdigit():
                            ts = pd.to_datetime(txt, format="%Y%m", errors="coerce")
                            if pd.isna(ts):
                                return None
                            if end:
                                ts = ts + pd.offsets.MonthEnd(0)  # fin de ese mes
                            return ts.normalize()
                        # YYYY-MM o YYYY-MM-DD
                        try:
                            if len(txt) == 7 and txt[4] == "-":  # YYYY-MM
                                ts = pd.to_datetime(txt + "-01", errors="coerce")
                                if pd.isna(ts):
                                    return None
                                if end:
                                    ts = ts + pd.offsets.MonthEnd(1)
                                return ts.normalize()
                            ts = pd.to_datetime(txt, errors="coerce")
                            return None if pd.isna(ts) else ts.normalize()
                        except Exception:
                            return None

                    def _coerce_number(s: pd.Series) -> pd.Series:
                        return pd.to_numeric(
                            s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                            errors="coerce"
                        )

                    def _make_dt_from_any(s: pd.Series) -> pd.Series:
                        """Intenta varias estrategias para construir datetime a partir de YYYYMM, YYYY-MM, etc."""
                        s_str = s.astype(str).str.strip()

                        # 1) parseo flexible directo
                        dt = pd.to_datetime(s_str, errors="coerce")

                        # 2) si falla mucho, probar YYYYMM
                        if dt.isna().mean() > 0.4:
                            num = s_str.str.replace(r"\D", "", regex=True)
                            mask6 = num.str.fullmatch(r"\d{6}")
                            dt_try = pd.to_datetime(num.where(mask6), format="%Y%m", errors="coerce")
                            dt = dt.fillna(dt_try)

                        # 3) intentar YYYY-MM truncando a 7 chars
                        if dt.isna().mean() > 0.4:
                            dt_try2 = pd.to_datetime(s_str.str.slice(0, 7), format="%Y-%m", errors="coerce")
                            dt = dt.fillna(dt_try2)

                        return dt

                    # ---------- resolver eje X y filtrar rango ----------
                    if len(x_cols) == 1:
                        x_col = x_cols[0]

                        # construir serie datetime robusta (aunque no tildes "parsear X")
                        dt_series = _make_dt_from_any(df_plot[x_col]) if x_is_date else _make_dt_from_any(df_plot[x_col])

                        # aplicar filtros
                        ds = _parse_bound_txt(desde_txt, end=False)
                        hs = _parse_bound_txt(hasta_txt, end=True)
                        if ds or hs:
                            mask = pd.Series(True, index=df_plot.index)
                            if ds is not None:
                                mask &= (dt_series >= ds)
                            if hs is not None:
                                mask &= (dt_series <= hs)
                            df_plot = df_plot[mask]
                            dt_series = dt_series[mask]

                        # usar datetime si tenemos suficientes válidos
                        if dt_series.notna().mean() >= 0.6:
                            df_plot["_X"] = dt_series
                            x_plot_label = x_col
                            x_plot_is_dt = True
                        else:
                            df_plot["_X"] = df_plot[x_col]
                            x_plot_label = x_col
                            x_plot_is_dt = False

                    elif len(x_cols) == 2:
                        df_plot, x_col = construir_x_compuesto(df_plot, x_cols)
                        df_plot["_X"] = df_plot[x_col]
                        x_plot_label = f"{x_cols[0]} - {x_cols[1]}"
                        x_plot_is_dt = False
                    else:
                        st.error("Solo se admite 1 o 2 columnas para X.")
                        st.stop()

                    # ---------- convertir Y a numérico ----------
                    for yc in y_cols:
                        if yc in df_plot.columns:
                            df_plot[yc] = _coerce_number(df_plot[yc])

                    df_plot = df_plot.dropna(subset=["_X"] + y_cols)
                    if df_plot.empty:
                        # ayuda: mostrar rango disponible
                        try:
                            _xdt_all = _make_dt_from_any(df_src[x_cols[0]])
                            if _xdt_all.notna().any():
                                rmin = _xdt_all.min().date()
                                rmax = _xdt_all.max().date()
                                st.warning(f"No quedaron filas. Rango disponible: {rmin} → {rmax}. Ajustá 'Desde/Hasta'.")
                            else:
                                st.warning("No quedaron filas válidas para graficar.")
                        except Exception:
                            st.warning("No quedaron filas válidas para graficar.")
                        st.stop()

                    # ---------- límites Y ----------
                    y_min, y_max = calcular_y_limites(df_plot, es_largo=False, y_cols=y_cols, pad=y_pad)

                    # ---------- plot ----------
                    fig, ax = plt.subplots(figsize=(12, 5))
                    if tipo_chart == "Línea":
                        for yc in y_cols:
                            ax.plot(df_plot["_X"], df_plot[yc], marker="o", linewidth=2, label=yc)
                    else:
                        x_vals = np.arange(len(df_plot["_X"]))
                        width = min(0.8 / len(y_cols), 0.25)
                        for i, yc in enumerate(y_cols):
                            ax.bar(x_vals + i * width, df_plot[yc].values, width=width, label=yc)
                        ax.set_xticks(x_vals + width * (len(y_cols) - 1) / 2)
                        ax.set_xticklabels(df_plot["_X"].astype(str).tolist(), rotation=45)

                    import matplotlib.ticker as mtick
                    # ¿Todas las series Y son variaciones del IPC? (mensual o interanual)
                    y_son_porcentaje = all(
                        re.search(r"(^|_)v_(m|i_a)_?ipc($|_)", yc, re.IGNORECASE) for yc in y_cols
                    )

                    if y_son_porcentaje:
                        # Los datos vienen como 25 = 25%  → usamos xmax=100
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
                        ax.set_ylabel("Variación (%)")
                    else:
                        ax.set_ylabel("Valor")                  
                                        
                    ax.set_ylim(y_min, y_max)
                    ax.set_title(titulo or (", ".join(y_cols) + " vs " + x_plot_label))
                    ax.set_xlabel(x_plot_label)
                    ax.set_ylabel("Valor")
                    if len(y_cols) > 1:
                        ax.legend()

                    # Formato bonito del eje X si es fecha
                    if x_plot_is_dt:
                        import matplotlib.dates as mdates
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                        fig.autofmt_xdate(rotation=45)

                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                    st.download_button("⬇️ Descargar PNG", data=buf.getvalue(),
                                    file_name="grafico_csv.png", mime="image/png")


        # --------------------------------
        # TAB 2: Top-K (formato ancho)
        # --------------------------------
        with tabs[1]:
            form2 = st.form("form_topk")
            # Sugerir excluir columnas “id” (Año/Trimestre/etc.)
            sugeridas_excluir = [c for c in ["Año", "Anio", "Year", "Trimestre", "Quarter"] if c in df_g.columns]
            id_cols = form2.multiselect("Columnas a EXCLUIR (identificadores)", options=cols, default=sugeridas_excluir)
            k = form2.slider("Top-K", 3, 25, 10, 1)
            titulo2 = form2.text_input("Título", value=f"Top {k} categorías por valor total")
            submit2 = form2.form_submit_button("Graficar Top-K")

            if submit2:
                if len([c for c in df_g.columns if c not in id_cols]) < 1:
                    st.error("No hay columnas de categorías/valores (todas están excluidas).")
                else:
                    top = agrupar_topk_ancho(df_g, id_cols=id_cols, k=k)
                    if top.empty:
                        st.warning("No se pudieron calcular totales (¿valores no numéricos?).")
                    else:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        ax2.barh(top["Categoria"], top["Valor"])
                        ax2.invert_yaxis()
                        ax2.set_title(titulo2)
                        ax2.set_xlabel("Valor")
                        for i, v in enumerate(top["Valor"].values):
                            ax2.annotate(f"{v:,.0f}", xy=(v, i), xytext=(6, 0),
                                         textcoords="offset points", va="center", ha="left")
                        fig2.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=180, bbox_inches="tight")
                        st.download_button("⬇️ Descargar PNG", data=buf2.getvalue(), file_name="topk_csv.png", mime="image/png")

        # --------------------------------
        # TAB 3: Tipos de cambio (MEP/CCL/Oficial)
        # --------------------------------
        with tabs[2]:
            # Requisitos mínimos de columnas
            req = ["Fecha", "MEP", "CCL"]
            faltan = [c for c in req if c not in df_g.columns]
            if faltan:
                st.warning(f"Este modo requiere columnas: {', '.join(req)}. Faltan: {', '.join(faltan)}")
            else:
                ctc1, ctc2, ctc3 = st.columns(3)
                desde = ctc1.text_input("Desde (YYYY-MM-DD o YYYYMM)", value="")
                hasta = ctc2.text_input("Hasta (YYYY-MM-DD o YYYYMM)", value="")
                freq = ctc3.selectbox("Frecuencia", ["M (mensual)","W (semanal)","D (diaria)"], index=0)
                inc_of = st.checkbox("Incluir TC OFICIAL (si está en el CSV)", value=True)
                titulo_tc = st.text_input("Título", value="Evolución de tipos de cambio")

                if st.button("Graficar TC"):
                    series = ["MEP", "CCL"]
                    # Si hay una columna que es el oficial con nombre largo, la función del módulo la mapea
                    if inc_of:
                        # Si ya hay "TC OFICIAL" literal o una columna “tipo de cambio de referencia…3500…mayorista”
                        if any(str(c).strip().upper() == "TC OFICIAL" for c in df_g.columns):
                            series.append("TC OFICIAL")
                    try:
                        d_tc = filtrar_tc(df_g.copy(),
                                          desde=desde or df_g["Fecha"].iloc[0],
                                          hasta=hasta or df_g["Fecha"].iloc[-1],
                                          col_fecha="Fecha",
                                          cols_tc=tuple(series),
                                          freq=freq[0])  # D/W/M
                        if d_tc.empty:
                            st.warning("No hay datos en ese rango/frecuencia.")
                        else:
                            fig3, ax3 = plt.subplots(figsize=(12,5))
                            k = max(1, len(d_tc)//30)
                            for s in series:
                                if s in d_tc.columns:
                                    ax3.plot(d_tc["_FechaDT"], d_tc[s], marker="o", linestyle="-", label=s, markevery=k)
                            ax3.set_title(titulo_tc); ax3.set_xlabel("Fecha"); ax3.set_ylabel("ARS por USD"); ax3.legend()
                            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                            fig3.autofmt_xdate(rotation=45)
                            fig3.tight_layout()
                            st.pyplot(fig3, use_container_width=True)
                            buf3 = io.BytesIO(); fig3.savefig(buf3, format="png", dpi=180, bbox_inches="tight")
                            st.download_button("⬇️ Descargar PNG", data=buf3.getvalue(), file_name="tc_csv.png", mime="image/png")
                    except Exception as e:
                        st.error(f"No se pudo graficar TC: {e}")

# =================================================================================================================================================

# ===== 6) Extraer tablas desde PDF (Estados) =====
with st.expander("6) Extraer tablas de estados contables en PDF", expanded=False):
    pdf_file = st.file_uploader("Subí el PDF del balance", type=["pdf"], key="pdf_bal")

    motor = st.selectbox(
        "Motor de extracción",
        [
            "Auto (Tabula → pdfplumber)",
            "Tabula — lattice",
            "Tabula — stream",
            "pdfplumber",
            "Asistido (split 2 columnas)"
        ],
        index=0
    )

    pages_str = st.text_input("Páginas a considerar (ej. 9,10 o 1-3,7)", placeholder="ej.: 9,10")
    diag = st.checkbox("Mostrar diagnóstico de texto en páginas", value=True)

    # Diagnóstico rápido: saber si hay texto o es escaneo
    if pdf_file and diag and pages_str.strip():
        _pags = []
        for chunk in pages_str.replace(" ", "").split(","):
            if "-" in chunk:
                a, b = chunk.split("-")
                _pags.extend(range(int(a), int(b) + 1))
            else:
                _pags.append(int(chunk))
        ok_texto = paginas_tienen_texto(pdf_file.getvalue(), _pags)
        st.info(f"Diagnóstico: {'hay texto' if ok_texto else 'NO hay texto (posible escaneo)'} en las páginas indicadas.")

    # Controles específicos del modo asistido
    split_opts = {}
    if motor == "Asistido (split 2 columnas)":
        st.markdown("**Modo asistido:** partimos la página en dos y leemos cada mitad con Tabula (stream).")
        cA, cB = st.columns(2)
        with cA:
            split_x_pct = st.slider("Corte vertical (%)", 40.0, 60.0, 50.0, 0.5,
                                    help="Dónde cortar entre izquierda (texto) y derecha (importes).")
            top_pct = st.slider("Margen superior (%)", 0.0, 20.0, 12.0, 0.5)
            bottom_pct = st.slider("Margen inferior (%)", 80.0, 100.0, 96.0, 0.5)
        with cB:
            st.caption("**Cortes de columnas dentro de la mitad derecha** (porcentaje relativo al ancho de la mitad).")
            right_cols_txt = st.text_input("Cortes derecha [% separados por coma]", value="70,88",
                                           help="Ej.: 70,88 separa *Cuenta | Col1 | Col2* dentro de la mitad derecha.")
            left_cols_txt = st.text_input("Cortes izquierda [%] (opcional)", value="")
        # parseo
        def _parse_pcts(s):
            s = s.strip()
            if not s: return []
            try:
                return [float(x) for x in s.replace(" ", "").split(",") if x.strip()!=""]
            except Exception:
                return []
        split_opts = dict(
            split_x_pct=split_x_pct,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            right_cols_pct=_parse_pcts(right_cols_txt),
            left_cols_pct=_parse_pcts(left_cols_txt),
        )

    if st.button("Extraer tablas del PDF"):
        if not pdf_file:
            st.error("Subí un PDF.")
            st.stop()

        pdf_bytes = pdf_file.getvalue()

        # Lista de páginas
        pags = []
        if pages_str.strip():
            for chunk in pages_str.replace(" ", "").split(","):
                if "-" in chunk:
                    a, b = chunk.split("-")
                    pags.extend(range(int(a), int(b) + 1))
                else:
                    pags.append(int(chunk))
        else:
            pags = [1]

        try:
            dfs_final = []

            # --- Modo asistido: split en 2 columnas y leer cada mitad
            if motor == "Asistido (split 2 columnas)":
                for pg in pags:
                    df_left, df_right = extraer_tabla_split_2columnas(
                        pdf_bytes, pg, **split_opts
                    )
                    if not df_left.empty:
                        st.caption(f"Página {pg} — Mitad IZQUIERDA")
                        st.dataframe(df_left.head(30), use_container_width=True)
                        dfs_final.append((pg, "izq", df_left))
                    if not df_right.empty:
                        st.caption(f"Página {pg} — Mitad DERECHA")
                        # ... ya obtuviste df_left, df_right según los cortes elegidos ...

                        st.caption("Mitad derecha — preprocesada")
                        st.dataframe(df_right.head(30), use_container_width=True)

                        # 👉 NUEVO: separar celdas que traen “dos importes” en una
                        df_right = split_joined_numbers_df(df_right)

                        # Limpieza numérica extra por si ya estaban separadas (suele ser la(s) últimas columna(s))
                        for c in df_right.columns[-2:]:
                            df_right[c] = normalizar_numeros_columna(df_right[c])

                        # seguir como antes: guardar, exportar, etc.
                        dfs_final.append((pg, "der", df_right))
            else:
                # --- Rutas “clásicas”
                if motor.startswith("Tabula"):
                    try:
                        import tabula  # noqa
                    except Exception:
                        st.error("Tabula no está disponible en este entorno.")
                        dfs = []
                    else:
                        flavor_lattice = ("lattice" in motor.lower())
                        flavor_stream  = ("stream"  in motor.lower())
                        dfs = []
                        for pg in pags:
                            dfs_pg = leer_tablas_tabula(
                                pdf_bytes,
                                pages_str=str(pg),
                                lattice=flavor_lattice,
                                stream=flavor_stream
                            )
                            for df in dfs_pg:
                                st.caption(f"Página {pg}")
                                st.dataframe(df.head(30), use_container_width=True)
                                dfs.append((pg, "auto", df))
                        dfs_final = dfs

                elif motor == "pdfplumber":
                    dfs = leer_tablas_pdfplumber(pdf_bytes, pags)
                    for i, df in enumerate(dfs, start=1):
                        st.caption(f"Tabla {i}")
                        st.dataframe(df.head(30), use_container_width=True)
                        dfs_final.append(("plumber", i, df))
                else:
                    # AUTO → lattice → stream → pdfplumber
                    try:
                        import tabula  # noqa
                        for pg in pags:
                            dfs = leer_tablas_tabula(pdf_bytes, str(pg), True, False)
                            if not dfs:
                                dfs = leer_tablas_tabula(pdf_bytes, str(pg), False, True)
                            if not dfs:
                                # fallback por página con pdfplumber
                                dfs = leer_tablas_pdfplumber(pdf_bytes, [pg])
                            for df in dfs:
                                st.caption(f"Página {pg}")
                                st.dataframe(df.head(30), use_container_width=True)
                                dfs_final.append((pg, "auto", df))
                    except Exception:
                        for pg in pags:
                            dfs = leer_tablas_pdfplumber(pdf_bytes, [pg])
                            for df in dfs:
                                st.caption(f"Página {pg}")
                                st.dataframe(df.head(30), use_container_width=True)
                                dfs_final.append((pg, "plumber", df))

            if not dfs_final:
                st.error("No pude extraer tablas.\n• Si el PDF es escaneado, vas a necesitar OCR.\n• Si es ‘doble columna’, probá el modo asistido.")
            else:
                # Exportar a Excel (cada tabla en una hoja)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    for j, (pg, tag, df) in enumerate(dfs_final, start=1):
                        # Encabezado: si la primera fila luce completa
                        if df.shape[0] > 1 and df.iloc[0].isna().sum() == 0:
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                        nombre_hoja = f"p{pg}_{tag}_{j}" if isinstance(pg, int) else f"{tag}_{j}"
                        # Asegurar nombres válidos para Excel
                        nombre_hoja = str(nombre_hoja)[:31]
                        df.to_excel(writer, sheet_name=nombre_hoja, index=False)
                st.download_button(
                    "⬇️ Descargar Excel (tablas)",
                    data=output.getvalue(),
                    file_name="tablas_pdf.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Fallo en extracción: {e}")

# =================================================================================================================================================

# ===== 7) Comparación interna de precios (CUPs) =====
with st.expander("7) Comparación interna de precios (CUPs)", expanded=False):
    import re
    import io
    import numpy as np
    import pandas as pd

    st.write(
        "Subí el **CSV de expos/impos** (ventas/compras) y el **Excel de países** con la "
        "etiqueta de Cooperante/No Cooperante / PJBONT. Elegí el tipo de análisis (Vinos o Commodities) "
        "y generá el análisis."
    )

    # --- Selector de vertiente ---
    tipo_analisis = st.radio(
        "Tipo de análisis", ["Vinos", "Commodities / No commodities"], horizontal=True
    )

    # --- Uploaders y opciones de lectura ---
    c1, c2 = st.columns(2)
    with c1:
        csv_expos = st.file_uploader("CSV de expos/impos (ventas / compras)", type=["csv"], key="csv_cups")
        sep_opt = st.selectbox("Separador CSV", [";", ",", "\\t (tab)"], index=0, key="csv_sep")
        enc_opt = st.selectbox("Encoding CSV", ["latin1", "utf-8", "utf-8-sig", "cp1252"], index=0, key="csv_enc")
    with c2:
        xlsx_paises = st.file_uploader("Excel de países (coop./no coop. / PJBONT)", type=["xlsx", "xls"], key="xlsx_paises_cups")
        normalizar_paises = st.checkbox("Normalizar nombres de países (diccionario simple)", value=True)
        correr = st.button("Ejecutar análisis CUP")

    # --- Helpers de mapeo y limpieza ---
    keywords = {
        "Brasil": "Brasil", "Dinamarca": "Dinamarca", "Reino Unido": "Reino Unido",
        "Estados Unidos": "Estados Unidos", "República del Paraguay": "Paraguay", "Paraguay": "Paraguay",
        "Francia": "Francia", "Alemania": "Alemania", "Canadá": "Canadá", "Perú": "Perú", "Malasia": "Malasia",
        "Nueva Zelanda": "Nueva Zelanda", "Taiwán": "Taiwán", "China": "China",
        "Uruguay": "Uruguay", "México": "México", "Colombia": "Colombia", "Puerto Rico": "Puerto Rico"
    }
    def simplify_country_name(country_name: str) -> str:
        if not isinstance(country_name, str):
            return country_name
        for k, v in keywords.items():
            if k.lower() in country_name.lower():
                return v
        return country_name

    # Parser útil para Vinos (si la descripción lo permite)
    def parse_product_description(description: str):
        """Devuelve: Marca, Varietal (entre marca y año), Cosecha (YYYY), 'Botellas por caja'."""
        try:
            if not isinstance(description, str) or not description.strip():
                return pd.Series([None, None, None, None])
            marca = description.split()[0]
            varietal_match = re.search(rf"{re.escape(marca)}\s+(.*?)\s+\b\d{{4}}\b", description)
            varietal = varietal_match.group(1) if varietal_match else None
            cosecha_match = re.search(r"\b\d{4}\b", description)
            cosecha = cosecha_match.group() if cosecha_match else None
            bot_match = re.search(r"(\d+\s*botellas.*?ml)", description, flags=re.IGNORECASE)
            botellas = bot_match.group(1) if bot_match else None
            return pd.Series([marca, varietal, cosecha, botellas])
        except Exception:
            return pd.Series([None, None, None, None])

    posibles = {
        "pais": ["Pais Factura", "País Factura", "Pais", "País", "País Facturado"],
        "producto": ["Producto", "Descripción", "Descripcion producto", "Desc Producto", "Descripcion", "Descripción Producto"],
        "precio": ["Precio", "Precio Unitario", "Precio_unit", "Unit Price", "FOB UNIT", "Precio FOB Unitario"],
        "fecha": ["Fecha Contrato", "F. Contrato", "Fecha", "Fecha Operación", "Fecha Operacion"],
        "cliente": ["Cliente Factura", "Cliente", "Razon Social"],
        "cantidad": ["Cantidad", "Qty", "Cantidad Unidades", "Cantidad Kilos"],
        "pjbont": ["PJBONT", "pjbont", "PJBONT?"],  # puede venir en CSV y/o en Excel
        "jurisd": ["Jurisdicción Cooperante / No Cooperante", "Jurisdiccion Cooperante / No Cooperante", "Jurisdiccion", "Jurisdicción"],
        # para commodities:
        "tipo": ["Tipo", "tipo", "TIPO"],
        "calibre": ["Calibre", "Cal.", "calibre"]
    }
    def pick_col(df, candidatos, obligatorio=True):
        for c in candidatos:
            if c in df.columns:
                return c
        if obligatorio:
            raise ValueError(f"No se encontró ninguna de las columnas requeridas: {candidatos}")
        return None

    if correr:
        if not csv_expos or not xlsx_paises:
            st.error("Subí **ambos** archivos (CSV de expos/impos y Excel de países).")
            st.stop()

        sep_map = {";": ";", ",": ",", "\\t (tab)": "\t"}

        # --- Lectura de archivos
        try:
            df = pd.read_csv(csv_expos, sep=sep_map[sep_opt], encoding=enc_opt, engine="python", on_bad_lines="skip")
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            st.stop()

        try:
            df_paises = pd.read_excel(xlsx_paises, engine="openpyxl")
        except Exception as e:
            st.error(f"No pude leer el Excel de países: {e}")
            st.stop()

        st.caption("Columnas detectadas en CSV: " + ", ".join(df.columns.astype(str)))
        st.caption("Columnas detectadas en Excel países: " + ", ".join(df_paises.columns.astype(str)))

        # ===============================
        # BLOQUE CLAVE: PJBONT/ JURIS
        # ===============================

        # --- Pick de columnas mínimas (CSV y Excel por separado) ---
        # CSV (operaciones)
        col_pais_csv   = pick_col(df, posibles["pais"])
        col_prod       = pick_col(df, posibles["producto"])
        col_precio     = pick_col(df, posibles["precio"])
        col_fecha      = pick_col(df, posibles["fecha"])
        col_cli        = pick_col(df, posibles["cliente"], obligatorio=False)
        col_cant       = pick_col(df, posibles["cantidad"], obligatorio=False)
        col_pjb_csv    = pick_col(df, posibles["pjbont"], obligatorio=False)  # PJBONT en CSV es opcional

        # Excel (maestro de países)
        col_pais_xlsx  = pick_col(df_paises, posibles["pais"])                 # País en el Excel
        col_jur_xlsx   = pick_col(df_paises, posibles["jurisd"], obligatorio=False)  # Jurisdicción (si existe)
        col_pjb_xlsx   = pick_col(df_paises, posibles["pjbont"], obligatorio=False)  # PJBONT en Excel (si existe)

        # --- Normalizar país y merge con Excel de países ---
        if normalizar_paises:
            df[col_pais_csv] = df[col_pais_csv].apply(simplify_country_name)
            df_paises[col_pais_xlsx] = df_paises[col_pais_xlsx].apply(simplify_country_name)

        # Renombrar a nombres estables para el join
        df = df.rename(columns={col_pais_csv: "Pais Factura"})
        ren_cols_xlsx = {col_pais_xlsx: "Pais Factura"}
        if col_jur_xlsx:
            ren_cols_xlsx[col_jur_xlsx] = "Jurisdiccion"
        if col_pjb_xlsx:
            ren_cols_xlsx[col_pjb_xlsx] = "PJBONT_flag_raw"
        df_paises_ren = df_paises.rename(columns=ren_cols_xlsx).copy()

        # Normalizar Jurisdiccion y PJBONT del Excel
        def _norm_jur(x: str) -> str:
            s = str(x).strip().lower()
            if ("no" in s) and ("cooper" in s):
                return "No Cooperante"
            if "coop" in s:
                return "Cooperante"
            return str(x).strip().title()

        def _to_bool_pjb(x) -> bool:
            if pd.isna(x):
                return False
            s = str(x).strip().upper()
            s = (s.replace("Á","A").replace("É","E").replace("Í","I")
                   .replace("Ó","O").replace("Ú","U"))
            s = " ".join(s.split())
            positivos = {
                "PJBONT","PJB","PJ BONT","SI","SÍ","SI.","TRUE","1","X",
                "VINCULADO","PARTES VINCULADAS","INTRA","INTRA-GRUPO","INTRA GRUPO"
            }
            return s in positivos

        if "Jurisdiccion" in df_paises_ren.columns:
            df_paises_ren["Jurisdiccion"] = df_paises_ren["Jurisdiccion"].apply(_norm_jur)
        df_paises_ren["PJBONT_flag_xlsx"] = df_paises_ren.get("PJBONT_flag_raw", False)
        df_paises_ren["PJBONT_flag_xlsx"] = df_paises_ren["PJBONT_flag_xlsx"].apply(_to_bool_pjb)

        # Unir columnas necesarias del Excel
        cols_join = ["Pais Factura", "PJBONT_flag_xlsx"]
        if "Jurisdiccion" in df_paises_ren.columns:
            cols_join.append("Jurisdiccion")
        df = df.merge(df_paises_ren[cols_join], on="Pais Factura", how="left")

        # --- Precio a numérico ---
        df[col_precio] = (
            df[col_precio].astype(str)
                          .str.replace(".", "", regex=False)   # miles
                          .str.replace(",", ".", regex=False)  # decimal
        )
        df[col_precio] = pd.to_numeric(df[col_precio], errors="coerce")
        df = df.dropna(subset=[col_precio])

        # --- Fecha y período ---
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
        df["Mes_Contrato"] = df[col_fecha].dt.to_period("M")

        # --- Normalizaciones varias (CSV PJBONT si existe) ---
        if col_pjb_csv and col_pjb_csv in df.columns:
            df["_pjb_from_csv"] = df[col_pjb_csv].apply(_to_bool_pjb)
        else:
            df["_pjb_from_csv"] = False

        if "Jurisdiccion" in df.columns:
            df["Jurisdiccion"] = df["Jurisdiccion"].apply(_norm_jur)
        else:
            df["Jurisdiccion"] = ""

        # --- PJBONT final (CSV OR Excel) + flags finales ---
        df["PJBONT_final"] = df["_pjb_from_csv"] | df["PJBONT_flag_xlsx"].fillna(False)
        df["Debe Analizarse"] = df["PJBONT_final"] | df["Jurisdiccion"].eq("No Cooperante")
        df["Comparable"]      = (~df["PJBONT_final"]) & df["Jurisdiccion"].eq("Cooperante")

        # ---- Verificación rápida (opcional) ----
        with st.expander("🔎 Verificación rápida PJBONT / Jurisdicción", expanded=False):
            cols_verif = ["Pais Factura", "Jurisdiccion", "PJBONT_final", "_pjb_from_csv", "PJBONT_flag_xlsx"]
            show_cols = [c for c in cols_verif if c in df.columns]
            st.dataframe(
                (df[show_cols]
                    .groupby(["Pais Factura","Jurisdiccion","PJBONT_final"], dropna=False)
                    .size()
                    .reset_index(name="#ops")
                    .sort_values("#ops", ascending=False)
                    .head(20)),
                use_container_width=True
            )

        # ===============================
        # Ramas de análisis y salida
        # ===============================

        # --- Branch VINOS: extraer atributos y agrupar
        if tipo_analisis == "Vinos":
            df[["Marca", "Varietal", "Cosecha", "Botellas por caja"]] = df[col_prod].apply(parse_product_description)
            group_keys = ["Marca", "Cosecha", "Varietal", "Botellas por caja", "Mes_Contrato"]

        # --- Branch COMMODITIES: columnas Tipo/Calibre si existen; fallback a Producto
        else:
            tipo_col = pick_col(df, posibles["tipo"], obligatorio=False)
            calibre_col = pick_col(df, posibles["calibre"], obligatorio=False)

            if tipo_col and tipo_col in df.columns:
                df[tipo_col] = df[tipo_col].astype(str).str.strip()
            if calibre_col and calibre_col in df.columns:
                df[calibre_col] = df[calibre_col].astype(str).str.strip()

            group_keys = ["Mes_Contrato"]
            if col_prod in df.columns:
                group_keys.insert(0, col_prod)
            if tipo_col and tipo_col in df.columns:
                group_keys.insert(1, tipo_col)
            if calibre_col and calibre_col in df.columns:
                group_keys.insert(2, calibre_col)

        # --- Cálculo de cuartiles por grupo (comparables) + flags en filas “a analizar”
        for keys, g in df.groupby(group_keys, dropna=False):
            comp = g.loc[g["Comparable"], col_precio].dropna()
            if len(comp) > 1:
                stats = {
                    "MIN": comp.min(),
                    "1Q":  comp.quantile(0.25),
                    "MED": comp.median(),
                    "3Q":  comp.quantile(0.75),
                    "MAX": comp.max()
                }
                mask_eval = df.index.isin(g.index) & df["Debe Analizarse"]
                for k, val in stats.items():
                    df.loc[mask_eval, k] = val

        # --- Salida y export
        out_cols = []
        for c in group_keys:
            if c in df.columns and c not in out_cols:
                out_cols.append(c)

        if tipo_analisis == "Vinos":
            for c in ["Marca", "Varietal", "Cosecha", "Botellas por caja"]:
                if c in df.columns and c not in out_cols:
                    out_cols.append(c)

        resto = [
            (col_cli if col_cli else None), "Pais Factura", col_precio,
            (col_cant if col_cant else None), col_fecha,
            "PJBONT_final", "Jurisdiccion", "Debe Analizarse", "Comparable",
            "MIN", "1Q", "MED", "3Q", "MAX"
        ]
        for c in resto:
            if c and c in df.columns and c not in out_cols:
                out_cols.append(c)

        df_export = df[out_cols].copy()

        # Renombres amigables
        ren = {
            "Pais Factura": "País",
            col_precio: "Precio Unitario",
            col_fecha: "Fecha Contrato",
            "Jurisdiccion": "Jurisdicción",
            "PJBONT_final": "PJBONT"
        }
        if col_cli and col_cli in df_export.columns:
            ren[col_cli] = "Cliente"
        if col_cant and col_cant in df_export.columns:
            ren[col_cant] = "Cantidad"
        if "Mes_Contrato" in df_export.columns:
            ren["Mes_Contrato"] = "Contrato (Mes)"

        df_export = df_export.rename(columns=ren)

        # KPIs rápidos
        grupos_total = df.groupby(group_keys).ngroups
        grupos_con_comp = sum(len(g.loc[g["Comparable"], col_precio].dropna()) > 1
                              for _, g in df.groupby(group_keys, dropna=False))
        st.success(f"Listo. Grupos totales: **{grupos_total}** | Con comparables suficientes: **{grupos_con_comp}**")

        st.dataframe(df_export.head(30), use_container_width=True)

        # Excel con formato condicional “Debe Analizarse”
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            sh = "Análisis CUP"
            df_export.to_excel(writer, index=False, sheet_name=sh)
            wb = writer.book
            ws = writer.sheets[sh]
            if "Debe Analizarse" in df_export.columns:
                col_idx = df_export.columns.get_loc("Debe Analizarse")
                fmt_green = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
                ws.conditional_format(
                    1, 0, len(df_export), len(df_export.columns)-1,
                    {"type": "formula", "criteria": f"=${chr(65+col_idx)}2=TRUE", "format": fmt_green}
                )

        st.download_button(
            "⬇️ Descargar Excel (CUP)",
            data=buf.getvalue(),
            file_name=("analisis_cup_vinos.xlsx" if tipo_analisis=="Vinos" else "analisis_cup_commodities.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

