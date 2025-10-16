# plot_csv.py
# Graficador interactivo con soporte de X compuesto (Año+Trimestre) y loop para varios CSV.

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def _map_tc_oficial_col(df):
    # Si ya existe "TC OFICIAL", no tocamos nada
    if any(str(c).strip().upper() == "TC OFICIAL" for c in df.columns):
        return df
    # Buscar una columna que parezca el oficial mayorista A3500
    for c in df.columns:
        if not isinstance(c, str):
            continue
        name = c.strip()
        up = name.upper()
        if ("TIPO DE CAMBIO DE REFERENCIA" in up) and ("3500" in up) and ("MAYORISTA" in up):
            return df.rename(columns={c: "TC OFICIAL"})
    return df

def _to_num_tc(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(".", "", regex=False)   # miles 1.234 -> 1234
    s = s.str.replace(",", ".", regex=False)  # decimales 1,23 -> 1.23
    return pd.to_numeric(s, errors="coerce")

def _parse_fecha_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    # YYYYMM -> YYYY-MM-01
    mask_yyyymm = s.str.fullmatch(r"\d{6}")
    s.loc[mask_yyyymm] = s.loc[mask_yyyymm].str.slice(0,4) + "-" + s.loc[mask_yyyymm].str.slice(4,6) + "-01"
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def filtrar_tc(df, desde, hasta, col_fecha="Fecha", cols_tc=("MEP", "CCL"), freq="M"):
    """
    freq:
      - "D" diario (sin agregación)
      - "W" semanal (promedio)
      - "M" mensual (promedio)  [RECOMENDADO]
    """
    df = _map_tc_oficial_col(df).copy()

    df[col_fecha] = _parse_fecha_col(df[col_fecha])
    for c in cols_tc:
        if c in df.columns:
            df[c] = _to_num_tc(df[c])

    # recorte por rango
    d = df[(df[col_fecha] >= pd.to_datetime(str(desde), errors="coerce")) &
           (df[col_fecha] <= pd.to_datetime(str(hasta), errors="coerce"))]

    d = d.sort_values(col_fecha)

    # Resample si corresponde
    if freq and freq.upper() in ("W", "M"):
        rule = "W" if freq.upper() == "W" else "M"
        d = (d.set_index(col_fecha)[list(cols_tc)]
               .resample(rule).mean()
               .reset_index())

    # Columnas auxiliares
    d["_FechaDT"]  = d[col_fecha]
    d["_FechaStr"] = d["_FechaDT"].dt.strftime("%Y-%m-%d")
    return d

def graficar_tc_lineas(df, cols_tc=("MEP", "CCL"), titulo=None):
    # -------- Matplotlib --------
    fig, ax = plt.subplots(figsize=(12,6))
    n = len(df)
    # marcar 1 de cada k puntos para no saturar
    k = max(1, n // 30)

    for c in cols_tc:
        if c in df.columns:
            ax.plot(df["_FechaDT"], df[c], marker="o", linestyle="-", label=c, markevery=k)

    ax.set_title(titulo or "Tipos de cambio")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("ARS por USD")
    ax.legend()

    # Ticks mensuales limpios
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    plt.show()

    # -------- Seaborn --------
    tmp = df[["_FechaDT"] + [c for c in cols_tc if c in df.columns]].melt(
        id_vars="_FechaDT", var_name="Serie", value_name="Valor"
    )
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=tmp, x="_FechaDT", y="Valor", hue="Serie", marker="o", ax=ax, estimator=None)
    ax.set_title(titulo or "Tipos de cambio")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("ARS por USD")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    plt.show()

def _norm_str(s: str) -> str:
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_num(series: pd.Series) -> pd.Series:
    # limpia %, separador de miles y coma decimal
    s = series.astype(str).str.strip().str.replace("%", "", regex=False)
    s = s.str.replace(".", "", regex=False)   # miles 1.234 -> 1234
    s = s.str.replace(",", ".", regex=False)  # decimales 4,5 -> 4.5
    return pd.to_numeric(s, errors="coerce")

def filtrar_serie_ipc(
    df,
    region="Nacional",
    descripcion="NIVEL GENERAL",
    col_region="Region",
    col_desc="Descripcion",
    col_periodo="Periodo",
    col_val="v_m_IPC",
    desde=None,
    hasta=None,
):
    d = df.copy()

    # normalizar textos y periodo
    d[col_region] = d[col_region].map(_norm_str)
    d[col_desc]   = d[col_desc].map(_norm_str)
    d[col_periodo] = pd.to_numeric(d[col_periodo], errors="coerce").astype("Int64")

    # filtrar región y descripción
    region = _norm_str(region)
    d = d[d[col_region].str.upper() == region.upper()]
    descripcion = _norm_str(descripcion)
    d = d[d[col_desc].str.upper().str.contains(re.escape(descripcion.upper()))]

    # rango YYYYMM
    if desde is not None: d = d[d[col_periodo] >= int(desde)]
    if hasta is not None: d = d[d[col_periodo] <= int(hasta)]
    d = d.sort_values(col_periodo)

    # numeric robusto
    if "Indice_IPC" in d.columns: d["Indice_IPC"] = _to_num(d["Indice_IPC"])
    if "v_m_IPC"   in d.columns: d["v_m_IPC"]   = _to_num(d["v_m_IPC"])
    if "v_i_a_IPC" in d.columns: d["v_i_a_IPC"] = _to_num(d["v_i_a_IPC"])

    # reconstruir mensual si se pidió y hay índice (alineado por índice)
    if col_val == "v_m_IPC" and "Indice_IPC" in d.columns:
        vm_calc = d.groupby([col_region, col_desc])["Indice_IPC"].transform(lambda s: s.pct_change() * 100)
        d["v_m_IPC"] = d["v_m_IPC"].fillna(vm_calc)

    # etiqueta YYYY-MM y devolver
    d["_PeriodoStr"] = d[col_periodo].astype("Int64").astype(str).str.slice(0,4) + "-" + d[col_periodo].astype("Int64").astype(str).str.slice(4,6)
    return d[[col_periodo, "_PeriodoStr", col_val]].dropna()


def graficar_serie_ipc(d, y_col="v_m_IPC", titulo=None):
    is_pct = y_col in ("v_m_IPC", "v_i_a_IPC")
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(d["_PeriodoStr"], d[y_col], marker="o", linewidth=2)
    ax.fill_between(d["_PeriodoStr"], d[y_col], alpha=0.15)
    ax.set_title(titulo or y_col)
    ax.set_xlabel("Periodo (YYYY-MM)")
    ax.set_ylabel(y_col if not is_pct else f"{y_col} (%)")
    for t in ax.get_xticklabels(): t.set_rotation(45)
    # etiquetas en puntos
    for x, y in zip(d["_PeriodoStr"], d[y_col]):
        ax.annotate(f"{y:.1f}%" if is_pct else f"{y:.1f}",
                    (x, y), textcoords="offset points", xytext=(0,6), ha="center")
    fig.tight_layout()
    plt.show()

def filtrar_nivel_general(df, region="Nacional",
                          col_region="Region", col_desc="Descripcion",
                          col_periodo="Periodo", col_val="v_m_IPC",
                          desde=None, hasta=None):
    d = df.copy()
    d[col_periodo] = pd.to_numeric(d[col_periodo], errors="coerce").astype("Int64")
    d[col_val] = pd.to_numeric(d[col_val], errors="coerce")
    # Región
    d = d[d[col_region].astype(str).str.strip().str.upper() == str(region).strip().upper()]
    # Solo NIVEL GENERAL
    d = d[d[col_desc].astype(str).str.upper().str.contains("NIVEL GENERAL")]
    # Rango
    if desde is not None: d = d[d[col_periodo] >= int(desde)]
    if hasta is not None: d = d[d[col_periodo] <= int(hasta)]
    # Etiqueta YYYY-MM
    d = d.sort_values(col_periodo)
    d["_PeriodoStr"] = d[col_periodo].astype("Int64").astype(str).str.slice(0,4) + "-" + d[col_periodo].astype("Int64").astype(str).str.slice(4,6)
    return d

def graficar_ipc_nivel_general(d, col_x="_PeriodoStr", col_y="v_m_IPC", titulo=None):
    # ¿La métrica es un porcentaje?
    is_pct = col_y in ("v_m_IPC", "v_i_a_IPC")

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(d[col_x], d[col_y], marker="o", linewidth=2)
    ax.fill_between(d[col_x], d[col_y], alpha=0.15)
    ax.set_title(titulo or f"{col_y} — serie")
    ax.set_xlabel("Periodo (YYYY-MM)")
    ax.set_ylabel(col_y if not is_pct else f"{col_y} (%)")
    for t in ax.get_xticklabels():
        t.set_rotation(45)

    # etiquetas en cada punto
    for x, y in zip(d[col_x], d[col_y]):
        txt = f"{y:.1f}%" if is_pct else f"{y:.1f}"
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    fig.tight_layout()
    plt.show()

def preparar_categorias_periodo(df, periodo, region="Nacional",
                                col_region="Region", col_periodo="Periodo",
                                col_desc="Descripcion", col_val="v_i_a_IPC"):
    d = df.copy()
    d[col_periodo] = pd.to_numeric(d[col_periodo], errors="coerce")
    d = d[(d[col_region].astype(str).str.strip().str.upper()==region.upper())
          & (d[col_periodo]==int(periodo))]
    d[col_val] = pd.to_numeric(d[col_val], errors="coerce")
    d = d[[col_desc, col_val]].dropna()
    d[col_desc] = d[col_desc].astype(str).str.strip()
    return d

def barras_comparativo_categorias(d1, d2=None, p1_label="p1", p2_label=None,
                                  col_desc="Descripcion", col_val="v_i_a_IPC",
                                  titulo=None, top_k=None):
    # merge (una o dos columnas de valores)
    if d2 is not None:
        dfm = d1.merge(d2, on=col_desc, how="inner", suffixes=(f"_{p1_label}", f"_{p2_label}"))
        # elegir orden por p1
        orden = dfm.sort_values(f"{col_val}_{p1_label}", ascending=False)
        if top_k: orden = orden.head(top_k)
        cats = orden[col_desc]
        fig, ax = plt.subplots(figsize=(10,8))
        ax.barh(cats, orden[f"{col_val}_{p1_label}"], label=p1_label, alpha=0.6)
        ax.barh(cats, orden[f"{col_val}_{p2_label}"], label=p2_label, alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel(col_val)
        ax.set_ylabel("Categoría")
        ax.set_title(titulo or f"{col_val}: {p1_label} vs {p2_label}")
        ax.legend()
        # valores con offset
        for bars in ax.containers:
            for b in bars:
                w = b.get_width(); y = b.get_y()+b.get_height()/2
                ax.annotate(f"{w:.1f}%", (w, y), xytext=(6,0), textcoords="offset points",
                            va="center", ha="left")
        fig.tight_layout()
        plt.show()
    else:
        orden = d1.sort_values(col_val, ascending=False)
        if top_k: orden = orden.head(top_k)
        fig, ax = plt.subplots(figsize=(10,8))
        ax.barh(orden[col_desc], orden[col_val])
        ax.invert_yaxis()
        ax.set_xlabel(col_val); ax.set_ylabel("Categoría")
        ax.set_title(titulo or f"{col_val} por categoría — {p1_label}")
        for i, v in enumerate(orden[col_val].values):
            ax.annotate(f"{v:.1f}%", xy=(v, i), xytext=(6,0), textcoords="offset points", va="center", ha="left")
        fig.tight_layout()
        plt.show()

def normalizar_periodo(col):
    # espera YYYYMM (num o str) -> int
    return pd.to_numeric(col, errors="coerce").astype("Int64")

def filtrar_ipc(df, region="Nacional", col_region="Region",
                col_periodo="Periodo", col_cat="Descripcion",
                col_valor="v_m_IPC", desde=None, hasta=None):
    d = df.copy()
    d[col_periodo] = normalizar_periodo(d[col_periodo])
    d[col_valor]   = pd.to_numeric(d[col_valor], errors="coerce")
    if region is not None:
        d = d[d[col_region].astype(str).str.strip().str.upper() == str(region).strip().upper()]
    if desde is not None:
        d = d[d[col_periodo] >= int(desde)]
    if hasta is not None:
        d = d[d[col_periodo] <= int(hasta)]
    # ordenar por periodo
    d = d.sort_values(col_periodo)
    # etiqueta X bonita YYYY-MM
    d["_PeriodoStr"] = d[col_periodo].astype("Int64").astype(str).str.slice(0,4) + "-" + d[col_periodo].astype("Int64").astype(str).str.slice(4,6)
    # limpiar categorías
    d[col_cat] = d[col_cat].astype(str).str.strip()
    return d

def graficar_ipc(d, col_periodo_str="_PeriodoStr", col_cat="Descripcion", col_valor="v_m_IPC", titulo=None):
    # Matplotlib
    fig, ax = plt.subplots()
    for nombre, g in d.groupby(col_cat):
        ax.plot(g[col_periodo_str], g[col_valor], marker="o", linestyle="-", label=str(nombre))
    ax.set_title(titulo or "IPC (v_m_IPC) por categoría — Región seleccionada")
    ax.set_xlabel("Periodo (YYYY-MM)")
    ax.set_ylabel("v_m_IPC")
    ax.legend(loc="best", fontsize=8)
    for t in ax.get_xticklabels(): t.set_rotation(45)
    fig.tight_layout()
    plt.show()

    # Seaborn
    fig, ax = plt.subplots()
    sns.lineplot(data=d, x=col_periodo_str, y=col_valor, hue=col_cat, marker="o", ax=ax)
    ax.set_title(titulo or "IPC (v_m_IPC) por categoría — Región seleccionada")
    ax.set_xlabel("Periodo (YYYY-MM)")
    ax.set_ylabel("v_m_IPC")
    for t in ax.get_xticklabels(): t.set_rotation(45)
    fig.tight_layout()
    plt.show()


def split_codigo_nombre(cat: str):
    """Devuelve (codigo, nombre) a partir de 'A - Agricultura …'.
       Soporta -, – o —, y cae a 'cat' si no matchea.
    """
    s = str(cat).strip()
    m = re.match(r'^([A-ZÑ])\s*[-–—]\s*(.+)$', s)
    if m:
        return m.group(1), m.group(2).strip()
    # fallback: primera palabra como código
    partes = s.split(None, 1)
    if len(partes) == 2:
        return partes[0], partes[1]
    return s, s

def pedir_int(texto, default=5, minv=1, maxv=100):
    s = pedir(texto, str(default))
    try:
        val = int(float(str(s).replace(",", ".")))
    except Exception:
        val = default
    return max(minv, min(maxv, val))

def agrupar_topk_ancho(df, id_cols, k=5):
    # id_cols: columnas a EXCLUIR (p.ej. Año, Trimestre). El resto son categorías.
    value_cols = [c for c in df.columns if c not in id_cols]
    # Convertir todas a numérico (coerce NaN si hay texto)
    tmp = df[value_cols].apply(pd.to_numeric, errors="coerce")
    s = tmp.sum(axis=0, skipna=True)  # suma total por categoría
    top = (s.sort_values(ascending=False)
             .head(k)
             .reset_index())
    top.columns = ["Categoria", "Valor"]
    return top

from matplotlib.patches import Patch

def graficar_top_categorias_ancho(df, id_cols, k=5, titulo=None, anotar=True, referencias="leyenda"):
    # 1) Top-k sumando columnas (formato ancho)
    top = agrupar_topk_ancho(df, id_cols, k)
    # 2) Extraer códigos (A, B, C...) y nombres completos
    cods, noms = zip(*[split_codigo_nombre(c) for c in top["Categoria"]])
    top["Codigo"] = list(cods)
    top["Nombre"] = list(noms)

    # ---------------- MATPLOTLIB ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top["Codigo"], top["Valor"])
    ax.invert_yaxis()  # mayor arriba
    ax.set_xlabel("Valor")
    ax.set_ylabel("Código")
    ax.set_title(titulo or f"Top {k} categorías")

    if anotar:
        dx_pts = 8  # separación en puntos (probá 6–12)
        for i, v in enumerate(top["Valor"].values):
            ax.annotate(f"{v:,.0f}", xy=(v, i), xytext=(dx_pts, 0),
                        textcoords="offset points", va="center", ha="left")

    # referencias: leyenda (por defecto) o tabla
    if referencias == "leyenda":
        # dejamos espacio a la derecha para la leyenda
        fig.subplots_adjust(right=0.72)
        handles = [Patch(label=f"{row.Codigo} — {row.Nombre}") for _, row in top.iterrows()]
        fig.legend(handles=handles, title="Referencias", loc="center left",
                   bbox_to_anchor=(0.70, 0.3), frameon=True)
    elif referencias == "tabla":
        # pequeña tabla a la derecha
        fig.subplots_adjust(right=0.70)
        ref_text = "\n".join(f"{row.Codigo} — {row.Nombre}" for _, row in top.iterrows())
        ax.text(1.02, 0.5, ref_text, transform=ax.transAxes, va="center", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    fig.tight_layout()
    plt.show()

    # ---------------- SEABORN ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, y="Codigo", x="Valor", order=top["Codigo"].tolist(), ax=ax)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Código")
    ax.set_title(titulo or f"Top {k} categorías")
    if anotar:
        dx_pts = 8  # mismo offset
        for p in ax.patches:
            w = p.get_width()
            y = p.get_y() + p.get_height()/2
            ax.annotate(f"{w:,.0f}", xy=(w, y), xytext=(dx_pts, 0),
                        textcoords="offset points", va="center", ha="left")

    if referencias == "leyenda":
        fig.subplots_adjust(right=0.70)
        handles = [Patch(label=f"{row.Codigo} — {row.Nombre}") for _, row in top.iterrows()]
        fig.legend(handles=handles, title="Referencias", loc="center left",
                   bbox_to_anchor=(0.70, 0.3), frameon=True)
    elif referencias == "tabla":
        fig.subplots_adjust(right=0.70)
        ref_text = "\n".join(f"{row.Codigo} — {row.Nombre}" for _, row in top.iterrows())
        ax.text(1.02, 0.5, ref_text, transform=ax.transAxes, va="center", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    fig.tight_layout()
    plt.show()

def pedir_int(texto, default=5, minv=1, maxv=100):
    s = pedir(texto, str(default))
    try:
        val = int(float(str(s).replace(",", ".")))
    except Exception:
        val = default
    return max(minv, min(maxv, val))

def agrupar_topk(df, cat_col, val_col, k=5):
    tmp = df[[cat_col, val_col]].copy()
    tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce")
    g = (tmp.groupby(cat_col, dropna=False, as_index=False)[val_col]
             .sum()
             .sort_values(val_col, ascending=False)
             .head(k))
    # Para que el mayor quede arriba en barh
    g = g.reset_index(drop=True)
    return g

def graficar_top_categorias(df, cat_col, val_col, k=5, titulo=None, anotar=True):
    top = agrupar_topk(df, cat_col, val_col, k)

    # --- Matplotlib (horizontal) ---
    fig, ax = plt.subplots()
    ax.barh(top[cat_col], top[val_col])
    ax.invert_yaxis()  # mayor arriba
    ax.set_xlabel(val_col)
    ax.set_ylabel(cat_col)
    ax.set_title(titulo or f"Top {k} {cat_col} por {val_col}")
    if anotar:
        dx_pts = 8  # separación en puntos (probá 6–12)
        for i, v in enumerate(top["Valor"].values):
            ax.annotate(f"{v:,.0f}", xy=(v, i), xytext=(dx_pts, 0),
                        textcoords="offset points", va="center", ha="left")
    fig.tight_layout()
    plt.show()

    # --- Seaborn (horizontal) ---
    fig, ax = plt.subplots()
    sns.barplot(data=top, y=cat_col, x=val_col, order=top[cat_col].tolist(), ax=ax)
    ax.set_xlabel(val_col)
    ax.set_ylabel(cat_col)
    ax.set_title(titulo or f"Top {k} {cat_col} por {val_col}")
    if anotar:
        dx_pts = 8  # mismo offset
        for p in ax.patches:
            w = p.get_width()
            y = p.get_y() + p.get_height()/2
            ax.annotate(f"{w:,.0f}", xy=(w, y), xytext=(dx_pts, 0),
                        textcoords="offset points", va="center", ha="left")
    fig.tight_layout()
    plt.show()


def pedir_float(texto, default=0.1, minv=0.0, maxv=0.5):
    s = pedir(texto, str(default))
    try:
        val = float(str(s).replace(",", "."))
    except Exception:
        val = default
    return max(minv, min(maxv, val))


def calcular_y_limites(df, es_largo, y_cols, hue_col=None, pad=0.20):
    import numpy as np
    if es_largo:
        s = pd.to_numeric(df[y_cols[0]], errors="coerce")
    else:
        s = pd.to_numeric(df[y_cols].stack(), errors="coerce")
    y_min = float(np.nanmin(s.values))
    y_max = float(np.nanmax(s.values))
    rango = y_max - y_min
    if rango == 0:
        rango = abs(y_min) if y_min != 0 else 1.0
    return (y_min - pad * rango, y_max + pad * rango)

def pedir(texto, default=None):
    val = input(texto).strip()
    return val if val else default

def normalizar_ruta(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\&\s*', '', s)  # quita "& " de PowerShell si aparece
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip('"').strip("'")

def intentar_parsear_datetime(serie):
    if pd.api.types.is_object_dtype(serie):
        try:
            return pd.to_datetime(serie, errors="raise")
        except Exception:
            return serie
    return serie

def intentar_a_numero(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalizar_trimestre(val):
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    mapa = {"I":1,"II":2,"III":3,"IV":4,"T1":1,"T2":2,"T3":3,"T4":4,"Q1":1,"Q2":2,"Q3":3,"Q4":4}
    if s in mapa: 
        return mapa[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 4 else None
    except Exception:
        return None

def construir_x_compuesto(df, x_cols):
    """
    x_cols: lista de 2 columnas [anio_col, trim_col]
    Devuelve (df_ordenado, nombre_columna_x)
    """
    a_col, t_col = x_cols[0], x_cols[1]
    a = pd.to_numeric(df[a_col], errors="coerce")
    t = df[t_col].map(normalizar_trimestre)
    # Etiqueta YYYY-Tn
    etiqueta = a.astype('Int64').astype(str).str.replace('<NA>', '') + "-T" + pd.Series(t, index=df.index).astype('Int64').astype(str).str.replace('<NA>', '')
    df = df.copy()
    df["_X"] = etiqueta
    df["_A"] = a
    df["_T"] = t
    df = df.sort_values(["_A", "_T"], kind="mergesort")
    return df, "_X"

def leer_csv_con_fallback(ruta, sep, enc_pref="utf-8"):
    import re
    intentos = [enc_pref or "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    intentos = list(dict.fromkeys(intentos))
    ultimo_err = None

    def _limpiar_cols(df):
        df.columns = [
            re.sub(r"\s+", " ", c.replace("\xa0", " ")).strip() if isinstance(c, str) else c
        for c in df.columns
        ]
        return df

    for enc in intentos:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, encoding_errors="strict")
            print(f"Leído OK con encoding: {enc}")
            return _limpiar_cols(df)   # <-- limpiar siempre antes de devolver
        except Exception as e:
            ultimo_err = e

    print(f"Advertencia: {ultimo_err}")
    df = pd.read_csv(ruta, sep=sep, encoding="latin-1", encoding_errors="replace")
    print("Leído con latin-1 (reemplazando caracteres no decodificables).")
    return _limpiar_cols(df)

def graficar(df, es_largo, x_col, x_label, y_cols, hue_col, titulo=None, y_pad=0.10):
    ylims = calcular_y_limites(df, es_largo, y_cols, hue_col, pad=y_pad)

    # === Matplotlib ===
    fig, ax = plt.subplots()
    if es_largo and hue_col:
        y = y_cols[0]
        for nombre, g in df.groupby(hue_col):
            ax.plot(g[x_col], g[y], marker="o", linestyle="-", label=str(nombre))
        default_title = f"{y} vs {x_label} por {hue_col}"
        ax.set_ylabel(y)
        ax.legend(title=hue_col)
    else:
        for y in y_cols:
            ax.plot(df[x_col], df[y], marker="o", linestyle="-", label=y)
        default_title = f"{', '.join(y_cols)} vs {x_label}"
        ax.set_ylabel("Valor" if len(y_cols) > 1 else y_cols[0])
        if len(y_cols) > 1:
            ax.legend()

    ax.set_title(titulo or default_title)          # <-- sin “Matplotlib — …”
    ax.set_xlabel(x_label)
    ax.set_ylim(*ylims)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()
    plt.show()

    # === Seaborn ===
    sns.set_theme()
    fig, ax = plt.subplots()
    if es_largo and hue_col:
        sns.lineplot(data=df, x=x_col, y=y_cols[0], hue=hue_col, marker="o", ax=ax)
        default_title = f"{y_cols[0]} vs {x_label} por {hue_col}"
    else:
        if len(y_cols) == 1:
            sns.lineplot(data=df, x=x_col, y=y_cols[0], marker="o", ax=ax)
            default_title = f"{y_cols[0]} vs {x_label}"
        else:
            tmp = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="serie", value_name="valor")
            sns.lineplot(data=tmp, x=x_col, y="valor", hue="serie", marker="o", ax=ax)
            default_title = f"{', '.join(y_cols)} vs {x_label}"

    ax.set_title(titulo or default_title)          # <-- sin “Seaborn — …”
    ax.set_xlabel(x_label)
    ax.set_ylabel("Valor" if len(y_cols) > 1 else y_cols[0])
    ax.set_ylim(*ylims)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()
    plt.show()


def main():
    print("=== Graficador simple de CSV (Matplotlib & Seaborn) ===\n")
    while True:
        # --- selección de archivo ---
        ruta_raw = pedir("Ruta del CSV (arrastrá el archivo aquí o pegá la ruta): ")
        if not ruta_raw:
            print("No se proporcionó una ruta. Saliendo.")
            return
        ruta = normalizar_ruta(ruta_raw)
        if not Path(ruta).exists():
            print(f"No se encontró el archivo: {ruta}\n(Entrada original: {ruta_raw})")
            continuar = pedir("¿Intentar de nuevo? (s/n) [s]: ", default="s")
            if (continuar or "s").lower().startswith("s"):
                continue
            return

        sep = pedir("Separador (ENTER para coma ','): ", default=",")
        encoding = pedir("Encoding (ENTER para utf-8): ", default="utf-8")

        try:
            df = leer_csv_con_fallback(ruta, sep, encoding)
        except Exception as e:
            print(f"Error leyendo el CSV: {e}")
            continuar = pedir("¿Intentar con otro archivo? (s/n) [s]: ", default="s")
            if (continuar or "s").lower().startswith("s"):
                continue
            return

        print("\nColumnas detectadas:")
        print(", ".join(df.columns))
        print()

        # --- menú de modo ---
        modo = pedir("Modo de gráfico (linea/barras/ipc/tc) [linea]: ", default="linea").lower()

        if modo.startswith("b"):  # ===== MODO BARRAS (TOP-K) =====
            # ¿Categorías están como columnas? (formato ANCHO)
            ancho = pedir("¿Tus categorías están en columnas (formato ancho)? (s/n) [s]: ", default="s").lower().startswith("s")
            
            if ancho:
                # Proponemos excluir Año/Trimestre si existen
                default_excluir = [c for c in ["Año", "Trimestre"] if c in df.columns]
                default_txt = ", ".join(default_excluir) if default_excluir else ""
                excluir_in = pedir(f"Columnas a EXCLUIR (ej: Año, Trimestre) [{default_txt}]: ", default=default_txt)
                id_cols = [c.strip() for c in excluir_in.split(",") if c.strip()]

                topk = pedir_int("¿Cuántas categorías? [5]: ", default=5)
                titulo = pedir("Título del gráfico (ENTER para auto): ", default=None)
            
                graficar_top_categorias_ancho(df, id_cols=id_cols, k=topk, titulo=titulo, anotar=True)

            else:
                # Formato LARGO: pedimos columna de categoría y columna de valor
                cat_col = pedir("Columna de categoría (eje Y): ")
                if not cat_col or cat_col not in df.columns:
                    print("Columna de categoría inválida.")
                    continue

                val_col = pedir("Columna de valor numérico (eje X): ")
                if not val_col or val_col not in df.columns:
                    print("Columna de valor inválida.")
                    continue

                topk = pedir_int("¿Cuántas categorías? [5]: ", default=5)
                titulo = pedir("Título del gráfico (ENTER para auto): ", default=None)

                # Esta función la tenés del paso anterior (top-k en formato largo)
                graficar_top_categorias(df, cat_col, val_col, k=topk, titulo=titulo, anotar=True)

        elif modo.startswith("ipc"):  # ===== MODO IPC: Región + Descripción + rango + métrica =====
            # ayudas rápidas (si el usuario pone "?")
            reg_ask = pedir('Región (ENTER="Nacional", ? para listar): ', default="Nacional").strip()
            if reg_ask == "?":
                regs = sorted(df["Region"].dropna().astype(str).map(_norm_str).unique().tolist())
                print("Regiones disponibles:", ", ".join(regs))
                reg_ask = pedir('Región (ENTER="Nacional"): ', default="Nacional").strip()

            desc_ask = pedir('Descripción/Categoría (ej: "NIVEL GENERAL", ? para listar): ').strip()
            if desc_ask == "?":
                cats = sorted(df["Descripcion"].dropna().astype(str).map(_norm_str).unique().tolist())
                print("Algunas descripciones:", ", ".join(cats[:25]), " ...")
                desc_ask = pedir('Descripción/Categoría (ej: "NIVEL GENERAL"): ').strip()

            metrica = pedir("Métrica (v_m_IPC/v_i_a_IPC/Indice_IPC) [v_m_IPC]: ", default="v_m_IPC").strip()
            desde  = pedir("Periodo desde (YYYYMM): ").strip()
            hasta  = pedir("Periodo hasta (YYYYMM): ").strip()
            if not (desde and hasta):
                print("Necesito ambos períodos (YYYYMM).")
                continue

            d = filtrar_serie_ipc(
                df,
                region=reg_ask or "Nacional",
                descripcion=desc_ask or "NIVEL GENERAL",
                col_region="Region", col_desc="Descripcion",
                col_periodo="Periodo", col_val=metrica,
                desde=desde, hasta=hasta
            )

            if d.empty:
                print("No hay datos que coincidan con región/descripcion/métrica/rango.")
                continue

            titulo = pedir(
                "Título del gráfico (ENTER para auto): ",
                default=f"{desc_ask or 'NIVEL GENERAL'} — {reg_ask or 'Nacional'} ({desde}-{hasta}) · {metrica}"
            )
            graficar_serie_ipc(d, y_col=metrica, titulo=titulo)

        elif modo.startswith("tc"):  # ===== MODO TC (MEP/CCL/OFICIAL) =====
            # Intentamos mapear el nombre largo del oficial a "TC OFICIAL"
            df = _map_tc_oficial_col(df)

            faltan_basicas = [c for c in ["Fecha", "MEP", "CCL"] if c not in df.columns]
            if faltan_basicas:
                print("Faltan columnas en el CSV para este modo:", ", ".join(faltan_basicas))
                print("Se esperaban columnas: Fecha, MEP, CCL (y opcional una columna del oficial).")
                continue

            desde = pedir("Periodo desde (YYYY-MM-DD o YYYYMM): ").strip()
            hasta = pedir("Periodo hasta (YYYY-MM-DD o YYYYMM): ").strip()
            if not (desde and hasta):
                print("Necesito ambos límites de fecha.")
                continue

            # Resample para un eje X prolijo
            freq = pedir("Frecuencia (D=diaria, W=semanal, M=mensual) [M]: ", default="M").strip().upper()
            if freq not in ("D", "W", "M"): freq = "M"

            incluye_of = pedir("¿Incluir TC OFICIAL? (s/n) [s]: ", default="s").lower().startswith("s")
            series = ["MEP", "CCL"]
            if incluye_of and any(str(c).strip().upper() == "TC OFICIAL" for c in df.columns):
                series.append("TC OFICIAL")

            d = filtrar_tc(df, desde=desde, hasta=hasta, col_fecha="Fecha", cols_tc=tuple(series), freq=freq)
            if d.empty:
                print("No hay datos en ese rango.")
                continue

            titulo = pedir("Título del gráfico (ENTER para auto): ",
                        default="Evolución de los principales tipos de cambio")
            graficar_tc_lineas(d, cols_tc=tuple(series), titulo=titulo)
        
        else:                  # ===== MODO LÍNEA (tu flujo original) =====
            formato = pedir("Formato de los datos (ancho/largo) [ancho]: ", default="ancho").lower()
            es_largo = formato.startswith("l")

            x_in = pedir("Columna(s) para X (una o dos, separadas por coma, ej: Año, Trimestre): ")
            if not x_in:
                print("Debés indicar al menos una columna para X.")
                continue
            x_cols = [c.strip() for c in x_in.split(",") if c.strip()]
            if any(xc not in df.columns for xc in x_cols):
                faltan = [xc for xc in x_cols if xc not in df.columns]
                print(f"Columna(s) X inválida(s): {', '.join(faltan)}")
                continue

            y_input = pedir("Nombre(s) de columna(s) Y (si varias, separadas por coma): ")
            if not y_input:
                print("Debés indicar al menos una columna Y.")
                continue
            y_cols = [c.strip() for c in y_input.split(",") if c.strip()]
            if any(y not in df.columns for y in y_cols):
                faltan = [y for y in y_cols if y not in df.columns]
                print(f"Columna(s) Y inválida(s): {', '.join(faltan)}")
                continue

            hue_col = None
            if es_largo:
                hue_col = pedir("Columna de categoría/grupo (opcional en largo, ENTER si no hay): ", default=None)
                if hue_col and hue_col not in df.columns:
                    print(f"Columna de categoría inválida: {hue_col}")
                    continue

            # --- construir X ---
            if len(x_cols) == 1:
                x_col = x_cols[0]
                df[x_col] = intentar_parsear_datetime(df[x_col])
                x_label = x_col
            elif len(x_cols) == 2:
                df, x_col = construir_x_compuesto(df, x_cols)
                x_label = f"{x_cols[0]} - {x_cols[1]}"
            else:
                print("Sólo se admite 1 o 2 columnas para X.")
                continue

            # --- Y a numérico ---
            df = intentar_a_numero(df, y_cols)

            # --- título y margen Y ---
            titulo = pedir("Título del gráfico (ENTER para auto): ", default=None)
            y_pad = pedir_float("Margen Y (0.0–0.5, ej 0.1 = 10%) [0.1]: ", default=0.10)

            # --- graficar ---
            graficar(df, es_largo, x_col, x_label, y_cols, hue_col, titulo=titulo, y_pad=y_pad)

        # --- ¿otro CSV? ---
        otra = pedir("\n¿Querés procesar otro CSV? (s/n) [s]: ", default="s")
        if otra and otra.lower()[0] in ("n", "q"):
            print("Listo. ¡Gracias!")
            break

if __name__ == "__main__":
    main()

# Fin del script
