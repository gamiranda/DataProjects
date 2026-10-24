import io
import zipfile
from urllib.request import urlopen

import plotnine as pn
import polars as pl


def download_sinasc_data(ano: int, salvar_parquet: str | None = None) -> pl.DataFrame | None:
    
    url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_{ano}_csv.zip"

    with urlopen(url) as resp:
        conteudo = resp.read()

    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        nome_csv = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(nome_csv) as f:
            df = pl.read_csv(
                f.read(),
                separator=";",
                encoding="utf8-lossy",
                infer_schema=False, 
            )

    if salvar_parquet:
        df.write_parquet(salvar_parquet)
        return None

    return df


columns_to_remove = [
    "contador", "ORIGEM", "NUMEROLOTE", "VERSAOSIST", "DTCADASTRO", "DTRECEBIM",
    "DTRECORIGA", "DIFDATA", "OPORT_DN", "DTDECLARAC", "TPFUNCRESP", "TPDOCRESP",
    "STDNEPIDEM", "STDNNOVA", "CODPAISRES", "NATURALMAE", "CODMUNNATU", "CODUFNATU",
    "SERIESCMAE", "DTNASCMAE", "DTULTMENST", "TPMETESTIM", "CODOCUPMAE", "CODANOMAL",
    "HORANASC","ESCMAE2010","ESCMAE","APGAR1","APGAR5","STCESPARTO"
]

def remove_unused_features(df: pl.DataFrame, colunas: list[str] = columns_to_remove) -> pl.DataFrame:
    
    return df.drop(colunas, strict=False)


def create_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:

    UF_BY_CODE = {
        12: "AC", 27: "AL", 16: "AP", 13: "AM", 29: "BA", 23: "CE", 53: "DF",
        32: "ES", 52: "GO", 21: "MA", 51: "MT", 50: "MS", 31: "MG", 15: "PA",
        25: "PB", 41: "PR", 26: "PE", 22: "PI", 24: "RN", 43: "RS", 33: "RJ",
        11: "RO", 14: "RR", 42: "SC", 35: "SP", 28: "SE", 17: "TO",
    }

    _REGIONS = {
        "North": [11, 12, 13, 14, 15, 16, 17],
        "Northeast": [21, 22, 23, 24, 25, 26, 27, 28, 29],
        "Southeast": [31, 32, 33, 35],
        "South": [41, 42, 43],
        "Midwest": [50, 51, 52, 53],
    }
    REGIONS_BY_CODE = {cod: reg for reg, cods in _REGIONS.items() for cod in cods}

    return (
        df.with_columns(
            pl.col("DTNASC").cast(pl.Utf8).str.pad_start(8, "0").str.to_date("%d%m%Y", strict=False).alias("DTNASC_padded"),
            pl.col("CODMUNRES").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int32, strict=False).alias("STATE_CODE"),
        ).with_columns(
            pl.col("STATE_CODE").replace_strict(UF_BY_CODE, default=None, return_dtype=pl.Utf8).alias("STATE"),
            pl.col("STATE_CODE").replace_strict(REGIONS_BY_CODE, default=None, return_dtype=pl.Utf8).alias("REGION"),
            #pl.col("DTNASC_padded").dt.year().alias("YEAR"),
            #pl.col("DTNASC_padded").dt.month().alias("MONTH"),
            pl.col("DTNASC_padded").dt.day().alias("DAY"),
            (pl.col("DTNASC_padded").dt.weekday() % 7 + 1).alias("DAY_WEEK"),
            pl.col("DTNASC_padded").dt.quarter().alias("QUARTER"),
            pl.col("DTNASC_padded").dt.strftime("%Y-%m").alias("YEAR_MONTH"),
            pl.when(pl.col("PESO").cast(pl.Int32, strict=False) <= 2500).then(1).otherwise(0).alias("LOWBIRTHWEIGHT"),
        ).drop("STATE_CODE", "DTNASC")
    )


def optimize_dtypes(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    
    DTYPES = {
        "IDADEMAE": pl.UInt8, "IDADEPAI": pl.UInt8,
        "QTDFILVIVO": pl.UInt8, "QTDFILMORT": pl.UInt8, "QTDGESTANT": pl.UInt8,
        "QTDPARTNOR": pl.UInt8, "QTDPARTCES": pl.UInt8,
        "SEMAGESTAC": pl.UInt8, "CONSPRENAT": pl.UInt8, "MESPRENAT": pl.UInt8,
        "PESO": pl.UInt16,

        "SEXO": pl.UInt8, "RACACOR": pl.UInt8, "RACACORMAE": pl.UInt8,
        "ESTCIVMAE": pl.UInt8, "ESCMAEAGR1": pl.UInt8,
        "GRAVIDEZ": pl.UInt8, "GESTACAO": pl.UInt8, "PARTO": pl.UInt8,
        "CONSULTAS": pl.UInt8, "LOCNASC": pl.UInt8,
        "TPAPRESENT": pl.UInt8, "STTRABPART": pl.UInt8,
        "TPNASCASSI": pl.UInt8, "TPROBSON": pl.UInt8,
        "KOTELCHUCK": pl.UInt8, "IDANOMAL": pl.UInt8, "PARIDADE": pl.UInt8,
        
        "CODMUNRES": pl.Categorical, "CODMUNNASC": pl.Categorical, "CODESTAB": pl.Categorical,
          
        "YEAR": pl.Int16, "MONTH": pl.UInt8, "DAY": pl.UInt8,
        "DAY_WEEK": pl.UInt8, "QUARTER": pl.UInt8, "LOWBIRTHWEIGHT": pl.UInt8,
        "YEAR_MONTH": pl.Categorical, "STATE": pl.Categorical, "REGION": pl.Categorical,
    }
    

    cols = df.collect_schema().names() 
    return df.with_columns(pl.col(c).cast(t) for c, t in DTYPES.items() if c in cols)
    

def plot_bars(df, variavel, mapa, titulo, xlabel, valor_nulo=9, tamanho_texto=9, rotacao=30):
    
    rotulo_ignorado = mapa.get(valor_nulo, "Ignorado")

    df_plot = df.with_columns(
        pl.col(variavel)
        .cast(pl.Int64, strict=False)
        .fill_null(valor_nulo)
        .replace_strict(mapa, default=rotulo_ignorado, return_dtype=pl.Utf8)
        .alias(variavel)
    )

    ordem = (
        df_plot.group_by(variavel)
        .len()
        .sort("len", descending=True)
        [variavel]
        .to_list()
    )

    barras = (
        pn.ggplot(df_plot, pn.aes(x=variavel))
        + pn.geom_bar(fill="#374e55")
        + pn.geom_text(
            pn.aes(label=pn.after_stat("count")),
            stat="count",
            va="bottom",
            size=tamanho_texto,
        )
        + pn.labs(title=titulo, x=xlabel, y="Frequência")
        + pn.theme_minimal()
        + pn.scale_x_discrete(limits=ordem)
        + pn.theme(
            legend_position="none",
            axis_text_x=pn.element_text(rotation=rotacao, ha="right"),
        )
    )

    return barras
