import pandas as pd
from glob import glob

files = glob("../../data/raw/bit/*.csv")

dtype_dic_bit = {
    'total_rec': str,
    'FOLIO': str,
    'INGRESO': str,
    'UBICP': str,
    'HERR': str,
    'SIGLA': str,
    'UBICH': str,
    'VIAJE': float,
    'OS': str,
    'OT': str,
    'CC': str,
    'FECHA': str,
    'INICIO': str,
    'DESTINO': str, 
    'MEDIDOR':str,
    'HR_KM_INI': float,
    'HR_KM_FIN': float,
    'ESTADOMOTOR': str,
    'HORAINI': str,
    'HORAFIN': str,
    'RESPONSABLE': str,
    'RUT': str,
    'ESTADO': str,
    'agno': float
    }

def read_data_from_files(files):
    bit_df = pd.DataFrame()

    bit_set = 1

    for f in files:
        print(f"Leyendo archivo {bit_set}")
        region = f.split("_")[1].replace("fos", "")[:2]
        ano = f.split("_")[1].replace("fos", "")[2:6]

        df = pd.read_csv(f, sep = ";", dtype = dtype_dic_bit)

        df["region"] = region
        df["ano"] = ano

        df["set"] = bit_set
        bit_set += 1
        bit_df = pd.concat([bit_df, df])

    bit_df.index = pd.to_datetime(bit_df["FECHA"], format="%d/%m/%y")

    del bit_df["total_rec"]
    del bit_df["FOLIO"]
    del bit_df["INGRESO"]
    del bit_df["UBICP"]
    del bit_df["UBICH"]
    del bit_df["VIAJE"]
    del bit_df["OS"]
    del bit_df["OT"]
    del bit_df["CC"]
    del bit_df["INICIO"]
    del bit_df["DESTINO"]
    del bit_df["ESTADOMOTOR"]
    del bit_df["HORAINI"]
    del bit_df["HORAFIN"]
    del bit_df["RESPONSABLE"]
    del bit_df["RUT"]
    del bit_df["ESTADO"]
    del bit_df["agno"]
    del bit_df["FECHA"]

    return bit_df


bit_df = read_data_from_files(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
# Rename columns
bit_df.columns = ["activo", "sigla", "medidor", "ini", "fin", "region", "ano", "set"]

# Delta
bit_df["delta"] = bit_df["fin"] - bit_df["ini"]
bit_df = bit_df[bit_df["delta"] > 0]
# --------------------------------------------------------------
# Resample data
# --------------------------------------------------------------

sampling = {
    "sigla": "last",
    "medidor": "last",
    "ini": "first",
    "fin": "last",
    "region": "last",
    "ano": "last",
    "set": "last",
    "delta": "sum"
}

data_resampled = (
    bit_df.groupby([pd.Grouper(freq="D"), "activo"])
    .agg(sampling)
    .dropna()
    .reset_index()
)

df_aux_2 = bit_df[bit_df['sigla'] == "10U-MMVO-363"]
df_aux = data_resampled[data_resampled['sigla'] == "10U-MMVO-363"]

# --------------------------------------------------------------
# Tipologia
# --------------------------------------------------------------
dtype_dic_rnm = {
    'ASSETNUM': str,
    'DESCRIPTION': str,
    'LOCATION': str,
    'SIGLA': str,
    'STATUS': str,
    'ASSETTYPE': str,
    'PATENTE': str,
    'PURCHASEPRICE': float,
    'TIPOCOMBUST': str,
    'CAPESTANQUE': float,
    'RENDIMIENTO': float,
    'ASIGNACION': str,
    'TARIFAHORA': float,
    'VENDOR': str,
    'CLASE': str,
    'CLASSIFICATIONID': str,
    'GORE': str,
    'PARENT': str,
    'MEDIDOR': str,
    'UMEDIDA': str,
    'INSPECTOR': str,
    'AÃ‘O': str,
    'COLOR': str,
    'FECHABAJA': str,
    'FECHAREMATE': str,
    'FECHACOMP': str,
    'FECHATRASP': str,
    'LUGARREMA': str,
    'MARCA': str,
    'MODELO': str,
    'MOTOR': str,
    'SERIE': str,
    'SIGAC': str,
    'OBSERV': str,
    'PROTRASP': str,
    'POTENCIA': str,
    'RESBAJA': str,
    'RESCOMPRA': str,
    'TRACCION': str,
    'VALORREMA': str,
    'FECHA_FACTURA': str,
    'NFACTURA': str,
    'NUMTRASP': str,
    'VALORCOM': str,
    'VENGAS': str,
    'VENRT': str,
    'FECHART': str,
    'FECHAGAS': str,
    'DESTRASP': str,
    'CAPACIDAD_M3': str,
    'CAPACIDAD_PAX': str,
    'CAPACIDAD_TON': str,
    'MFACTREMATE': str,
    'IMAGEN': str,
    'UFECHA': str
    }


#asset features
rnm_df = pd.read_csv("../../data/external/asset_vehiculosv2.csv", sep=";", dtype=dtype_dic_rnm)
#add type, loc, year, ulectura
data_resampled = pd.merge(left=data_resampled, right=rnm_df[['ASSETNUM', 'CLASE', 'LOCATION', 'AÑO', 'ULECTURA']], how='left', left_on='activo', right_on='ASSETNUM')
del data_resampled["ASSETNUM"]
#rename columns
data_resampled.columns = ["FECHA","activo", "sigla", "medidor", "ini", "fin", "region", "bit-year", "set", "delta","type", "loc", "year", "ulectura"]
#dropna
data_resampled = data_resampled.dropna(subset=['type', 'ulectura'])
#drop sin medidor
data_resampled = data_resampled[data_resampled['ulectura'] != "SIN MEDIDOR"]

# Reindexar
data_resampled.index = pd.to_datetime(data_resampled["FECHA"])
del data_resampled["FECHA"]

# data types
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled["ulectura"] = data_resampled["ulectura"].apply(lambda x: float(str(x).replace(".", "").replace(",", ".")) if pd.notna(x) else None)

# put the columns in order
data_resampled = data_resampled[["activo", "sigla", "medidor", "ini", "fin", "region", "bit-year", "set","type", "loc", "year", "ulectura", "delta"]]

#review
data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
data_resampled.to_csv("../../data/interim/01_data_processed.csv")
