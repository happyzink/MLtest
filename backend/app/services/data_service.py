import pandas as pd
import os
from app.core.config import UPLOAD_DIR


def save_and_parse(file_bytes: bytes, filename: str) -> dict:
    """Save uploaded Excel file and return parsed metadata."""
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    df = pd.read_excel(filepath)
    col_types = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype or "float" in dtype:
            col_types[col] = "numeric"
        elif "datetime" in dtype:
            col_types[col] = "datetime"
        else:
            col_types[col] = "categorical"

    preview = df.head(20).fillna("").to_dict(orient="records")

    return {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": col_types,
        "preview": preview,
    }


def load_dataframe(filename: str) -> pd.DataFrame:
    """Load a previously uploaded Excel file as DataFrame."""
    filepath = os.path.join(UPLOAD_DIR, filename)
    return pd.read_excel(filepath)


def get_split_ratio(n: int) -> tuple[float, str]:
    """Auto-determine train/test split ratio based on dataset size."""
    if n < 1000:
        return 0.30, "70/30"
    elif n <= 10000:
        return 0.20, "80/20"
    else:
        return 0.10, "90/10"
