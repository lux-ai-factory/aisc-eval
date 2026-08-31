import pandas as pd


def validate_dataframe_against_datashape(df: pd.DataFrame, datashape: dict) -> dict:
    errors, warnings = [], []
    for feature in datashape.get("features", []):
        if feature.get("role") == "ignore":
            continue
        name = feature["name"]
        if name not in df.columns:
            errors.append(f"missing column: {name}")
            continue
        series = df[name]
        expected = feature.get("semantic_type")
        if expected == "numeric" and not pd.api.types.is_numeric_dtype(series):
            errors.append(f"incompatible dtype for {name}: expected {feature.get('dtype', expected)}")
            continue
        if expected == "boolean" and not pd.api.types.is_bool_dtype(series):
            errors.append(f"incompatible dtype for {name}: expected boolean")
            continue
        mapping = feature.get("category_mapping") or {}
        if mapping:
            observed = {str(value) for value in series.dropna().unique()}
            known = set(mapping)
            if observed - known:
                warnings.append(f"{name}: unseen categories {sorted(observed - known)!r}")
            if known - observed:
                warnings.append(f"{name}: expected categories never observed {sorted(known - observed)!r}")
    return {"errors": errors, "warnings": warnings}
