import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert currency-like strings to numeric values for model training."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if col == "Date":
            continue
        if (
            cleaned[col].dtype == "object"
            or str(cleaned[col].dtype).startswith("string")
            or str(cleaned[col].dtype) == "str"
        ):
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def load_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    stock_path = base_dir / "cleaned_stock_data.csv"
    macro_path = base_dir / "macro-stock-forecasting" / "data" / "cleaned_stock_data_macro.csv"

    stock_df = pd.read_csv(stock_path)
    stock_df = clean_numeric_columns(stock_df)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")

    if macro_path.exists():
        macro_df = pd.read_csv(macro_path)
        macro_df = clean_numeric_columns(macro_df)
        macro_df["Date"] = pd.to_datetime(macro_df["Date"], errors="coerce")
        merged_df = stock_df.merge(macro_df, on="Date", how="left", suffixes=("", "_macro"))
        merged_df = merged_df.sort_values("Date").reset_index(drop=True)
        return merged_df

    stock_df = stock_df.sort_values("Date").reset_index(drop=True)
    return stock_df


def get_stock_tickers(df: pd.DataFrame) -> list[str]:
    close_cols = [col for col in df.columns if col.endswith(" Close")]
    tickers = sorted(col[:-len(" Close")] for col in close_cols)
    return tickers


def build_regression_model(df: pd.DataFrame, target_cols: list[str] | None = None):
    df = df.copy()

    if target_cols is None:
        target_cols = get_stock_tickers(df)
    target_cols = [ticker for ticker in target_cols if f"{ticker} Close" in df.columns and f"{ticker} Volume" in df.columns]

    macro_feature_cols = [
        col
        for col in df.columns
        if col != "Date"
        and col not in [f"{ticker} Close" for ticker in target_cols]
        and col not in [f"{ticker} Volume" for ticker in target_cols]
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().any()
    ]

    print("Using stock tickers:", target_cols)
    print(f"Macro features: {len(macro_feature_cols)}")

    for ticker in target_cols:
        target_col = f"{ticker} Close"
        volume_col = f"{ticker} Volume"

        if target_col not in df.columns or volume_col not in df.columns:
            print(f"Skipping missing target or volume for: {ticker}")
            continue

        working_df = df.dropna(subset=[target_col, volume_col]).reset_index(drop=True)
        baseline_features = [target_col, volume_col]
        full_features = baseline_features + macro_feature_cols

        for feature_group, feature_names in [("baseline", baseline_features), ("full", full_features)]:
            feature_names = [col for col in feature_names if pd.api.types.is_numeric_dtype(working_df[col])]
            X = working_df[feature_names].astype(float)
            y = working_df[target_col].astype(float)

            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X,
                y,
                working_df.index.to_numpy(),
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )

            model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            print(f"\n{ticker.upper()} - {feature_group.upper()} MODEL")
            print(f"Features used: {len(feature_names)}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R^2: {r2:.4f}")

            results = pd.DataFrame({
                "row_index": idx_test,
                "Date": working_df.loc[idx_test, "Date"].values,
                "actual": y_test.values,
                "predicted": preds,
            })
            results = results.sort_values("Date")
            print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    df = load_dataset()
    print("Loaded data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    build_regression_model(df)
