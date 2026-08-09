import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import KFold

    return KFold, OneHotEncoder, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell
def _(pd):
    train = pd.read_csv("./dataset/train.csv")
    test = pd.read_csv("./dataset/test.csv")
    return test, train


@app.cell
def _(test, train):
    print("train:", train.shape)
    print("test:", test.shape)
    train.head()
    return


@app.cell
def _(train):
    train.info()
    return


@app.cell
def _(train):
    train.describe()
    return


@app.cell
def _(plt, sns, train):
    missing_pct = train.isna().mean().sort_values(ascending=False)*100
    missing_pct = missing_pct[missing_pct>0]

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#4C72B0", ax=ax)
    ax.set_xlabel("% missing")
    ax.set_title("Missing Value per Column (train.csv)")
    plt.tight_layout()
    plt.show()
    return (missing_pct,)


@app.cell
def _(missing_pct):
    print(missing_pct)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Insight dari missing values
    100% release_date_df3 dan categories gaguna karena missing semua. Maka, drop aja! Sisanya, kita coba replace pake mean or drop juga
    """)
    return


@app.cell
def _(test, train):
    # Drop kolom yang 100% kosong DI TEST (karena percuma jadi fitur kalau test-nya kosong semua)
    cols_to_drop = ["categories", "release_date_df3", "last_update", 
                     "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "other_sales",
                     "release_date"]  # release_date juga di-drop karena 95% missing di test

    train_labeled = train.drop(columns=cols_to_drop)
    test_labeled = test.drop(columns=cols_to_drop)

    # developer cuma missing dikit (4.7% train, 0.1% test) -> aman diisi, JANGAN di-drop barisnya
    train_labeled['developer'] = train_labeled['developer'].fillna('Unknown')
    test_labeled['developer'] = test_labeled['developer'].fillna('Unknown')

    # untuk kolom numerik lain yang mungkin masih ada NaN, isi pakai mean DARI TRAIN
    # (bukan mean masing-masing, biar konsisten & tidak bocor info)
    numeric_cols = train_labeled.select_dtypes(include='number').columns.drop('quality_metric', errors='ignore')
    train_means = train_labeled[numeric_cols].mean()

    train_labeled[numeric_cols] = train_labeled[numeric_cols].fillna(train_means)
    test_labeled[numeric_cols.intersection(test_labeled.columns)] = test_labeled[numeric_cols.intersection(test_labeled.columns)].fillna(train_means)

    print(f"train.csv rows: {len(train)}")
    print(f"train_labeled rows: {len(train_labeled)}")  # harusnya ~56145, tidak menyusut
    print(f"test.csv rows: {len(test)}")
    print(f"test_labeled rows: {len(test_labeled)}")     # harusnya tetap 68983, tidak menyusut
    return test_labeled, train_labeled


@app.cell
def _(train_labeled):
    train_labeled.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting trend dan distribusi
    """)
    return


@app.cell
def _(plt, sns, train_labeled):
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(train_labeled["quality_metric"], bins=15, kde=True, ax=ax1)
    ax1.axvline(train_labeled["quality_metric"].mean())
    ax1.axvline(train_labeled["quality_metric"].median(), linestyle=":")
    ax1.set_title("Distribusi Quality Metric")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(sns, train_labeled):
    sns.scatterplot(data=train_labeled, x="critic_score", y="quality_metric")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Engineering
    """)
    return


@app.cell
def _(OneHotEncoder, pd, train_labeled):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_labeled[['primary_genre']])

    # Transform, hasilnya array
    genre_ohe_array = ohe.transform(train_labeled[["primary_genre"]])

    # Ubah jadi DataFrame dengan nama kolom yang jelas
    genre_ohe_df = pd.DataFrame(
        genre_ohe_array,
        columns=ohe.get_feature_names_out(["primary_genre"]),
        index=train_labeled.index  # penting biar index-nya nyambung
    )

    # Gabungkan dengan DataFrame asli
    train_labeled_new = pd.concat([train_labeled, genre_ohe_df], axis=1)
    return ohe, train_labeled_new


@app.cell
def _(KFold, train_labeled_new):
    publisher_mean = train_labeled_new.groupby('publisher')['quality_metric'].mean()
    overall_mean = train_labeled_new['quality_metric'].mean()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_labeled_new['publisher_encoded'] = 0.0

    for train_id, val_id in kf.split(train_labeled_new):
        fold_train = train_labeled_new.iloc[train_id]
        fold_mean = fold_train.groupby('publisher')['quality_metric'].mean()
        train_labeled_new.loc[train_labeled_new.index[val_id], 'publisher_encoded'] = train_labeled_new.iloc[val_id]['publisher'].map(fold_mean)

    train_labeled_new['publisher_encoded'] = train_labeled_new['publisher_encoded'].fillna(overall_mean)
    return overall_mean, publisher_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelling
    """)
    return


@app.cell
def _(train_labeled_new):
    feature_cols = [col for col in train_labeled_new.columns 
                     if col not in ['id', 'quality_metric', 'game_name', 'primary_genre', 
                                     'publisher', 'developer', 'categories', 
                                     'release_date_df3', 'last_update', 'release_date']]

    feature_cols_v2 = [c for c in feature_cols if c not in ['critic_score', 'genre_weight']]


    X= train_labeled_new[feature_cols]
    y = train_labeled_new["quality_metric"]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state=42
    )
    return X_train, X_val, feature_cols, y_train, y_val


@app.cell
def _(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coba XGBoost
    """)
    return


@app.cell
def _(X_train, y_train):
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    return (xgb_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _(X_val, model, y_val):
    from sklearn.metrics import root_mean_squared_error

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    print(rmse)
    return (root_mean_squared_error,)


@app.cell
def _(X_val, root_mean_squared_error, xgb_model, y_val):
    y_pred_xgb = xgb_model.predict(X_val)

    rmse_xgb = root_mean_squared_error(y_val, y_pred_xgb)

    print(rmse_xgb)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Making submission
    """)
    return


@app.cell
def _(ohe, pd, test_labeled):
    genre_ohe_array_test = ohe.transform(test_labeled[["primary_genre"]])  

    genre_ohe_df_test = pd.DataFrame(
        genre_ohe_array_test,                           
        columns=ohe.get_feature_names_out(["primary_genre"]),
        index=test_labeled.index
    )

    test_labeled_new = pd.concat([test_labeled, genre_ohe_df_test], axis=1) 
    return (test_labeled_new,)


@app.cell
def _(
    feature_cols,
    overall_mean,
    pd,
    publisher_mean,
    test_labeled_new,
    xgb_model,
):
    test_labeled_new['publisher_encoded'] = test_labeled_new['publisher'].map(publisher_mean)
    test_labeled_new['publisher_encoded'] = test_labeled_new['publisher_encoded'].fillna(overall_mean)

    X_test = test_labeled_new[feature_cols]

    test_predictions = xgb_model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_labeled_new['id'],
        'quality_metric': test_predictions
    })

    submission.to_csv('submission.csv', index=False)

    print(submission.tail())
    return (X_test,)


@app.cell
def _(np, y_train, y_val):
    from sklearn.metrics import mean_squared_error

    baseline_pred = np.full_like(y_val, y_train.mean(), dtype=float)
    baseline_rmse = mean_squared_error(y_val, baseline_pred) ** 0.5
    print(f"Baseline (nebak rata-rata): {baseline_rmse:.3f}")
    print(f"RF kamu: 14.127")
    return (mean_squared_error,)


@app.cell
def _(train_labeled_new):
    # lihat apakah publisher_encoded hampir sama persis dengan quality_metric
    print(train_labeled_new[['publisher_encoded', 'quality_metric']].corr())
    return


@app.cell
def _(X_val, mean_squared_error, xgb_model, y_val):
    val_pred_xgb = xgb_model.predict(X_val)
    val_rmse_xgb = mean_squared_error(y_val, val_pred_xgb) ** 0.5
    print(f"XGB Val RMSE: {val_rmse_xgb:.3f}")
    return


@app.cell
def _(X_test, X_train):
    print(X_test.describe())
    print(X_train.describe())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
