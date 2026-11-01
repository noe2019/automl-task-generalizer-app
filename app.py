import io

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import SAMPLES

st.set_page_config(page_title="AutoML Task Generalizer", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Task Generalizer")
st.caption("Profile a dataset → detect the task → compare models with cross-validation → download the best pipeline")

with st.sidebar:
    st.header("1 · Data")
    src = st.radio("Source", ["Sample dataset", "Upload CSV"])
    if src == "Upload CSV":
        up = st.file_uploader("CSV file (max ~20k rows recommended)", type="csv")
        df, default_target = (pd.read_csv(up), None) if up else (None, None)
    else:
        name = st.selectbox("Sample", list(SAMPLES))
        fn, default_target = SAMPLES[name]
        df = fn()
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

if df is None:
    st.info("Upload a CSV file to start.")
    st.stop()

st.subheader("2 · Data profile")
c = st.columns(4)
c[0].metric("Rows", f"{len(df):,}")
c[1].metric("Columns", df.shape[1])
c[2].metric("Missing cells", f"{df.isna().mean().mean():.1%}")
c[3].metric("Duplicate rows", int(df.duplicated().sum()))
prof = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().mean(), "unique": df.nunique()})
l, r = st.columns([1, 2])
l.dataframe(prof.style.format({"missing": "{:.1%}"}), use_container_width=True)
r.dataframe(df.head(10), use_container_width=True)

cols = list(df.columns)
target = st.selectbox("3 · Target column", cols, index=cols.index(default_target) if default_target in cols else len(cols) - 1)
data = df.dropna(subset=[target]).head(20000)
y = data[target]
X = data.drop(columns=[target])
is_clf = y.dtype == object or str(y.dtype) == "category" or y.dtype == bool or y.nunique() <= 10
task = st.radio("Detected task (override if needed)", ["Classification", "Regression"], index=0 if is_clf else 1, horizontal=True)

num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num and X[c].nunique() <= 50]
X = X[num + cat]
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])
if task == "Classification":
    models = {"Logistic regression": LogisticRegression(max_iter=2000), "Random forest": RandomForestClassifier(300, random_state=0, n_jobs=-1),
              "Gradient boosting": HistGradientBoostingClassifier(random_state=0), "k-NN": KNeighborsClassifier(15)}
    cv, metrics, main = StratifiedKFold(5, shuffle=True, random_state=0), ["accuracy", "f1_weighted"], "accuracy"
    if y.nunique() == 2:
        metrics.append("roc_auc")
else:
    models = {"Ridge": Ridge(), "Random forest": RandomForestRegressor(300, random_state=0, n_jobs=-1),
              "Gradient boosting": HistGradientBoostingRegressor(random_state=0), "k-NN": KNeighborsRegressor(15)}
    cv, metrics, main = KFold(5, shuffle=True, random_state=0), ["r2", "neg_mean_absolute_error"], "r2"

if st.button("🚀 Run AutoML", type="primary"):
    rows, prog = [], st.progress(0.0)
    for i, (n, m) in enumerate(models.items()):
        res = cross_validate(Pipeline([("pre", pre), ("model", m)]), X, y, cv=cv, scoring=metrics, n_jobs=1)
        rows.append({"model": n, **{k: abs(res[f"test_{k}"].mean()) for k in metrics}, "fit_time_s": res["fit_time"].mean()})
        prog.progress((i + 1) / len(models))
    board = pd.DataFrame(rows).rename(columns={"neg_mean_absolute_error": "MAE"}).sort_values(main, ascending=False)
    best = board.iloc[0].model
    st.session_state.update(board=board, best=best, task=task, target=target)
    st.session_state["pipe"] = Pipeline([("pre", pre), ("model", models[best])]).fit(X, y)

if "board" in st.session_state and st.session_state.get("target") == target:
    board, best = st.session_state.board, st.session_state.best
    st.subheader("4 · Leaderboard (5-fold CV)")
    st.dataframe(board.style.highlight_max(subset=[main], color="#DCE6F1").format(precision=3), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(board, x="model", y=main, title=f"{main} by model"), use_container_width=True)
    buf = io.BytesIO()
    joblib.dump(st.session_state.pipe, buf)
    st.success(f"Best model: **{best}** — retrained on all data and ready to deploy.")
    st.download_button("⬇️ Download best pipeline (.joblib)", buf.getvalue(), f"best_model_{best.replace(' ', '_')}.joblib")
    st.code("import joblib, pandas as pd\nmodel = joblib.load('best_model.joblib')\nmodel.predict(pd.read_csv('new_data.csv'))", language="python")
