"""Recomputes every figure quoted in the README, independently of the notebook."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "dpe_existing_dwellings_2024-12.csv.gz"
LABELS = list("ABCDEFG")


@pytest.fixture(scope="module")
def analysis():
    raw = pd.read_csv(DATA, low_memory=False)
    numeric = raw.select_dtypes(include=np.number)
    X = numeric.loc[:, numeric.isna().mean() < 0.5]
    X = X.loc[:, X.nunique() > 1]
    X = X.fillna(X.mean())
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    return raw, X, Z, pca


def test_dataset(analysis):
    raw, X, _, _ = analysis
    assert len(raw) == 8718
    assert X.shape[1] == 95
    dates = pd.to_datetime(raw["Date_établissement_DPE"])
    assert (dates.min().date().isoformat(), dates.max().date().isoformat()) == ("2024-12-15", "2024-12-22")
    assert (raw["Type_bâtiment"] == "immeuble").mean() == pytest.approx(0.03, abs=0.005)


def test_no_identifier_or_address_in_the_data(analysis):
    raw = analysis[0]
    forbidden = ("Adresse", "Code_postal", "Coordonnée", "N°_RPLS", "Invariant", "Nom__commune", "Identifiant")
    assert not [c for c in raw.columns if c.startswith(forbidden)]


def test_variance_figures(analysis):
    _, _, _, pca = analysis
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    assert round(ratio[0] * 100, 1) == 34.5
    assert round(ratio[1] * 100, 1) == 14.7
    assert round(cumulative[1] * 100) == 49
    assert int(np.argmax(cumulative >= 0.80)) + 1 == 11


def test_manual_eigendecomposition_matches_scikit_learn(analysis):
    _, _, Z, pca = analysis
    eigenvalues = np.linalg.eigh(np.corrcoef(Z, rowvar=False))[0][::-1]
    assert np.allclose(eigenvalues / eigenvalues.sum(), pca.explained_variance_ratio_)


def test_axes_and_label(analysis):
    raw, X, Z, pca = analysis
    loadings = pd.DataFrame(pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2]), index=X.columns)
    assert loadings[0].abs().idxmax() in {"Coût_total_5_usages", "Conso_5_usages_é_finale"}
    assert "Surface_chauffée_installation_chauffage_n°1" in loadings[1].abs().nlargest(3).index
    rank = raw["Etiquette_DPE"].map({label: i for i, label in enumerate(LABELS)})
    scores = pca.transform(Z)
    assert round(pd.Series(scores[:, 0]).corr(rank, method="spearman"), 2) == 0.34
    assert round(X["Conso_5_usages_par_m²_é_primaire"].corr(rank, method="spearman"), 2) == 0.94


def test_reduced_pca(analysis):
    _, X, _, pca = analysis
    weights = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    selected = weights.index[np.hypot(weights["PC1"], weights["PC2"]) >= 0.1]
    assert len(selected) == 58
    Z_sel = StandardScaler().fit_transform(X[selected])
    pca_sel = PCA().fit(Z_sel)
    scores = pca_sel.transform(Z_sel)
    cos2 = (scores[:, :2] ** 2).sum(axis=1) / (scores**2).sum(axis=1)
    assert round(np.cumsum(pca_sel.explained_variance_ratio_)[1] * 100) == 76
    assert round((cos2 >= 0.5).mean() * 100) == 80
