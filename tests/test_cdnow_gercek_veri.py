"""Real-data check: the BG/NBD code reproduces Fader, Hardie & Lee (2005) on the CDNOW sample.

tests/acik_veri/cdnow_ozet.csv is the per-customer summary of the public CDNOW 1/10 sample (2 357 customers who first
bought in Q1 1997; source: Bruce Hardie's datasets page, copy in pymc-marketing's data/cdnow_transactions.csv).
Same-day purchases are merged into one visit. Calibration ends 1997-09-30 (39 weeks); x_bekleme counts visits in the
next 273 days (holdout). Built by notebooks/cdnow_gercek_veri.ipynb.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from analitik import bgnbd, mbgnbd
from analitik.bgnbd import BGNBDParametreleri

VERI = np.genfromtxt(Path(__file__).parent / "acik_veri" / "cdnow_ozet.csv", delimiter=",", names=True)
X, T_X, T = VERI["x"], VERI["t_x"], VERI["T"]

# Paper, Table/Section 5 (time unit: weeks): r, alpha, a, b and log-likelihood.
MAKALE = {"r": 0.243, "alfa": 4.414, "a": 0.793, "b": 2.426}
MAKALE_LL = -9582.4


def _mle_hafta(log_olabilirlik):
    """Plain maximum likelihood (no prior) in weeks, as in the paper."""
    x, t_x, t_son = X, T_X / 7, T / 7

    def amac(teta):
        deger = -log_olabilirlik(bgnbd.tetadan_parametre(teta), x, t_x, t_son)
        return deger if np.isfinite(deger) else 1e300

    # Same start as the production default (alpha = 30 days); from a far start Nelder-Mead can drift to a, b -> inf.
    baslangic = bgnbd.parametreden_teta(BGNBDParametreleri(r=1.0, alfa=30.0 / 7, a=1.0, b=1.0))
    sonuc = minimize(amac, baslangic, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 40_000})
    return bgnbd.tetadan_parametre(sonuc.x), -sonuc.fun


def test_ozet_makaledeki_veriyle_ayni():
    assert len(X) == 2357
    assert int(X.sum()) == 2457          # repeat transactions in the calibration period (paper)
    assert int(VERI["x_bekleme"].sum()) == 1882


def test_bgnbd_mle_makale_parametrelerini_bulur():
    prm, ll = _mle_hafta(bgnbd.log_olabilirlik)
    for ad, deger in MAKALE.items():
        assert round(getattr(prm, ad), 3) == deger, ad
    assert round(ll, 1) == MAKALE_LL


def test_map_onseli_buyuk_veride_mle_ye_yakin():
    """The weak prior of the production fit (days) moves no parameter by more than 5% on 2 357 customers."""
    prm = bgnbd.fit(X, T_X, T)
    hedef = {**MAKALE, "alfa": MAKALE["alfa"] * 7}
    for ad, deger in hedef.items():
        assert abs(getattr(prm, ad) / deger - 1) < 0.05, ad


def test_mbgnbd_olabilirligi_bgnbd_den_kotu_degil():
    _, ll_bg = _mle_hafta(bgnbd.log_olabilirlik)
    _, ll_mbg = _mle_hafta(mbgnbd.log_olabilirlik)
    assert ll_mbg >= ll_bg - 1e-6
