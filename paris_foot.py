#!/usr/bin/env python3
"""
Paris foot : backtest + modèle de Poisson + value bets
5 championnats (Angleterre, Espagne, Italie, France, Allemagne)

Installation :  pip install pandas numpy
Usage :
    python paris_foot.py backtest     -> teste les stratégies sur 5 saisons
    python paris_foot.py predict      -> analyse les matchs à venir
    python paris_foot.py site         -> génère la page web docs/index.html (pour GitHub)
Options : --edge 0.05  (avantage minimum exigé, 0.05 = 5 %)
          --maxodds 4  (cote maximale acceptée)
"""
import math
import os
import sys
import urllib.request
from collections import defaultdict, deque

import numpy as np
import pandas as pd

LEAGUES = {"E0": "Angleterre", "SP1": "Espagne", "I1": "Italie",
           "D1": "Allemagne", "F1": "France"}
WARMUP = "2021"                                   # saison de chauffe du modèle
SEASONS = ["2122", "2223", "2324", "2425", "2526"]  # 5 saisons testées
CURRENT = "2627"                                  # saison en cours
DATA = os.environ.get("PARIS_DATA", "data")
BASE = "https://www.football-data.co.uk"
START_BANK, STAKE_PCT = 10.0, 0.05


# ----------------------------------------------------------------- données
def download(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def normalize(raw, season, league=None):
    d = pd.DataFrame(index=raw.index)
    d["league"] = league if league else raw.get("Div")
    d["season"] = season
    d["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed",
                               errors="coerce")
    d["home"], d["away"] = raw["HomeTeam"], raw["AwayTeam"]
    d["hg"], d["ag"] = pick(raw, "FTHG"), pick(raw, "FTAG")
    d["oh"] = pick(raw, "AvgH", "B365H")
    d["od"] = pick(raw, "AvgD", "B365D")
    d["oa"] = pick(raw, "AvgA", "B365A")
    d["oo"] = pick(raw, "Avg>2.5", "B365>2.5")
    d["ou"] = pick(raw, "Avg<2.5", "B365<2.5")
    return d.dropna(subset=["date", "home", "away"])


def load_history():
    frames = []
    for s in [WARMUP] + SEASONS + [CURRENT]:
        for lg in LEAGUES:
            path = f"{DATA}/{s}_{lg}.csv"
            if not os.path.exists(path) or s == CURRENT:   # saison en cours : on rafraîchit
                try:
                    download(f"{BASE}/mmz4281/{s}/{lg}.csv", path)
                except Exception as e:
                    if not os.path.exists(path):
                        print(f"  ! {s} {lg} indisponible ({e})")
                        continue
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            d = normalize(raw, s, lg).dropna(subset=["hg", "ag"])
            frames.append(d)
    if not frames:
        sys.exit("Aucune donnée téléchargée : vérifie ta connexion internet.")
    return pd.concat(frames).sort_values("date", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ modèle
def pois(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def markets(lh, la, n=10):
    ph = np.array([pois(lh, i) for i in range(n + 1)])
    pa = np.array([pois(la, i) for i in range(n + 1)])
    g = np.outer(ph, pa)
    g /= g.sum()
    i, j = np.indices(g.shape)
    t = i + j
    return {"pH": g[i > j].sum(), "pD": g[i == j].sum(), "pA": g[i < j].sum(),
            "pO15": g[t >= 2].sum(), "pO25": g[t >= 3].sum(),
            "pO35": g[t >= 4].sum(), "pBTTS": g[(i > 0) & (j > 0)].sum()}


class Model:
    """Force d'attaque/défense sur les 12 derniers matchs, réduite vers la moyenne."""

    def __init__(self, k=12, shrink=4, min_games=5):
        self.hist = defaultdict(lambda: deque(maxlen=k))
        self.streak = defaultdict(int)       # matchs d'affilée sans marquer
        self.lg = defaultdict(lambda: [0.0, 0.0, 0])
        self.shrink, self.min_games = shrink, min_games

    def _strength(self, lg, team, avg):
        h = self.hist[(lg, team)]
        sc = (sum(x[0] for x in h) + self.shrink * avg) / (len(h) + self.shrink) / avg
        co = (sum(x[1] for x in h) + self.shrink * avg) / (len(h) + self.shrink) / avg
        return sc, co

    def predict(self, lg, h, a):
        sh, sa, n = self.lg[lg]
        if n < 50 or len(self.hist[(lg, h)]) < self.min_games \
                or len(self.hist[(lg, a)]) < self.min_games:
            return None
        avgh, avga = sh / n, sa / n
        avg = (avgh + avga) / 2
        att_h, def_h = self._strength(lg, h, avg)
        att_a, def_a = self._strength(lg, a, avg)
        lh, la = avgh * att_h * def_a, avga * att_a * def_h
        out = markets(max(lh, 0.05), max(la, 0.05))
        out["lh"], out["la"] = lh, la
        return out

    def update(self, lg, h, a, hg, ag):
        self.hist[(lg, h)].append((hg, ag))
        self.hist[(lg, a)].append((ag, hg))
        self.streak[(lg, h)] = 0 if hg > 0 else self.streak[(lg, h)] + 1
        self.streak[(lg, a)] = 0 if ag > 0 else self.streak[(lg, a)] + 1
        s = self.lg[lg]
        s[0] += hg
        s[1] += ag
        s[2] += 1


def run_model(df):
    """Passe chronologique : chaque prédiction n'utilise que le passé (pas de triche)."""
    m = Model()
    keys = ["pH", "pD", "pA", "pO15", "pO25", "pO35", "pBTTS", "lh", "la"]
    cols = {k: [] for k in keys + ["sh", "sa"]}
    for r in df.itertuples():
        p = m.predict(r.league, r.home, r.away)
        for k in keys:
            cols[k].append(p[k] if p else np.nan)
        cols["sh"].append(m.streak[(r.league, r.home)])
        cols["sa"].append(m.streak[(r.league, r.away)])
        m.update(r.league, r.home, r.away, r.hg, r.ag)
    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    out["pU25"] = 1 - out["pO25"]
    return out, m


# ---------------------------------------------------------------- backtest
def pct(x):
    return f"{x:+.1f}%" if pd.notna(x) else "  n/a"


def backtest(edge, maxodds):
    print("Chargement des données…")
    df = load_history()
    res, _ = run_model(df)
    d = res[res.season != WARMUP].copy()
    print(f"{len(d)} matchs analysés ({d.date.min():%d/%m/%Y} → {d.date.max():%d/%m/%Y})\n")

    # ---- 1) Stratégie « séries sans marquer »
    print("=" * 60)
    print("1) STRATÉGIE SÉRIES : une équipe n'a pas marqué depuis N matchs")
    print("=" * 60)
    team_rows = pd.concat([
        pd.DataFrame({"s": d.sh, "scored": d.hg > 0}),
        pd.DataFrame({"s": d.sa, "scored": d.ag > 0})])
    print(f"Base : une équipe marque dans {team_rows.scored.mean()*100:.1f}% des matchs\n")
    d["tot"] = d.hg + d.ag
    d["ms"] = d[["sh", "sa"]].max(axis=1)
    d["o25"] = d.tot >= 3
    d["btts"] = (d.hg > 0) & (d.ag > 0)
    print(" N | éq. concernées | marque ensuite | matchs | BTTS | Over2.5 | ROI Over2.5 | ROI Under2.5")
    for n in range(2, 7):
        t = team_rows[team_rows.s >= n]
        m = d[d.ms >= n]
        ov = np.where(m.o25, m.oo - 1, -1)[m.oo.notna()]
        un = np.where(~m.o25, m.ou - 1, -1)[m.ou.notna()]
        print(f" {n} | {len(t):14d} | {t.scored.mean()*100:13.1f}% | {len(m):6d} |"
              f" {m.btts.mean()*100:3.0f}% | {m.o25.mean()*100:5.0f}%  |"
              f" {pct(ov.mean()*100 if len(ov) else np.nan):>11} | {pct(un.mean()*100 if len(un) else np.nan):>11}")
    print("\n→ Si « marque ensuite » reste proche de la base et que les ROI sont négatifs,"
          "\n  la série ne donne aucun avantage : les cotes la contiennent déjà.\n")

    # ---- 2) Value bets Poisson (cotes réelles)
    print("=" * 60)
    print("2) MODÈLE DE POISSON : paris à valeur (cotes moyennes réelles)")
    print("=" * 60)
    sels = [("1", "pH", "oh", d.hg > d.ag), ("N", "pD", "od", d.hg == d.ag),
            ("2", "pA", "oa", d.hg < d.ag), ("Over2.5", "pO25", "oo", d.tot >= 3),
            ("Under2.5", "pU25", "ou", d.tot <= 2)]
    parts = []
    for name, pc, oc, win in sels:
        parts.append(pd.DataFrame({"date": d.date, "sel": name, "p": d[pc],
                                   "odds": d[oc], "win": win}))
    bets = pd.concat(parts).dropna(subset=["p", "odds"])
    bets = bets[bets.odds <= maxodds]
    bets["ev"] = bets.p * bets.odds - 1
    bets["profit"] = np.where(bets.win, bets.odds - 1, -1.0)

    print(f"Référence (parier sur tout, cote ≤ {maxodds}) : ROI {pct(bets.profit.mean()*100)}"
          f" sur {len(bets)} paris  ← c'est la marge du bookmaker\n")
    print(" Avantage min | paris | % gagnés | ROI mise fixe")
    for th in [0.0, 0.03, 0.05, 0.10, 0.15]:
        b = bets[bets.ev >= th]
        if len(b):
            print(f"   {th*100:4.0f}%      | {len(b):5d} | {b.win.mean()*100:6.1f}%  | {pct(b.profit.mean()*100)}")
    b = bets[bets.ev >= edge]
    print(f"\nDétail par pari à avantage ≥ {edge*100:.0f}% :")
    for name, g in b.groupby("sel"):
        print(f"   {name:9s} {len(g):5d} paris   ROI {pct(g.profit.mean()*100)}")

    # ---- 3) Simulation de bankroll
    print("\n" + "=" * 60)
    print(f"3) SIMULATION : {START_BANK:.0f} € de départ, mise {STAKE_PCT*100:.0f}% du capital")
    print("=" * 60)
    bk, peak, dd = START_BANK, START_BANK, 0.0
    for r in b.sort_values("date").itertuples():
        stake = bk * STAKE_PCT
        bk += stake * (r.odds - 1) if r.win else -stake
        peak = max(peak, bk)
        dd = max(dd, 1 - bk / peak)
    print(f"{len(b)} paris → capital final {bk:.2f} €   (pire baisse depuis un sommet : {dd*100:.0f}%)")
    print("\nATTENTION : même un ROI positif ici n'est pas une garantie. Les bookmakers"
          "\nlimitent les comptes gagnants et les cotes réelles bougent. Valide d'abord"
          "\nen « paper trading » avec la commande predict.")


# -------------------------------------------------------------- prédictions
def predict(edge, maxodds):
    print("Chargement des données…")
    df = load_history()
    _, m = run_model(df)
    path = f"{DATA}/fixtures.csv"
    try:
        download(f"{BASE}/fixtures.csv", path)
    except Exception as e:
        if not os.path.exists(path):
            sys.exit(f"Impossible de récupérer les matchs à venir : {e}")
    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    fx = normalize(raw, CURRENT)
    fx = fx[fx.league.isin(LEAGUES)].sort_values("date")
    if fx.empty:
        sys.exit("Aucun match à venir trouvé dans fixtures.csv.")

    picks = []
    print(f"\n{len(fx)} matchs à venir.\n")
    for r in fx.itertuples():
        p = m.predict(r.league, r.home, r.away)
        sh, sa = m.streak[(r.league, r.home)], m.streak[(r.league, r.away)]
        if not p:
            continue
        flag = "  ⚠ SÉRIE" if max(sh, sa) >= 3 else ""
        print(f"{r.date:%d/%m} [{LEAGUES[r.league]}] {r.home} - {r.away}{flag}")
        print(f"   buts attendus {p['lh']:.2f}-{p['la']:.2f} | sans marquer : {sh}/{sa}")
        print(f"   1 {p['pH']*100:.0f}% N {p['pD']*100:.0f}% 2 {p['pA']*100:.0f}% | "
              f"+1.5 {p['pO15']*100:.0f}% +2.5 {p['pO25']*100:.0f}% +3.5 {p['pO35']*100:.0f}% | "
              f"BTTS {p['pBTTS']*100:.0f}%")
        print(f"   cotes justes : +1.5 {1/p['pO15']:.2f} | +2.5 {1/p['pO25']:.2f} | "
              f"-2.5 {1/(1-p['pO25']):.2f} | BTTS {1/p['pBTTS']:.2f} | "
              f"DNB dom. {(p['pH']+p['pA'])/p['pH']:.2f} | DNB ext. {(p['pH']+p['pA'])/p['pA']:.2f}")
        for name, prob, odds in [("1", p["pH"], r.oh), ("N", p["pD"], r.od), ("2", p["pA"], r.oa),
                                 ("Over2.5", p["pO25"], r.oo), ("Under2.5", 1 - p["pO25"], r.ou)]:
            if pd.notna(odds) and odds <= maxodds and prob * odds - 1 >= edge:
                picks.append((prob * odds - 1, f"{r.date:%d/%m} {r.home}-{r.away}", name, odds, prob))

    print("\n" + "=" * 60)
    print(f"VALUE BETS (avantage ≥ {edge*100:.0f}%, cote ≤ {maxodds})")
    print("=" * 60)
    if not picks:
        print("Aucun pari à valeur aujourd'hui : ne pas parier est une décision valable.")
    for ev, match, name, odds, prob in sorted(picks, reverse=True):
        print(f"{match:35s} {name:9s} cote {odds:.2f}  proba modèle {prob*100:.0f}%  avantage {ev*100:+.0f}%")
    print("\nPour +1.5, BTTS, DNB : compare la cote de ton bookmaker à la « cote juste »"
          "\nci-dessus. Intéressant seulement si sa cote est nettement supérieure (≥ +5%)."
          "\nNote chaque pari fictif dans un tableur pendant quelques semaines avant de miser.")


# --------------------------------------------------------------- page web
def fixtures_rows(m, edge, maxodds):
    path = f"{DATA}/fixtures.csv"
    try:
        download(f"{BASE}/fixtures.csv", path)
    except Exception:
        if not os.path.exists(path):
            return []
    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    fx = normalize(raw, CURRENT)
    fx = fx[fx.league.isin(LEAGUES)].sort_values("date")
    rows = []
    for r in fx.itertuples():
        p = m.predict(r.league, r.home, r.away)
        if not p:
            continue
        picks = []
        for name, prob, odds in [("1", p["pH"], r.oh), ("N", p["pD"], r.od), ("2", p["pA"], r.oa),
                                 ("+2.5", p["pO25"], r.oo), ("-2.5", 1 - p["pO25"], r.ou)]:
            if pd.notna(odds) and odds <= maxodds and prob * odds - 1 >= edge:
                picks.append((prob * odds - 1, name, float(odds), prob))
        rows.append({"date": r.date, "lg": LEAGUES[r.league], "home": r.home, "away": r.away,
                     "sh": m.streak[(r.league, r.home)], "sa": m.streak[(r.league, r.away)],
                     "p": p, "picks": sorted(picks, reverse=True)})
    return rows


CSS = """
:root{--bg:#f4f5f7;--card:#fff;--tx:#1a1d21;--mut:#6b7280;--ac:#0f766e;--bad:#b91c1c;--bd:#e2e4e8;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root{--bg:#101214;--card:#1b1e22;--tx:#eceef1;--mut:#9aa1ab;--ac:#2dd4bf;--bad:#f87171;--bd:#2c3036}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif}
.w{max-width:560px;margin:0 auto;padding:12px}h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:16px 0 6px}
.c{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:10px 12px;margin-bottom:8px}
.m{color:var(--mut);font-size:13px}.v{border-left:4px solid var(--ac)}.w2{border-left:4px solid var(--bad)}
.t{font-size:13px;margin-top:4px}pre{white-space:pre-wrap;font-size:11px;overflow-x:auto}
summary{font-weight:600;padding:8px 0}
"""


def build_html(rows, bt, edge):
    from datetime import datetime
    from html import escape as e
    out = ["<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>Paris Foot</title>",
           "<meta name='viewport' content='width=device-width, initial-scale=1, viewport-fit=cover'>",
           f"<style>{CSS}</style></head><body><div class='w'>",
           "<h1>⚽ Paris Foot</h1>",
           f"<div class='m'>Mis à jour le {datetime.utcnow():%d/%m/%Y à %H:%M} UTC · "
           f"avantage minimum {edge*100:.0f}% · mise = 5% du capital</div>"]
    value = [(pk, r) for r in rows for pk in r["picks"][:1]]
    value.sort(key=lambda x: x[0][0], reverse=True)
    out.append(f"<h2>Paris à valeur ({len(value)})</h2>")
    if not value:
        out.append("<div class='c'>Aucun pari à valeur pour l'instant. Ne pas parier est une décision valable.</div>")
    for (ev, name, odds, prob), r in value:
        cls = "w2" if ev > 0.20 else "v"
        note = " · ⚠ trop beau pour être vrai, vérifie blessures/forme" if ev > 0.20 else ""
        out.append(f"<div class='c {cls}'><b>{e(r['home'])} - {e(r['away'])}</b>"
                   f"<div class='m'>{r['date']:%d/%m} · {r['lg']}</div>"
                   f"<div>Pari <b>{name}</b> à la cote <b>{odds:.2f}</b> · proba {prob*100:.0f}% · "
                   f"avantage <b>{ev*100:+.0f}%</b>{note}</div></div>")
    out.append(f"<h2>Tous les matchs ({len(rows)})</h2>")
    for r in rows:
        p = r["p"]
        flag = " ⚠ série" if max(r["sh"], r["sa"]) >= 3 else ""
        dnb1, dnb2 = (p["pH"] + p["pA"]) / p["pH"], (p["pH"] + p["pA"]) / p["pA"]
        out.append(
            f"<div class='c'><b>{e(r['home'])} - {e(r['away'])}</b>{flag}"
            f"<div class='m'>{r['date']:%d/%m} · {r['lg']} · buts attendus {p['lh']:.1f}-{p['la']:.1f} · "
            f"sans marquer {r['sh']}/{r['sa']}</div>"
            f"<div class='t'>1 {p['pH']*100:.0f}% · N {p['pD']*100:.0f}% · 2 {p['pA']*100:.0f}%</div>"
            f"<div class='t'><b>Cotes justes</b> : +1.5 {1/p['pO15']:.2f} · +2.5 {1/p['pO25']:.2f} · "
            f"-2.5 {1/(1-p['pO25']):.2f} · +3.5 {1/p['pO35']:.2f} · BTTS {1/p['pBTTS']:.2f} · "
            f"DNB dom {dnb1:.2f} · DNB ext {dnb2:.2f}</div></div>")
    if not rows:
        out.append("<div class='c'>Aucun match trouvé (trêve internationale ?).</div>")
    out.append("<p class='m'>Parie seulement si la cote de ton bookmaker dépasse la cote juste d'au moins 5%. "
               "Aucune méthode ne garantit de gagner.</p>")
    out.append(f"<details><summary>Résultat du backtest (5 saisons)</summary><pre>{e(bt)}</pre></details>")
    out.append("</div></body></html>")
    return "".join(out)


def site(edge, maxodds):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backtest(edge, maxodds)
    bt = buf.getvalue()
    df = load_history()
    _, m = run_model(df)
    rows = fixtures_rows(m, edge, maxodds)
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(build_html(rows, bt, edge))
    print(f"docs/index.html généré ({len(rows)} matchs).")


if __name__ == "__main__":
    args = sys.argv[1:]
    edge = float(args[args.index("--edge") + 1]) if "--edge" in args else 0.05
    maxodds = float(args[args.index("--maxodds") + 1]) if "--maxodds" in args else 4.0
    if args and args[0] == "backtest":
        backtest(edge, maxodds)
    elif args and args[0] == "predict":
        predict(edge, maxodds)
    elif args and args[0] == "site":
        site(edge, maxodds)
    else:
        print(__doc__)
