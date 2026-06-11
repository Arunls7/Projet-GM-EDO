# -*- coding: utf-8 -*-
"""
============================================================================
 Projet ING1 - Math Appliquees & Science des Donnees (CY Tech)
 Resolution de l'EDO d'ordre 2 :  -Q'' + mu*Q + g(Q) = 0
 avec les conditions aux limites  Q(-inf) = c  et  Q(+inf) = 0.
----------------------------------------------------------------------------
 Idee generale (ce qu'on a compris du sujet) :
   - On multiplie l'equation par Q' et on integre : ca fait apparaitre une
     "energie" H(x) qui est conservee (H' = 0). Comme Q et Q' tendent vers 0
     en +inf, cette energie vaut 0 partout. On en deduit Q'^2 = f(Q) avec
     f(y) = mu*y^2 + 2*G(y). On est donc passe d'une EDO d'ordre 2 a une
     EDO d'ordre 1, beaucoup plus simple a integrer.
   - La fonction P(x) = -2*G(x)/x^2 sert a reperer les bons parametres :
     un couple (c, mu) marche ssi P'(c)=0, P(c)>0 et P < P(c) sur [0,c[.
 Auteurs : Bouhou, Konda, Kuganesan, Taki, Beqiraj.
============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# ----------------------------------------------------------------------
# Style des figures (on met tout ici pour que les graphes soient lisibles
# aussi bien dans le rapport que sur les slides de soutenance).
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,    # on enleve les bords du haut/droite : plus propre
    "axes.spines.right": False,
})

BLEU   = "#1f5fa8"
ROUGE  = "#c0392b"
VERT   = "#218c5a"
ORANGE = "#e08a1e"


# ----------------------------------------------------------------------
# Briques de base : G et f
# ----------------------------------------------------------------------
def G_num(g, y):
    """G(y) = integrale de g entre 0 et y.

    On utilise quad (quadrature de scipy) car g est quelconque : on ne peut
    pas toujours calculer la primitive a la main. Petite precaution : en y=0
    l'integrale vaut 0, on le renvoie directement pour eviter une division
    par 0 plus loin dans P.
    """
    if abs(y) < 1e-12:                 # cas y ~ 0 : G(0) = 0 par definition
        return 0.0
    val, _ = quad(g, 0.0, y, limit=200)  # le "_" est l'erreur estimee, on l'ignore
    return val


def f_func(g, mu, y):
    """f(y) = mu*y^2 + 2*G(y).  Rappel : l'energie donne Q'^2 = f(Q)."""
    return mu * y**2 + 2.0 * G_num(g, y)


# ----------------------------------------------------------------------
# 1) Detection des parametres admissibles (c, mu)
# ----------------------------------------------------------------------
def find_c_values(g, c_max=5.0, n_grid=4000, tol=1e-7):
    """Renvoie tous les couples (c, mu) qui passent l'algorithme.

    Conditions a verifier (cf. partie theorique) :
        (i)   P'(c) = 0      -> c est un MAXIMUM interieur de P
        (ii)  P(c) > 0
        (iii) P(x) < P(c) pour tout x dans [0, c[
    et alors on pose mu = P(c).

    Strategie : on echantillonne P sur une grille, on derive numeriquement,
    puis on cherche les endroits ou P' passe de + a - (= sommet de P).
    Remarque importante : si g ne convient pas (ex. g(t) = -t^3), P est
    monotone, il n'y a aucun sommet interieur et la fonction renvoie []
    -> c'est le comportement attendu : pas de solution dans ce cas.
    """
    # --- on construit P sur la grille ---
    xs = np.linspace(1e-3, c_max, n_grid)        # on part de 1e-3 (pas de 0 : division)
    G_vals = np.array([G_num(g, x) for x in xs]) # G(x) en chaque point
    P_vals = -2.0 * G_vals / xs**2               # definition de P
    P_prime = np.gradient(P_vals, xs)            # derivee approchee (diff. finies)

    resultats = []
    for i in range(1, len(xs) - 1):
        mu = P_vals[i]
        if mu <= 0:                              # condition (ii) : on veut mu > 0
            continue
        # condition (i) : sommet local -> P' change de signe + vers -
        if P_prime[i - 1] > 0.0 and P_prime[i + 1] < 0.0:
            c = xs[i]
            # condition (iii) : on verifie qu'aucun point de ]0,c[ ne depasse P(c).
            # Astuce : pres de c, P remonte tres vite vers mu (de facon quadratique),
            # donc on tolere l'egalite a 1e-6 pres pour ne pas rater le vrai sommet.
            x_test = np.linspace(1e-4, c * (1 - 1e-4), 400)
            P_test = np.array([-2.0 * G_num(g, x) / x**2 for x in x_test])
            if np.max(P_test) <= mu + 1e-6:
                resultats.append((float(c), float(mu)))

    # Nettoyage : un meme sommet peut etre detecte sur deux points voisins
    # de la grille. On fusionne les candidats trop proches (on garde le
    # meilleur, celui dont P est le plus grand).
    resultats.sort(key=lambda cm: cm[0])
    fusionnes = []
    for c, mu in resultats:
        if fusionnes and abs(c - fusionnes[-1][0]) < 0.05:
            if mu > fusionnes[-1][1]:
                fusionnes[-1] = (c, mu)
        else:
            fusionnes.append((c, mu))
    return fusionnes


# ----------------------------------------------------------------------
# 2) Integration du profil Q
# ----------------------------------------------------------------------
def solve_Q(g, c, mu, L=18.0, n_eval=2000):
    """Resout Q' = -sqrt(f(Q)) et renvoie (xs, Qs) pour tracer le profil.

    On part du milieu Q(0) = c/2 (un point ou f > 0, donc tout va bien) puis
    on integre DANS LES DEUX SENS : vers la droite Q descend vers 0, vers la
    gauche Q remonte vers c. On borne Q dans [0, c] : comme f a un zero double
    a chaque bord, les paliers sont des asymptotes -> pas d'explosion (c'est ce
    qui plantait quand on prenait un mauvais g au depart).
    """
    def rhs(x, state):
        Q = min(max(state[0], 0.0), c)       # on force Q a rester dans [0, c]
        val = max(f_func(g, mu, Q), 0.0)     # securite : pas de racine de negatif
        return [-np.sqrt(val)]               # signe "-" car Q est decroissante

    # tolerances tres serrees : on veut une solution propre pres des paliers
    opts = dict(max_step=0.02, dense_output=True, rtol=1e-10, atol=1e-12)
    sol_d = solve_ivp(rhs, [0.0,  L], [c / 2.0], **opts)   # cote x >= 0
    sol_g = solve_ivp(rhs, [0.0, -L], [c / 2.0], **opts)   # cote x <= 0

    # on recolle les deux morceaux sur une grille commune
    xs = np.linspace(-L, L, n_eval)
    Qs = np.where(xs >= 0, sol_d.sol(xs)[0], sol_g.sol(xs)[0])
    Qs = np.clip(Qs, 0.0, c)
    return xs, Qs


# ----------------------------------------------------------------------
# 3) Solution analytique exacte (cas bistable symetrique)
#    -> sert UNIQUEMENT a verifier que notre code numerique est correct.
# ----------------------------------------------------------------------
def Q_exact_bistable(x, c, mu, x0=0.0):
    """Quand f(y) = (mu/c^2) y^2 (c-y)^2, l'EDO d'ordre 1 devient logistique
    et on sait la resoudre a la main : c'est une tangente hyperbolique.
        Q(x) = c / (1 + exp( sqrt(mu) * (x - x0) ))
    """
    return c / (1.0 + np.exp(np.sqrt(mu) * (x - x0)))


# ----------------------------------------------------------------------
# Fonctions g testees
# ----------------------------------------------------------------------
def g_bistable(t):
    """Notre exemple principal : g(t) = (1/2) t^3 - (3/2) t^2.
    Le terme de reaction mu*y + g(y) = (1/2) y (y-1)(y-2) est BISTABLE
    (trois zeros : 0, 1, 2). C'est le cas type d'Allen-Cahn.
    On verifie facilement g(0) = g'(0) = 0, et l'algo trouve c = 2, mu = 1.
    """
    return 0.5 * t**3 - 1.5 * t**2


def g_bistable3(t):
    """Meme famille mais avec c = 3 (reaction = y (y - c/2)(y - c)).
    On obtient mu = c^2/2 = 4.5. Sert a montrer un 2e profil different.
    """
    c = 3.0
    reaction = t * (t - c / 2.0) * (t - c)   # = mu*y + g(y)
    mu = c**2 / 2.0
    return reaction - mu * t                 # on isole g = reaction - mu*y


def g_cubique_naif(t):
    """CONTRE-EXEMPLE : g(t) = -t^3. Ici P(x) = x^2/2 est croissante, donc
    PAS de sommet interieur -> pas de solution. On le garde pour le montrer.
    """
    return -t**3


# ======================================================================
# GENERATION DES FIGURES (une fonction par figure du rapport)
# ======================================================================
def fig_P(g, label_g, c_max, fname, attendu=None):
    """Trace P et marque le sommet (c, mu) detecte par l'algorithme."""
    xs = np.linspace(1e-3, c_max, 1500)
    P = np.array([-2.0 * G_num(g, x) / x**2 for x in xs])
    couples = find_c_values(g, c_max=c_max)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, P, color=BLEU, lw=2.2, label=r"$P(x) = -\,2G(x)/x^2$")
    for c, mu in couples:                    # boucle vide si pas de solution
        ax.scatter([c], [mu], color=ROUGE, zorder=5, s=70)
        ax.annotate(rf"$c \approx {c:.2f},\ \mu = P(c) \approx {mu:.2f}$",
                    (c, mu), textcoords="offset points", xytext=(-10, -22),
                    ha="right", color=ROUGE, fontsize=12)
        ax.axhline(mu, color=ROUGE, lw=0.8, ls=":", alpha=0.6)
        ax.fill_between(xs, P, mu, where=(xs < c), color=VERT, alpha=0.10)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    if couples:
        titre = rf"Fonction $P$ pour {label_g}"
    else:
        titre = rf"$P$ pour {label_g} : pas de max intérieur $\Rightarrow$ pas de solution"
    ax.set_title(titre, fontsize=13)
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return couples


def fig_profil_vs_exact(fname):
    """Compare la solution numerique a la formule exacte (test de validation)."""
    c, mu = 2.0, 1.0
    xs, Qs = solve_Q(g_bistable, c, mu)
    Qe = Q_exact_bistable(xs, c, mu)
    err = np.max(np.abs(Qs - Qe))            # ecart max : doit etre minuscule

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, Qs, color=BLEU, lw=2.6, label="solution numérique (RK45)")
    ax.plot(xs, Qe, color=ROUGE, lw=1.4, ls="--",
            label=r"solution exacte $Q(x)=\dfrac{c}{1+e^{\sqrt{\mu}\,x}}$")
    ax.axhline(c, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.text(-17, c + 0.05, rf"palier $c = {c:.0f}$", fontsize=11, alpha=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$Q(x)$")
    ax.set_title(rf"Profil de transition  ($c={c:.0f},\ \mu={mu:.0f}$)"
                 rf"  —  écart max $= {err:.1e}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return err


def fig_deux_profils(fname):
    """Deux fronts pour deux valeurs de c, detectees automatiquement."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for g, lab, col in [(g_bistable, r"$c=2,\ \mu=1$", BLEU),
                        (g_bistable3, r"$c=3,\ \mu=4.5$", ORANGE)]:
        couples = find_c_values(g, c_max=5.0)
        c, mu = couples[0]                   # on prend le 1er couple trouve
        xs, Qs = solve_Q(g, c, mu)
        ax.plot(xs, Qs, lw=2.4, color=col,
                label=rf"{lab}  (détecté : $c\approx{c:.2f}$)")
        ax.axhline(c, color=col, lw=0.7, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$Q(x)$")
    ax.set_title("Profils de transition pour deux non-linéarités bistables")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_translation(fname):
    """Illustre l'unicite a translation : on decale le front et c'est toujours
    une solution (on utilise la formule exacte pour des courbes parfaites)."""
    c, mu = 2.0, 1.0
    xs = np.linspace(-18, 18, 1500)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for a, col in [(-5, BLEU), (0, VERT), (5, ROUGE)]:
        ax.plot(xs, Q_exact_bistable(xs, c, mu, x0=a), lw=2.2, color=col,
                label=rf"$Q(\cdot + a),\ a = {a}$")
    ax.axhline(c, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$Q(x)$")
    ax.set_title("Invariance par translation : toute translatée est encore solution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_phase(fname):
    """Portrait de phase (Q, Q') : on voit l'orbite qui relie 0 a c."""
    c, mu = 2.0, 1.0
    # L'orbite verifie Q'^2 = f(Q), donc Q' = -sqrt(f(Q)) sur [0, c].
    ys = np.linspace(0, c, 400)
    fy = np.array([f_func(g_bistable, mu, y) for y in ys])
    fy = np.clip(fy, 0, None)
    Qp = -np.sqrt(fy)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    # champ de vecteurs leger en fond (juste pour le visuel)
    Y, V = np.meshgrid(np.linspace(-0.4, c + 0.4, 22),
                       np.linspace(-1.1, 1.1, 22))
    dV = np.array([[mu * y + g_bistable(y) for y in row] for row in Y])  # Q'' = mu Q + g(Q)
    speed = np.hypot(V, dV)
    ax.quiver(Y, V, V / speed, dV / speed, color="#b9c4d4",
              angles="xy", scale=32, width=0.0025)
    ax.plot(ys, Qp, color=ROUGE, lw=2.8, label="orbite hétérocline")
    ax.scatter([0, c], [0, 0], color="k", zorder=5, s=55)
    ax.annotate("équilibre 0", (0, 0), xytext=(6, 10),
                textcoords="offset points", fontsize=11)
    ax.annotate("équilibre $c$", (c, 0), xytext=(-70, 10),
                textcoords="offset points", fontsize=11)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$Q'$")
    ax.set_title("Plan de phase : connexion des deux équilibres")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    import os
    out = "figures"
    os.makedirs(out, exist_ok=True)

    # Petit test rapide : on affiche ce que l'algo trouve sur nos 3 exemples.
    print(">>> find_c_values sur les exemples :")
    print("  g bistable (c=2)   :", find_c_values(g_bistable, c_max=5.0))
    print("  g bistable (c=3)   :", find_c_values(g_bistable3, c_max=5.0))
    print("  g = -t^3 (naif)    :", find_c_values(g_cubique_naif, c_max=5.0))

    # Generation de toutes les figures du rapport.
    fig_P(g_bistable, r"$g(t)=\frac{1}{2} t^3-\frac{3}{2} t^2$", 4.0,
          f"{out}/fig1_P_bistable.png")
    fig_P(g_cubique_naif, r"$g(t)=-t^3$", 4.0,
          f"{out}/fig2_P_echec.png")
    err = fig_profil_vs_exact(f"{out}/fig3_profil_vs_exact.png")
    fig_deux_profils(f"{out}/fig4_deux_profils.png")
    fig_translation(f"{out}/fig5_translation.png")
    fig_phase(f"{out}/fig6_phase.png")

    print(f"\n>>> Validation : ecart max numerique/exact = {err:.2e}")
    print(">>> Figures generees avec succes.")
