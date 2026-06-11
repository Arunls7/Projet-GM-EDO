# Résolution d'une équation différentielle d'ordre deux

**Projet GM & MF — ING1 · CY Tech · 2025–2026**

Auteurs : Arun Kuganesan · Haytham Bouhou · Jérémie Konda · Amine Taki · Ulysse Beqiraj

---

## Problématique

On étudie l'équation différentielle d'ordre deux

```
-Q'' + μ Q + g(Q) = 0   sur ℝ,    Q(-∞) = c,   Q(+∞) = 0,
```

où `g ∈ C²(ℝ)` vérifie `g(0) = g'(0) = 0`. La question centrale : pour quels
paramètres `μ, c > 0` existe-t-il une solution réalisant une **transition**
(un *front*) entre les paliers `c` et `0` ?

On démontre qu'une telle solution existe **si et seulement si** la fonction
`P(x) = -2G(x)/x²` admet un maximum intérieur en `c` (avec `P(c) > 0`), auquel
cas `μ = P(c)` et la solution est unique à translation près. Le problème se
rattache aux fronts des équations de réaction–diffusion de type **Allen–Cahn /
Nagumo**.

## Contenu du dépôt

```
.
├── rapport.pdf          # Rapport complet (analyse, synthèse, implémentation, annexe, biblio)
├── rapport.tex          # Source LaTeX du rapport
├── src/
│   └── edo_solver.py    # Code Python commenté (algorithme + solveur + figures)
├── figures/             # Figures générées (PNG)
├── slides/
│   └── soutenance.pptx  # Présentation de soutenance
└── README.md
```

## Implémentation

Deux fonctions principales dans `src/edo_solver.py` :

- `find_c_values(g)` : renvoie les couples `(c, μ)` admissibles (détection des
  maxima intérieurs de `P`, intégration via `scipy.integrate.quad`).
- `solve_Q(g, c, mu)` : intègre `Q' = -√(μQ² + 2G(Q))` avec
  `scipy.integrate.solve_ivp` (RK45) et trace le profil.

Une **solution analytique exacte** (forme logistique / tangente hyperbolique) est
disponible dans le cas bistable et sert à valider le code : l'écart maximal
mesuré entre solution numérique et solution exacte est de l'ordre de `10⁻¹⁰`.

## Exécution

Prérequis : Python ≥ 3.9 avec `numpy`, `scipy`, `matplotlib`.

```bash
pip install -r requirements.txt
python src/edo_solver.py
```

Le script affiche les couples `(c, μ)` détectés sur les exemples et régénère
toutes les figures du rapport.

## Exemple de référence

Non-linéarité bistable `g(t) = ½t³ - (3/2)t²`, soit une réaction
`μy + g(y) = ½ y(y-1)(y-2)`. L'algorithme détecte automatiquement `c = 2`,
`μ = 1`, et le profil exact vaut `Q(x) = c / (1 + e^{√μ x})`.

## Références principales

1. S. M. Allen, J. W. Cahn, *Acta Metallurgica*, 27(6), 1085–1095, 1979.
2. P. C. Fife, J. B. McLeod, *Arch. Rational Mech. Anal.*, 65(4), 335–361, 1977.
3. D. G. Aronson, H. F. Weinberger, *Advances in Mathematics*, 30(1), 33–76, 1978.
4. A. Kolmogorov, I. Petrovsky, N. Piskunov, *Bull. Univ. Moscou*, 1937.

(Liste complète dans `rapport.pdf`.)
