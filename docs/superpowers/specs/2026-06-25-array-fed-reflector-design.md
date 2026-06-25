# Modèle Array-Fed Reflector (AFR) — design

**Date :** 2026-06-25
**Statut :** validé, prêt pour plan d'implémentation

## Objectif

Doter `grd_generator` d'un **modèle physique réel d'antenne à réseau de feeds
illuminant un réflecteur** (*array-fed reflector*, AFR), conformément au besoin
`INIT.md` : « réseau d'antennes actives … passant par un réflecteur projeté sur
Terre », satellite géostationnaire, fréquence donnée en entrée.

Le modèle actuel synthétise des diagrammes gaussiens/Airy directement en espace
angulaire (u, v), sans réflecteur ni géométrie d'array physique. Ce design ajoute
un **module séparé** qui calcule les diagrammes secondaires à partir d'une
géométrie de réflecteur, d'un réseau de feeds et d'une fréquence. Les modes
phénoménologiques existants restent en place (oracle de test, comparaison).

## Décisions de cadrage (arbitrées avec l'utilisateur)

| Sujet | Décision |
|-------|----------|
| Modèle physique | **Array-fed reflector réel** |
| Méthode champ lointain | **Aperture-field + FFT** (optique géométrique). PO gardé en note future. |
| Réflecteur | **Offset paraboloïde simple** (diamètre D, focale F / rapport F/D, clearance d'offset). Pas de blocage. |
| Intégration | **Module séparé** `reflector/`, appelé par `generate` / `gui`. |
| Modèle de feed | **cos^q(θ)** paramétrable (q relié à l'edge taper). |
| Polarisation | **Vectoriel co/cross-pol (Ludwig-3).** Cross-pol calculée et stockée. |
| Fréquence | **Unique** (Hz/GHz), pilote λ (taille électrique d'ouverture, FFT, taper). |
| Export `.grd` | **Hors périmètre** (plan ultérieur) ; format documenté en note. |
| Zone de service | **Disque angulaire**, rayon `r` ∈ **[6°, 14°]**, centré sur le boresight. Affiché sur tous les displays. |
| Pointage par défaut | **Boresight = nadir** satellite ; feeds décalés pour paver la zone. |
| Contrainte couverture | **Enveloppe (max-combining) ≥ seuil (dBi) en tout point de la zone.** |
| Gate de tests | **`--cov-fail-under=100` conservé.** Script de génération AFR exclu de couverture (comme `generate.py`). |

## Modèle physique

### Géométrie (offset paraboloïde simple)

Réflecteur parent : paraboloïde `z = (x² + y²) / (4F)`. La section *offset* est la
portion d'ouverture projetée de diamètre `D` décalée du sommet par une hauteur de
clearance `h` (ou un angle d'offset équivalent). Le feed array est situé près de
la région focale ; chaque feed pointe vers le centre du réflecteur.

### Pointage et zone de service

- **Boresight système = nadir** (point sub-satellite). La zone de service est un
  disque angulaire centré en (u, v) = (0, 0), de rayon `r` ∈ [6°, 14°].
- Un feed **au foyer** produit un faisceau au boresight (nadir). Un feed
  **décalé** dans le plan focal produit un faisceau dépointé d'un angle ≈
  `BDF · (déplacement / F)` (*beam deviation factor*). Le **lattice de feeds**
  (pas, nombre) est donc choisi pour que les faisceaux adjacents se recouvrent à
  un niveau de crossover suffisant et **pavent** le disque de service.
- Tous les points de la zone doivent recevoir du signal : la directivité
  **enveloppe** (maximum atteignable en combinant les feeds, max-ratio combining)
  doit être ≥ un seuil en tout point du disque.

### Champ d'ouverture (optique géométrique)

Pour chaque feed `i` :
1. **Taper d'illumination** : amplitude du feed cos^q(θ) évaluée à l'angle
   sous-tendu par chaque point d'ouverture depuis le feed, × atténuation
   d'espace → distribution d'amplitude sur l'ouverture (edge taper).
2. **Tilt de phase** : un feed décalé impose un gradient de phase linéaire sur
   l'ouverture → dépointage du faisceau secondaire (BDF).
3. **Décomposition vectorielle** : réflexion du champ du feed sur la surface
   courbe → composantes `Ex(x, y)`, `Ey(x, y)` sur l'ouverture (la géométrie
   offset introduit une cross-pol systématique).

### Champ lointain (FFT + Ludwig-3)

- FFT 2D de `Ex` et `Ey` sur l'ouverture → champ lointain `Eθ`, `Eφ` sur la
  grille (u, v). Le pas d'échantillonnage / zero-padding fixe l'étendue et la
  résolution angulaire (relation pas ↔ étendue uv ≈ λ/D).
- Transformation **Ludwig-3** `(Eθ, Eφ) → (E_co, E_cross)`.
- Directivité dBi à partir de `|E_co|²` normalisé isotrope.

Sortie : pour chaque feed, un `ComplexField` co-pol (compatible `schemas.py`) sur
une `UVGrid`, + le champ cross-pol stocké en parallèle.

## Architecture

Nouveau sous-module `src/grd_generator/reflector/` :

```
reflector/
├── __init__.py        # API publique du modèle AFR
├── spec.py            # ReflectorSpec (D, F, offset, freq), FeedSpec (lattice, q/edge taper)
├── zone.py            # ServiceZone (rayon [6,14]°, centre nadir, masque uv, projection Terre)
├── optics.py          # GO : feed → champ d'ouverture vectoriel (taper + tilt), BDF, lattice→spacing ciel
├── farfield.py        # FFT 2D → Eθ/Eφ → Ludwig-3 → E_co/E_cross (UVGrid)
└── calibrate_afr.py   # dimensionnement du lattice : enveloppe ≥ seuil sur la zone (faisabilité)
```

Modules existants touchés :
- `generate.py` — option de génération via le modèle AFR : écrit le `.npz` avec
  les **mêmes clés** + champs cross-pol + métadonnées réflecteur (D, F/D, offset,
  freq, lattice, rayon de zone). Script, exclu de couverture.
- `gui.py` — panneau de paramètres réflecteur (D, F/D, offset, **fréquence**,
  edge taper, lattice) + **contrôle rayon de zone** ; réutilise le sélecteur
  d'élément / la carte de phase ; overlay du **cercle de zone** sur tous les
  panneaux.
- `plot.py` — étendre `draw_zone_and_antenna` pour tracer le disque de zone sur
  cartes (u, v), phase, enveloppe **et** globe.

Réutilisés tels quels : enveloppe (max-combining), rendu raster + isolignes,
projection globe, machinerie de combinaison.

## Flux de données

```
ReflectorSpec + FeedSpec[N] + freq + ServiceZone
   │  optics : feed → champ d'ouverture vectoriel (GO : taper cos^q + tilt BDF)
   ▼  Ex(x,y), Ey(x,y) par feed
farfield : FFT 2D → Eθ,Eφ → Ludwig-3 → E_co, E_cross (u,v)
   │  calibrate_afr : lattice tel que enveloppe ≥ seuil sur la zone (sinon ValueError)
   ▼
fields[N] (co) + cross[N] + zone → generate (.npz, clés existantes + cross + méta)
   ▼  enveloppe / plot / gui existants (overlay zone)
```

## Recherches à mener

| # | Sujet | Sortie attendue |
|---|-------|-----------------|
| R1 | Géométrie réflecteur offset (paraboloïde parent, section offset, F/D, clearance, projection d'ouverture) | Paramètres d'entrée + mapping géométrique |
| R2 | Array-fed reflector & **beam deviation factor** (déplacement feed → dépointage, scan loss, coma) ; lattice focal → espacement faisceaux sur le ciel → recouvrement | Loi feed→faisceau dépointé, loi pitch focal ↔ spacing ciel |
| R3 | Aperture-field GO + FFT (taper d'illumination, échantillonnage / zero-pad, relation pas ↔ étendue uv ≈ λ/D) | Algorithme far-field |
| R4 | Feed cos^q : q ↔ edge taper au bord du réflecteur ↔ efficacité d'ouverture / spillover | Paramétrage feed |
| R5 | Far-field vectoriel + **Ludwig-3** ; cross-pol induite par l'offset ; `Ex/Ey → Eθ/Eφ → co/cross` | Calcul polarisation |
| R8 | Mise à l'échelle en fréquence (λ → ouverture électrique, FFT, validité) | Branchement fréquence |
| R10 | **Critère de couverture de zone** : niveau de crossover (−3/−4 dB), dimensionnement du lattice pour enveloppe ≥ seuil sur disque de rayon `r`, **scan loss** en bord de zone (14°) | Critère + sizing |
| R9 | Validation : crête ≈ 10·log₁₀(η·(πD/λ)²), lobe ≈ k·λ/D, dépointage vs BDF | Tests de sanité physique |
| R6 | *(note)* Physical Optics (`Js = 2n×H`, intégration surfacique) | Voie future haute fidélité |
| R7 | *(note)* Format TICRA GRASP `.grd` | Plan d'export ultérieur |

Références de départ : Balanis *Antenna Theory* (réflecteurs, Ludwig-3) ; Rudge &
Adatia (array-fed reflectors, BDF) ; Imbriale *Spaceborne Antennas* ; notes TICRA
(GRASP, PO) pour R6/R7.

## Gestion d'erreurs

- `calibrate_afr` lève `ValueError` si la couverture (enveloppe ≥ seuil sur la
  zone) est infaisable avec le lattice / la fréquence donnés.
- `ServiceZone` valide `radius_deg ∈ [6, 14]`.
- `ReflectorSpec` valide `D > 0`, `F > 0`, `freq > 0` ; `FeedSpec` valide `q ≥ 0`.
- La zone doit tenir sous le limbe terrestre (réutilise la vérification existante
  de `sat_frame_geometry`).

## Tests (gate 100 %)

- **Ouverture circulaire uniforme → Airy** : crête = (πD/λ)², 1er null à
  1.22 λ/D (recoupe le mode `airy` existant comme oracle).
- Feed cos^q : crête à θ=0, edge taper attendu au bord du réflecteur.
- Ludwig-3 : entrée co-pol pure → cross-pol ≈ 0 ; offset → cross-pol non nulle.
- Fréquence : largeur de lobe ∝ λ (deux fréquences).
- BDF : un feed décalé produit un faisceau à la position attendue.
- `ServiceZone` : masque (u, v) correct, disque projeté sur la Terre, bornes
  [6, 14] validées.
- Couverture : `envelope_min` sur la zone ≥ seuil pour le lattice par défaut ;
  lattice trop épars → `ValueError` (faisabilité).
- Affichage : cercle de zone tracé sur cartes (u, v), phase, enveloppe et globe.
- `calibrate_afr` / script de génération AFR exclus de couverture (comme
  `generate.py`).

## Dépendances

Runtime inchangé : `numpy`, `pydantic`, `loguru` ; `matplotlib` + `PyQt6` pour la
GUI (déjà présents). FFT via `numpy.fft` (pas de scipy). Décision de ne pas
ajouter scipy conservée.

## Hors périmètre (notes pour plus tard)

- Physical Optics (R6) — fidélité supérieure (lobes, spillover) mais coûteux,
  inadapté au redraw interactif. Documenté comme évolution.
- Export `.grd` TICRA GRASP (R7) — la cross-pol est déjà stockée dans le `.npz`
  pour qu'un plan ultérieur la consomme.
- Balayage multi-fréquences.
- Réflecteur symétrique / dual ; visée mécanique vers une lat/lon arbitraire.
