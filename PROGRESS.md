# Progression — lot multi-fréquence, export HDF5 (branche `feat/batch-ku-hdf5`)

Spec : `docs/superpowers/specs/2026-09-17-batch-ku-hdf5-design.md`.

## État

- [x] **B1** — `reflector/hdf5_export.py` (écriture HDF5, structure de type TICRA
  Tools 25.0, échange H/V) + dépendance `h5py`.
- [x] **B2** — `BatchConfig`, `reflector/batch.py`, entrée `grd-batch`, arborescence
  de sortie, manifeste, NOTICE. Exécution réelle vérifiée (91 feeds x 2 fréquences,
  grille 161x161, `n_aperture=128`, `pad_factor=8` : ~42 s, ~826 KiB/fichier).
- [x] **B3** — `reflector/metrics.py`, `--check` (batch.py), `examples/example_batch_ku.py`
  (+ `plot.draw_calibration_envelope_map`) + test de lancement + README.

## Pour reprendre

```bash
uv run pytest -q            # suite complète, 100 % de couverture exigée
uv run ruff check .
uv run mypy src
```

## Décisions / écarts notés en cours de route

- `write_grd_hdf5` : `track_times=False` sur chaque dataset et `libver="earliest"` sur le fichier —
  nécessaire pour la reproductibilité octet pour octet (testée en traversant une seconde d'horloge
  réelle dans `test_write_grd_hdf5_is_byte_for_byte_reproducible`). Sans `track_times=False`, HDF5
  embarque un horodatage de modification par dataset.
- `swap_components` exposée comme fonction pure séparée (échange exact, testable sans E/S), utilisée
  par l'appelant (B2/`batch.py`) pour fabriquer les arguments du fichier V à partir des champs H.
- Ajout d'un override mypy `h5py.*` (`ignore_missing_imports`), même traitement que matplotlib.
- `BatchConfig` (B2) : n'impose aucune contrainte de valeur sur `n_feeds` (juste `>0`) — le
  docstring de `FeedsConfig` documente que seul un nombre hexagonal centré (1, 7, 19, 37, 61, 91,
  127, 169, … = 3k²−3k+1) donne un amas symétrique complet ; toute autre valeur tronque le dernier
  anneau de façon dépendante de l'ordre d'énumération, pas d'un critère de symétrie.
- `meta/batch-config.json` est une copie **octet pour octet** du texte du fichier de config fourni
  (pas une re-sérialisation du modèle pydantic) : les champs additionnels non modélisés (ex. un
  futur bloc `_provenance` figé au calibrage) survivent sans que `BatchConfig` ait besoin de les
  connaître (comportement pydantic par défaut : champs inconnus ignorés, pas refusés).
- `zone_radius_deg` de `meta/sim_params.json` (paramètre requis par `simulation_params_dict`,
  réutilisé tel quel) est dérivé du bord de la grille (`asin` du plus petit demi-côté) : purement
  informatif, ce n'est pas une contrainte du batch (qui n'a pas de notion de zone de service propre).
- Refus « dossier de sortie non vide » vérifié AVANT toute écriture (test dédié : le seul fichier
  sentinelle préexistant reste seul présent après l'échec).
- `metrics.py` (B3) : « largeur à −3 dB (diamètre équivalent) » calculée via l'angle solide EXACT
  de la région ≥ crête−3 dB (`Σ du·dv/cosθ`, même formule que `farfield.normalize_to_directivity`),
  converti en diamètre de calotte sphérique équivalente (`Ω=2π(1−cosα)`) — exact quel que soit le
  centrage de la région sur la sphère (symétrie de rotation), pas seulement pour un lobe centré au
  nadir. Testé exactement contre un lobe gaussien analytique centré au nadir (contour −3 dB = cercle
  exact en (u,v) ↔ calotte exacte).
- `guaranteed_radius_deg` (rayons −1/−3 dB et ≥39 dBi de l'enveloppe) = rayon du nœud en échec le
  plus proche du nadir (garantie sur tout disque plus petit), pas un rayon le long d'un seul axe.
- `--check` ne bloque PAS l'écriture (contrairement aux refus B2) : le jeu est toujours écrit,
  seul le code retour change — conforme à la spec (« code retour 1 si un critère bloquant… »,
  pas de « sans rien écrire » pour ce cas précis).
- `example_batch_ku.py --figure` restitue la combinaison n°1 du balayage (listes à un seul élément
  par axe pour choisir précisément la figure) ; overlay du plan de mission via
  `(u,v) = (sin θx, sin θy)` (conversion par axe, PAS `asin(hypot(u,v))` qui est l'inverse utilisé
  côté grille → degrés).
- `print()` délibéré dans `example_batch_ku._print_table` (résultat du script, pas un log de statut
  — ceux-ci restent en `logger.info`/`logger.debug` loguru) : voir le commentaire en tête de fonction.
- `GENERATOR_VERSION` (batch.py) est dérivée de la version du paquet installé
  (`importlib.metadata.version("grd_generator")`), pas une constante dupliquée en dur.
