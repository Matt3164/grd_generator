# Visualisation matplotlib des arrays générés

**Date :** 2026-06-22
**Statut :** validé, implémentation directe en session

## Objectif

Un script qui **affiche** le contenu d'un `.npz` (référence ou calibré) en
matplotlib : un **rendu direct** en espace angulaire (u, v) et un **rendu sur
Terre** (projection géographique avec côtes), fonctionnant **hors-ligne**.

## Décisions de cadrage (arbitrées)

| Sujet | Décision |
|-------|----------|
| matplotlib | **Extra optionnel `viz`** (`uv sync --extra viz`). Le cœur génération reste numpy/pydantic/loguru. `plot.py` + `examples/*` exclus de couverture (gate 100% préservé) + test de lancement (backend Agg). |
| Rendu direct | **Élément #0 + enveloppe** : deux cartes de directivité (dBi) en (u, v). |
| Rendu Terre | Projection (u,v)→(lat,lon) ; borésight **nadir par défaut** (sat_lon réglable, défaut 3°E), flags `--sat-lon/--boresight-lat/--boresight-lon`. |
| Côtes | **GeoJSON Natural Earth (50 m) vendored** dans `data/ne_50m_land.geojson` (1.6 Mo), tracé en matplotlib pur. **Aucune dépendance réseau / cartopy.** |

## Architecture

```
src/grd_generator/
├── plot.py                 (NOUVEAU, exclu de couverture) rendu + CLI grd-plot
└── examples/
    └── example_plot.py     (NOUVEAU, exclu) génère une référence et la trace
data/ne_50m_land.geojson    (vendored, côtes 50 m)
```

`plot.py` réutilise :
- `synth.directivity_dbi_from_field` (élément), `synth.combined_max_directivity_dbi`
  (enveloppe = `10·log10(Σ|Eᵢ|²)`, l'empreinte atteignable) ;
- `geometry.earth_intersection_latlon` (projection (u,v)→(lat,lon) + masque disque).

### API

- `load_array(path) -> tuple[grid_bounds, n_u, n_v, fields]` : lit le `.npz`
  (clés `u_min/u_max/v_min/v_max/n_u/n_v/fields`).
- `coastline_rings() -> list[NDArray]` : anneaux extérieurs (lon, lat) du GeoJSON.
- `render(npz_path, *, out=None, element=0, sat_lon=3.0, boresight_lat=0.0,
  boresight_lon=None, show=True) -> Path` : figure 1×3
  (élément u,v | enveloppe u,v | enveloppe sur Terre + côtes). `boresight_lon`
  défaut = `sat_lon` (nadir). Sauve un PNG (`out` défaut = `<npz>.png` à côté du
  `.npz`) **et** `plt.show()` (no-op sous Agg).
- `main()` : argparse `--npz` (requis), `--out`, `--element`, `--sat-lon`,
  `--boresight-lat`, `--boresight-lon`, `--no-show`.

### Rendu Terre — détail

1. `meshgrid` de la grille (u,v) → `earth_intersection_latlon(u, v, sat_lon,
   boresight_lat, boresight_lon)` → (lat, lon, hit).
2. Enveloppe dBi en `scatter(lon[hit], lat[hit], c=env[hit])` (robuste aux NaN
   hors disque ; pas de quad mesh sur coordonnées NaN).
3. Côtes : tracé des anneaux (lon, lat) en lignes fines.
4. Limites d'axes = bbox de l'empreinte + marge → zoom sur la zone éclairée.

## Gestion d'erreurs

- `load_array` : `ValueError` si une clé attendue manque dans le `.npz`.
- `render` : `IndexError` clair si `element` ≥ n (message explicite).
- matplotlib absent (extra non installé) : `import` échoue avec message invitant
  à `uv sync --extra viz` (try/except à l'import de `plot.py`).

## Tests

- `plot.py` et `examples/*` **exclus de couverture** (rendu graphique, effets de
  bord). Gate 100% préservé sur le reste.
- `test_example_plot.py` : test de lancement (`MPLBACKEND=Agg`, `--out` tmp),
  code retour 0 et PNG écrit.

## Dépendances

`[project.optional-dependencies] viz = ["matplotlib>=3.9"]`. Cœur inchangé.
GeoJSON vendored (pas de réseau).
</content>
