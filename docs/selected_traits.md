# Selected web phenotypes

The workbench groups descriptors into size, shape and color. Measurements derive
from immutable existing native-resolution masks/traits; original pipeline CSVs
are not rewritten. Web seed/dish exports include the new descriptors.

Size retains ellipse major/minor axis (mm) and raster projected area (mm²).
Shape exposes only aspect_ratio=L/W, circularity=4πA/P² and solidity=A/hull_area.
For the latter two ratios, A is the sum of external contour polygon areas, P is
the sum of contour perimeters, and the convex hull encloses all exterior points.
This avoids mixing raster pixel area with polygon hull area. Tiny degenerate
contours return null, not fabricated values. Fragmented masks remain QC candidates.
Perimeter-dependent circularity is sensitive to mask resolution and edge roughness.
No dimensionless ratio guarantees immunity to perspective or anisotropic distortion.

Color retains interior median Lab D65 per seed, adding C*=hypot(a*,b*) and
h=atan2(b*,a*) in [0,360). Hue is null at effectively zero chroma (<=1e-6).
Dish representative C*/h derive from its equal-seed mean Lab; angles are never
arithmetically averaged. These are derived views of Lab, not independent information.

Within-dish ΔE00 uses componentwise median Lab of the included seed set as reference.
The UI displays seed-level distances, mean and 90th percentile. All plots share
one horizontal scale within the current photo/mode. Draft excludes discarded;
confirmed includes only confirmed. The reference and distribution recompute when
inclusion changes. n<2 returns null; individual and dish CSVs preserve reference
coordinates and the definition. This is relative color dispersion, not color
measurement accuracy, disease or maturity. Physical specimens/batches and their
biological meanings must come from the experimental metadata.

Brightness span, within-seed variation, illumination and calibration diagnostics
are not presented as selected biological phenotypes. Existing calibration remains
available under its own panel. Same-device capture supports consistency but does
not establish that all residual error is a fixed systematic offset.

References: GrainScan (2014), https://doi.org/10.1186/1746-4811-10-23;
phenoSEED (2020), https://doi.org/10.1186/s13007-020-00591-8;
ImageJ definitions, https://imagej.net/ij/docs/guide/146-30.html.
