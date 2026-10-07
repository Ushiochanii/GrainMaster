# P0 module boundaries

Core algorithms are in `src/grainmaster/`; scripts are entry points.

| Module | Input | Output |
|---|---|---|
| `ruler.py` | Full uint8 BGR image | SpatialCalibration plus ruler overlay |
| `color_calibration.py` | Full uint8 BGR image | ColorCalibration, corrected BGR and chart overlay |
| `dishes.py` | Original full BGR image | DishROI list and unchanged-scale crops |
| `segmentation.py` | Original dish crop | SeedInstance list with full-crop boolean masks |
| `phenotyping.py` | Mask, corrected crop, spatial calibration | SeedTraits, per-dish aggregation |
| `pipeline.py` | Image path, YAML config | Calibration, CSV, masks, review preview and status |

`contracts.py` is a small dataclass interface. Bboxes use `(x,y,width,height)`. Centers and
ellipses are in original full-image pixels. Crops are translated without resizing. Masks
use crop coordinates. Dish IDs follow rows, left to right; seed IDs are local to each dish.

Segmentation uses original BGR crops; phenotyping uses corrected BGR crops. Preview overlays
the original full photograph.

No human training data exists yet. Classical watershed is a provisional P0 backend.
YOLO26s-seg at 1024 requires a trained seed checkpoint; pretrained COCO is a smoke test only.
Preparation retains annotation provenance and splits by original photograph.

Modules have independent synthetic tests and real outputs. The integration test processes
local 0462.jpg, checking completed outputs, finite measurements, consistent aggregation,
mask labels and four dishes. It cannot verify biological accuracy without human references.

Batch stage statuses are `ok`, `failed`, or `not_run`. Failure stops that photograph, retains
diagnostics, and continues the batch. Completion does not certify count or mask accuracy.
Artifacts are not tracked in Git.


## Secondary reference and diagnostic geometry

`card_scale.py` consumes the color chart geometry and returns original-image millimetre
scale tick coordinates, lattice indices and residual. `geometry.py` fits a conditional
chart-grid planar candidate and checks the long ruler, which is held out of fitting.
The pipeline saves diagnostics, flags scale disagreement, and never applies the candidate
to P0 measurements. This is independent of the single global color transform. It is not
camera calibration and does not estimate radial/tangential lens coefficients.
