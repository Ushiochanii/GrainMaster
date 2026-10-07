# GrainMaster Mozume visual identity

Author credit: **Sun Junhao**. All interface copy is English.

The identity combines a wheat kernel, a dish outline and measurement ticks.
Instrument navy (`#193d50`), dish teal (`#167b89`), grain gold (`#d7b576`) and
cool paper (`#f5f7f9`) match the workbench. The banner is a GPT Image illustration,
not a measured sample or an experimental result. The original geometric icon is an editable SVG; the current white photographic icon and wordmark use embedded raster artwork.

Assets live in `web/assets/`:

- `grainmaster-icon.svg`: primary mark and browser favicon.
- `grainmaster-icon-32.png`: PNG favicon fallback.
- `grainmaster-icon-180.png`: Apple touch icon.
- `grainmaster-icon-192.png`, `grainmaster-icon-512.png`: application icons.
- `grainmaster-wordmark.svg`: self-contained horizontal wordmark.
- `grainmaster-banner.png`: original generated banner, 2172 × 724 px.
- `site.webmanifest`: browser application metadata. It does not add offline analysis.

The main workbench uses a branded header with About and a fixed lower-right author credit. About shows the
banner with accessible HTML branding and author text, plus asset downloads.
The illustration is kept outside the measurement canvas to avoid obscuring data.

Before distribution, verify fresh loads, image import, dish navigation, mask layers,
measurement/export access, label-status feedback, keyboard focus and mobile layout.
This visual polish does not change analysis algorithms, model weights or validation claims.

## Homepage revision

The homepage header uses the banner as a continuous full-width background at
27% opacity, with a light overlay behind text and controls. The product name and
Seed phenotyping descriptor share one line. About stays in the header, while
`by Sun Junhao` is fixed at the lower-right viewport corner, without requiring
scrolling to a footer. The header and identity were checked at desktop widths
and mobile widths down to 320 px. The new white icon is derived with GPT Image from the
banner's kernel and teal circle; navy tile artwork is retained only as a previous
asset variant, not used in the current interface.

Current icon assets use the `grainmaster-icon-white` prefix. The `-source.png`
is the original generated asset; 32, 180, 192 and 512 pixel exports support browser
and mobile use. The white SVG wrappers embed the image and are self-contained;
the photographic kernel itself is a raster illustration, not a vector path.
`grainmaster-wordmark-white.svg` is the matching downloadable wordmark.

Layout reference: [Carbon global header](https://www.carbondesignsystem.com/building-blocks/core/patterns/global-header). The image treatment follows the user's requested full-width translucent banner.

## Product naming

The current public product name is **GrainMaster** (grain = cereal kernel).
Mozume is the intended beneficiary named by the creator, not a feature name or
part of the product title. About contains **Developed for Mozume**; the sidebar,
header, browser title, install metadata and current downloadable wordmark use
GrainMaster. The English descriptor states the platform's purpose:
"A platform for automated measurement of seed size, shape, and color."
The repository directory and earlier asset variants keep their existing names.

## Welcome panel

About opens automatically on the first visit in each browser profile. Dismissal
is remembered locally under `grainmaster.about.seen.v1`; subsequent visits open
the workbench directly. About remains available from the header. Close, Escape,
and Open workbench dismiss the panel. No brand asset downloads are shown.
