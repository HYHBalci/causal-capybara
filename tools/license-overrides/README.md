# Release license overrides

Some published crate archives omit their monorepo license and copyright files.
`manifest.json` maps each reviewed **crate name and version** to files recovered
from its upstream repository at the revision recorded in `.cargo_vcs_info.json`.
Each vendored file has a SHA-256 digest and a source URL with an immutable commit.
The collector checks the declared license, crate revision, and file digest before
including these notices in the installer. A new missing notice fails the build
until it is reviewed.

The two old `winapi-*-pc-windows-gnu` archives do not contain VCS metadata. Their
notice source is the pinned winapi-rs 0.3.6 release, whose GNU subcrate manifests
both specify version 0.4.0. This exception is recorded in the manifest.

Source-header notice excerpts supplement UNIC's monorepo copyright file. The
excerpts preserve the years and holders in the exact published crate. Full crate
source archive links are included in the generated inventory, including for MPL
components.

The manifest also records reviewed cases where upstream supplied no separate
license text or copyright notice (`r-efi` and `sigchld`), an explanation without
full terms (the objc2 family), or MPL source headers without a separate license
file (`selectors`). These entries retain the applicable canonical terms and
source provenance. Other-platform and build dependencies are included as an
over-inclusive inventory; their inclusion does not imply they ship as runtime
code in the Windows app.

To update an override, inspect the new archive and its pinned upstream source,
copy the actual notices without editing them, update its version/revision and
SHA-256 entries, and run `node tools/collect_release_licenses.mjs` after staging
Python, installing npm dependencies, and resolving Cargo.lock. Do not replace a
known upstream copyright notice with a generic license template.
