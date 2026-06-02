## Summary

<!-- What does this change do, and why? -->

## Checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` is clean
- [ ] `cargo test`, `cargo test --features experimental`, and
      `cargo test --no-default-features` pass
- [ ] If a SIMD kernel changed: the AVX-512 path is covered (CI runs the
      suite under Intel SDE; locally, run on an AVX-512 host or via SDE)
- [ ] No new system/numerical dependency (no BLAS / faer / ndarray / statrs)
- [ ] MSRV (1.89) still builds — CI enforces this
- [ ] Compatibility impact classified under `docs/compatibility-policy.md`
      if this touches public Rust, Python, C ABI, Go, persisted-format,
      feature, or MSRV surfaces
- [ ] `CHANGELOG.md` updated under `Unreleased` if user-facing
- [ ] `cargo deny check` passes (licenses / advisories / bans / sources)
- [ ] If `ordvec-python/` changed: `cargo clippy -p ordvec-python --all-targets -- -D warnings`, then `maturin develop` + `pytest ordvec-python/tests` pass

## Notes

<!-- Anything reviewers should know: trade-offs, follow-ups, benchmark deltas. -->
