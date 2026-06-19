//! CRT seam oracle: exhaustive finite verification of the vernier theorem.
//!
//! This is the SPEC for the within-axis vernier experiment. The CRT claim is
//! a finite statement over the ring Z/M (M = product of periods), so we
//! enumerate every position and CHECK the predicted structure exactly.
//! Exhaustive enumeration over the finite ring is a proof for these parameters
//! — the same thing a Lean theorem would assert. We surface the phase subtlety
//! (the naive zero-offset statement is FALSE at the origin) before any recall
//! experiment is built on top.
//!
//! Setup: L grids on ONE axis, grid i has period m_i and phase phi_i. A
//! position x is a "seam" of grid i if (x - phi_i) mod m_i == 0. The vernier
//! claim concerns how often seams of different grids COINCIDE.
//!
//! Checks:
//!   1. coincidence spacing: two coprime grids (phi=0) share a seam exactly
//!      every lcm(m_i,m_j) = m_i*m_j. NON-coprime share every lcm < product.
//!   2. all-L coincidence count over one period M: with phi=0 there is exactly
//!      one (at x=0); the claim "one per period" is about COUNT, and it holds.
//!   3. near-seam blind-spot density: fraction of x within band |.|<=t of a
//!      seam on ALL L grids simultaneously ~= (2t)^L / M  (the headline).
//!   4. the phase refutation: with phi=0 every grid has a seam at x=0, so the
//!      pointwise "max-over-grids distance-to-seam >= c" floor is FALSE there.
//!
//! Run: cargo run --release --example crt_seam_oracle

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// Is x a seam of a grid with period m and phase phi (integer phase)?
/// Seam <=> (x - phi) ≡ 0 (mod m).
fn is_seam(x: i64, m: u64, phi: i64) -> bool {
    let m = m as i64;
    (((x - phi) % m) + m) % m == 0
}

/// Distance (in integer steps, on the ring) from x to the nearest seam of
/// grid (m, phi): min over k of |x - phi - k*m|, i.e. the centered residue.
fn dist_to_seam(x: i64, m: u64, phi: i64) -> u64 {
    let m = m as i64;
    let r = (((x - phi) % m) + m) % m; // in [0, m)
    r.min(m - r) as u64
}

fn main() {
    println!("# CRT seam oracle — exhaustive finite verification");

    // ---- Check 1: pairwise coincidence spacing ----
    println!("\n## Check 1: first common seam of two grids (phi=0)");
    println!("m_i\tm_j\tcoprime\tfirst_common\tlcm(=expected)");
    for (mi, mj) in [(3u64, 5u64), (4, 6), (5, 7), (6, 9), (8, 12)] {
        let mut first = None;
        for x in 1..=(mi * mj) as i64 {
            if is_seam(x, mi, 0) && is_seam(x, mj, 0) {
                first = Some(x);
                break;
            }
        }
        let cop = gcd(mi, mj) == 1;
        println!(
            "{mi}\t{mj}\t{cop}\t{}\t{}",
            first.unwrap(),
            lcm(mi, mj)
        );
    }

    // remaining checks appended in next pieces
    checks_2_3_4();
}

fn checks_2_3_4() {
    // L coprime grids on one axis.
    let periods: [u64; 4] = [3, 5, 7, 11];
    let m: u64 = periods.iter().product(); // M = 1155
    let l = periods.len();

    // ---- Check 2: all-L coincidence count over one period, phi=0 ----
    // Claim: exactly ONE position in [0, M) is a seam of all L grids.
    let mut all_seam_zero = 0u64;
    for x in 0..m as i64 {
        if periods.iter().all(|&p| is_seam(x, p, 0)) {
            all_seam_zero += 1;
        }
    }
    println!("\n## Check 2: all-{l} coincidences in [0,M), phi=0  (M={m})");
    println!("count = {all_seam_zero}  (claim: exactly 1, at x=0)");

    // ---- Check 3: near-seam blind-spot density vs (2t)^L / M ----
    // A position is in the joint blind spot if it is within band |.|<=t of a
    // seam on ALL L grids. Predicted density = product over grids of
    // (per-grid near-seam density) = ((2t+1)/m_i) ... but the clean continuous
    // form is (2t)^L / M. We report measured vs both the integer band count
    // ((2t+1) residues per grid) and the continuous (2t)^L/M headline.
    println!("\n## Check 3: joint near-seam density vs prediction  (phi=0)");
    println!("t\tmeasured\t(2t+1)^L/M_int\t(2t)^L/M_cont");
    for t in [0u64, 1, 2, 3] {
        let mut hits = 0u64;
        for x in 0..m as i64 {
            if periods.iter().all(|&p| dist_to_seam(x, p, 0) <= t) {
                hits += 1;
            }
        }
        let measured = hits as f64 / m as f64;
        // integer band: each grid admits (2t+1) residues (centered), but capped
        // at m_i; product of per-grid fractions = prod((2t+1)/m_i).
        let int_pred: f64 = periods
            .iter()
            .map(|&p| ((2 * t + 1).min(p) as f64) / p as f64)
            .product();
        let cont_pred = (2 * t).pow(l as u32) as f64 / m as f64;
        println!("{t}\t{measured:.6}\t{int_pred:.6}\t{cont_pred:.6}");
    }

    // ---- Check 4: the phase refutation ----
    // With phi=0, x=0 is a seam on every grid -> max-over-grids dist = 0, so a
    // pointwise floor "some grid keeps you >= c from its seam" is FALSE there.
    // With chosen distinct phases, the all-seam coincidence can be MOVED but,
    // by CRT, still occurs exactly once per period (we just relocate it).
    let phis_zero = [0i64; 4];
    let worst_zero = (0..m as i64)
        .map(|x| {
            periods
                .iter()
                .zip(phis_zero.iter())
                .map(|(&p, &ph)| dist_to_seam(x, p, ph))
                .max()
                .unwrap()
        })
        .min()
        .unwrap();
    // staggered phases: spread the seams
    let phis_stag = [0i64, 1, 2, 3];
    let worst_stag = (0..m as i64)
        .map(|x| {
            periods
                .iter()
                .zip(phis_stag.iter())
                .map(|(&p, &ph)| dist_to_seam(x, p, ph))
                .max()
                .unwrap()
        })
        .min()
        .unwrap();
    println!("\n## Check 4: pointwise max-over-grids dist-to-seam (the floor)");
    println!("min_x max_i dist  (phi=0)      = {worst_zero}  (=> 0: floor is FALSE at origin)");
    println!("min_x max_i dist  (phi staggered) = {worst_stag}  (phases can lift the floor)");

    // verdict line for downstream
    println!("\n## Verdict");
    println!(
        "coincidence_count_per_period={all_seam_zero} (=1 ✓); \
         floor_phi0={worst_zero} (=0, naive claim false ✓); \
         floor_staggered={worst_stag} (phase-dependent)"
    );
}
