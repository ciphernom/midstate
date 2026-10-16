//! Bayesian Finality Estimation using a Beta-Binomial Model.

/// Sentinel returned by [`FinalityEstimator::calculate_safe_depth`] when the
/// requested risk cannot be met at *any* depth within the search range.
///
/// This happens whenever the posterior mass on "attacker holds >= 50% of the
/// hashpower" (`p <= 0.5`, where the per-block catch-up probability is `1.0`
/// and therefore depth-independent) already exceeds `target_risk`: no number of
/// confirmations can drive the expected reorg probability below the target.
///
/// It is deliberately `u64::MAX` so that callers which do not special-case it
/// are still fail-safe: `confirmations >= SAFE_DEPTH_UNATTAINABLE` is always
/// false, so nothing is ever declared final, and depth-subtraction
/// (`height.saturating_sub(safe_depth)`) saturates to 0 (prune nothing). The
/// previous behaviour returned the finite 10,000 cap here, which looked like a
/// real "safe depth" and could cause a transaction to be shown as final at a
/// depth whose true reorg risk was far above the requested threshold.
pub const SAFE_DEPTH_UNATTAINABLE: u64 = u64::MAX;

/// Dynamically estimates the safe re-organization depth using a Beta-Binomial model.
///
/// The probability of a successful deep reorg of length `z` by an attacker with
/// hashpower proportion `q` (and honest proportion `p`) is `(q/p)^z`. We model our 
/// uncertainty about `p` using a Beta distribution updated by network observations.
#[derive(Clone, Debug)]
pub struct FinalityEstimator {
    pub alpha: u64,
    pub beta: u64,
}

impl FinalityEstimator {
    /// Creates a new estimator with a prior belief. 
    ///
    /// # Examples
    /// ```
    /// use midstate::core::finality::FinalityEstimator;
    /// let estimator = FinalityEstimator::new(10, 2);
    /// assert_eq!(estimator.alpha, 10);
    /// ```
    pub fn new(alpha: u64, beta: u64) -> Self {
        Self { alpha, beta }
    }

    /// Records an honest block observation (e.g., linear chain extension).
    /// Caps at 10,000 to maintain agility and prevent overflow.
    ///
    /// # Examples
    /// ```
    /// use midstate::core::finality::FinalityEstimator;
    /// let mut est = FinalityEstimator::new(10, 2);
    /// est.observe_honest();
    /// assert_eq!(est.alpha, 11);
    /// ```
    pub fn observe_honest(&mut self) {
        self.alpha = self.alpha.saturating_add(1).min(10_000);
    }

    /// Records an adversarial observation (e.g., deep fork or orphaned block).
    ///
    /// # Examples
    /// ```
    /// use midstate::core::finality::FinalityEstimator;
    /// let mut est = FinalityEstimator::new(10, 2);
    /// est.observe_adversarial();
    /// assert_eq!(est.beta, 3);
    /// ```
    pub fn observe_adversarial(&mut self) {
        self.beta = self.beta.saturating_add(1).min(10_000);
    }

    /// Calculates the required block depth `z` to ensure the expected probability
    /// of a successful reorg is below `target_risk`.
    ///
    /// Returns [`SAFE_DEPTH_UNATTAINABLE`] if no depth in the search range meets
    /// `target_risk` (e.g. the posterior already places more than `target_risk`
    /// mass on an attacker majority). Callers MUST treat that value as "finality
    /// cannot be established at this confidence", not as a literal depth.
    ///
    /// # Examples
    /// ```
    /// use midstate::core::finality::FinalityEstimator;
    /// let mut est = FinalityEstimator::new(10, 2);
    /// 
    /// // Initially requires a high depth for 1 in a million risk
    /// let safe_initial = est.calculate_safe_depth(1e-6);
    /// 
    /// // After observing consistent honest network behavior, required depth drops
    /// for _ in 0..100 { est.observe_honest(); }
    /// let safe_later = est.calculate_safe_depth(1e-6);
    /// 
    /// assert!(safe_later < safe_initial);
    /// ```
    pub fn calculate_safe_depth(&self, target_risk: f64) -> u64 {
        const STEPS: usize = 1000;
        let mut log_weights = vec![0.0; STEPS];
        let mut max_log_w = f64::NEG_INFINITY;
        
        // 1. Calculate unnormalized log weights of the Beta PDF
        for i in 1..STEPS {
            let p = i as f64 / STEPS as f64;
            let log_w = (self.alpha as f64 - 1.0) * p.ln() + (self.beta as f64 - 1.0) * (1.0 - p).ln();
            log_weights[i] = log_w;
            if log_w > max_log_w {
                max_log_w = log_w;
            }
        }

        // 2. Normalize using Log-Sum-Exp to prevent float underflow
        let mut weights = vec![0.0; STEPS];
        let mut total_weight = 0.0;
        for i in 1..STEPS {
            let w = (log_weights[i] - max_log_w).exp();
            weights[i] = w;
            total_weight += w;
        }

        // 3. Binary search to find the lowest `z` satisfying the target risk in O(log Z)
        let mut low = 1;
        let mut high = 10_000;
        let mut ans = 10_000;
        // Whether ANY depth in [low, high] actually met the target. Without this
        // the loop returns its initial `ans` (10,000) even when nothing passed,
        // reporting a "safe depth" that is not safe. See SAFE_DEPTH_UNATTAINABLE.
        let mut found = false;

        while low <= high {
            let mid = low + (high - low) / 2;
            let mut expected_risk = 0.0;

            for i in 1..STEPS {
                let p = i as f64 / STEPS as f64;
                let w = weights[i] / total_weight;

                let catchup_prob = if p > 0.5 {
                    ((1.0 - p) / p).powi(mid as i32)
                } else {
                    1.0
                };

                expected_risk += catchup_prob * w;
            }

            if expected_risk <= target_risk {
                ans = mid;
                found = true;
                high = mid - 1; // Try to find a tighter/smaller safe depth
            } else {
                low = mid + 1;  // Not safe enough, increase depth
            }
        }

        if found { ans } else { SAFE_DEPTH_UNATTAINABLE }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let est = FinalityEstimator::new(15, 5);
        assert_eq!(est.alpha, 15);
        assert_eq!(est.beta, 5);
    }

    #[test]
    fn test_observe_caps() {
        let mut est = FinalityEstimator::new(9999, 9999);
        
        // Should cap at 10,000
        est.observe_honest();
        est.observe_honest();
        assert_eq!(est.alpha, 10_000);

        est.observe_adversarial();
        est.observe_adversarial();
        assert_eq!(est.beta, 10_000);
    }

    #[test]
    fn test_safe_depth_decreases_with_honesty() {
        let mut est = FinalityEstimator::new(10, 2);
        let initial_depth = est.calculate_safe_depth(1e-6);

        // Accumulate a track record of honest network behavior
        for _ in 0..100 {
            est.observe_honest();
        }
        
        let later_depth = est.calculate_safe_depth(1e-6);
        // Greater confidence means we require fewer blocks for the same risk tolerance
        assert!(later_depth < initial_depth);
    }

    #[test]
    fn test_safe_depth_increases_with_adversity() {
        let mut est = FinalityEstimator::new(100, 2);
        let initial_depth = est.calculate_safe_depth(1e-6);

        // Network starts experiencing forks/orphans
        for _ in 0..10 {
            est.observe_adversarial();
        }
        
        let later_depth = est.calculate_safe_depth(1e-6);
        // Lower confidence means we must wait longer to achieve the same risk tolerance
        assert!(later_depth > initial_depth);
    }

    #[test]
    fn test_extreme_adversarial_is_unattainable() {
        // If the prior strongly suggests the attacker has the majority of the
        // hash power (beta > alpha), the catch-up probability on the p <= 0.5
        // mass is 1.0 and depth-independent, so the target risk cannot be met at
        // ANY depth. The estimator must say so explicitly rather than returning
        // the 10,000 cap as if it were a real safe depth.
        let est = FinalityEstimator::new(2, 50);
        let depth = est.calculate_safe_depth(1e-6);
        assert_eq!(depth, SAFE_DEPTH_UNATTAINABLE);
    }

    #[test]
    fn test_attainable_returns_finite_depth() {
        // An honest-majority posterior must still yield a real, finite depth
        // (regression guard: the unattainable sentinel must not leak into the
        // normal case).
        let est = FinalityEstimator::new(200, 2);
        let depth = est.calculate_safe_depth(1e-6);
        assert_ne!(depth, SAFE_DEPTH_UNATTAINABLE);
        assert!(depth >= 1 && depth <= 10_000);
    }
}
