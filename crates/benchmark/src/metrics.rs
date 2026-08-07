use std::time::Duration;

#[derive(Debug, Clone, Copy, Default)]
pub struct LatencyStats {
    pub count: usize,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub wall_clock: Duration,
}

impl LatencyStats {
    pub fn from_durations(mut durations: Vec<Duration>, wall_clock: Duration) -> Self {
        if durations.is_empty() {
            return Self {
                wall_clock,
                ..Default::default()
            };
        }
        durations.sort_unstable();
        let count = durations.len();
        let sum: Duration = durations.iter().sum();
        let percentile = |p: f64| durations[(((count - 1) as f64) * p).round() as usize];
        Self {
            count,
            min: durations[0],
            max: durations[count - 1],
            mean: sum / count as u32,
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            wall_clock,
        }
    }

    pub fn throughput(&self) -> f64 {
        if self.wall_clock.is_zero() {
            0.0
        } else {
            self.count as f64 / self.wall_clock.as_secs_f64()
        }
    }
}
