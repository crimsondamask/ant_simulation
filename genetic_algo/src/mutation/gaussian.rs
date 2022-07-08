use rand::{Rng, RngCore};

use crate::chromosome::Chromosome;

use super::Mutation;

pub struct GaussianMutation {
    chance: f32,
    coefficient: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coefficient: f32) -> Self {
        assert!((0.0..=1.0).contains(&chance));
        Self {
            chance,
            coefficient,
        }
    }
}

impl Mutation for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            if rng.gen_bool(self.chance as f64) {
                *gene += sign * self.coefficient * rng.gen::<f32>();
            }
        }
    }
}
