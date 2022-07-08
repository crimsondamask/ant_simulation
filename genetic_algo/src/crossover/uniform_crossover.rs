use rand::Rng;

use super::Crossover;
use crate::chromosome::*;

#[derive(Clone, Debug, Default)]
pub struct UniformCrossOver;

impl Crossover for UniformCrossOver {
    fn crossover(
        &self,
        rng: &mut dyn rand::RngCore,
        parent_one: &Chromosome,
        parent_two: &Chromosome,
    ) -> Chromosome {
        assert!(parent_one.len() == parent_two.len());

        parent_one
            .iter()
            .zip(parent_two.iter())
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}
