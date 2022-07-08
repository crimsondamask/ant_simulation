use crate::chromosome::Chromosome;
use rand::RngCore;

mod uniform_crossover;

pub use uniform_crossover::UniformCrossOver;
pub trait Crossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_one: &Chromosome,
        parent_two: &Chromosome,
    ) -> Chromosome;
}
