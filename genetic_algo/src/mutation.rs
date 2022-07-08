mod gaussian;
use rand::RngCore;

use crate::chromosome::Chromosome;
pub use gaussian::GaussianMutation;

pub trait Mutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}
