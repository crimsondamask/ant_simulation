pub use self::roulette::*;
use crate::individual::*;

mod roulette;

use rand::RngCore;

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}
