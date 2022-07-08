use rand::prelude::SliceRandom;

use crate::individual::*;

use crate::*;

#[derive(Clone, Debug, Default)]
pub struct RouletteWheel;

impl SelectionMethod for RouletteWheel {
    fn select<'a, I>(&self, rng: &mut dyn rand::RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("Empty population!")
    }
}
