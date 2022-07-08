#![feature(type_alias_impl_trait)]
pub use self::{
    chromosome::*, crossover::*, individual::*, mutation::*, selection::*, statistics::*,
};
mod chromosome;
mod crossover;
mod individual;
mod mutation;
mod selection;
mod statistics;

use rand::RngCore;

pub struct GeneticAlgorithm<S> {
    pub selection_method: S,
    pub crossover_method: Box<dyn Crossover>,
    pub mutation_method: Box<dyn Mutation>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl Crossover + 'static,
        mutation_method: impl Mutation + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }
    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> (Vec<I>, Statistics)
    where
        I: Individual,
    {
        assert!(!population.is_empty());
        let new_population = (0..population.len())
            .map(|_| {
                let parent_one = self.selection_method.select(rng, population).chromosome();
                let parent_two = self.selection_method.select(rng, population).chromosome();

                let mut child = self.crossover_method.crossover(rng, parent_one, parent_two);

                self.mutation_method.mutate(rng, &mut child);

                I::create(child)
            })
            .collect();

        (new_population, Statistics::analyze(population))
    }
}
