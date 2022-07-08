use std::cmp::Ordering;

use crate::individual::Individual;

#[derive(Clone, Debug)]
pub struct Statistics {
    pub min_fitness: f32,
    pub max_fitness: f32,
    pub avg_fitness: f32,
}

impl Statistics {
    pub fn analyze<I>(population: &[I]) -> Self
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let length = population.len();
        let fitness_data = {
            let mut fitnesses: Vec<_> = population
                .iter()
                .map(|individual| individual.fitness())
                .collect();
            fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            fitnesses
        };

        let min_fitness = fitness_data[0];
        let max_fitness = fitness_data[length - 1];
        let avg_fitness = fitness_data.iter().sum::<f32>() / length as f32;

        Self {
            min_fitness,
            max_fitness,
            avg_fitness,
        }
    }
    pub fn min_fitness(&self) -> f32 {
        self.min_fitness
    }
    pub fn max_fitness(&self) -> f32 {
        self.max_fitness
    }
    pub fn avg_fitness(&self) -> f32 {
        self.avg_fitness
    }
}
