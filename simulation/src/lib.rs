use lib_genetic_algo::*;
use lib_genetic_algo::{Chromosome, Individual, RouletteWheel, UniformCrossOver};
use lib_neuralnetwork::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
mod brain;

use brain::*;
use na::{Point2, Rotation2};
use nalgebra as na;
use rand::{Rng, RngCore};

const VISION_RANGE: f32 = 0.25;
const VISION_ANGLE: f32 = PI + FRAC_PI_4;
const CELLS: usize = 18;
const SPEED_MIN: f32 = 0.001;
const SPEED_MAX: f32 = 0.005;
const SPEED_ACCEL: f32 = 0.2;
const ROTATION_ACCEL: f32 = FRAC_PI_2;

const GENERATION_LENGTH: usize = 2000;
const NUM_OBSTACLES: usize = 400;

pub struct Simulation {
    pub world: World,
    pub ga: GeneticAlgorithm<RouletteWheel>,
    pub age: usize,
}

#[derive(Debug, Clone)]
pub struct World {
    pub animals: Vec<Animal>,
    pub food: Vec<Food>,
    pub obstacles: Vec<Obstacle>,
}

#[derive(Debug, Clone)]
pub struct Animal {
    pub pos: Point2<f32>,
    pub rotation: Rotation2<f32>,
    pub speed: f32,
    pub vision: Vision,
    pub brain: Brain,
    pub(crate) score: usize,
    pub vision_input: Vec<f32>,
}
#[derive(Debug, Clone)]
pub struct Obstacle {
    pub pos: Point2<f32>,
}

pub struct AnimalIndividual {
    pub fitness: f32,
    pub chromosome: Chromosome,
}

impl AnimalIndividual {
    pub fn from_animal(animal: &Animal) -> Self {
        Self {
            fitness: animal.score as f32,
            chromosome: animal.as_chromosome(),
        }
    }
    pub fn to_animal(self, rng: &mut dyn RngCore) -> Animal {
        Animal::from_chromosome(self.chromosome, rng)
    }
}

impl Individual for AnimalIndividual {
    fn create(chromosome: Chromosome) -> Self {
        Self {
            fitness: 0.0,
            chromosome,
        }
    }

    fn chromosome(&self) -> &Chromosome {
        &self.chromosome
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }
}

#[derive(Debug, Clone)]
pub struct Food {
    pos: Point2<f32>,
}
#[derive(Debug, Clone)]
pub struct Vision {
    pub range: f32,
    pub angle: f32,
    pub cells: usize,
}

impl Vision {
    pub fn new(range: f32, angle: f32, cells: usize) -> Self {
        Self {
            range,
            angle,
            cells,
        }
    }
    pub fn cells(&self) -> usize {
        self.cells
    }
    pub fn process_vision(
        &self,
        position: Point2<f32>,
        rotation: Rotation2<f32>,
        food: &[Food],
        obstacles: &[Obstacle],
    ) -> Vec<f32> {
        let mut cells = vec![0.0; self.cells];

        for food in food {
            let vector = food.pos - position;
            let distance = vector.norm();

            if distance > self.range {
                continue;
            }
            let angle =
                nalgebra::Rotation2::rotation_between(&nalgebra::Vector2::x(), &vector).angle();
            let angle = angle - rotation.angle();
            let angle = nalgebra::wrap(angle, -PI, PI);

            if angle < -self.angle / 2.0 || angle > self.angle / 2.0 {
                continue;
            }
            let angle = angle + self.angle / 2.0;
            let cell = angle / self.angle * (self.cells as f32);
            let cell = (cell as usize).min(cells.len() - 1);

            cells[cell] += (self.range - distance) / self.range;
        }
        for obstacle in obstacles {
            // let vector = obstacle.pos - position;
            // let distance = vector.norm();

            // if distance > self.range {
            //     continue;
            // }
            // let angle =
            //     nalgebra::Rotation2::rotation_between(&nalgebra::Vector2::x(), &vector).angle();
            // let angle = angle - rotation.angle();
            // let angle = nalgebra::wrap(angle, -PI, PI);

            // if angle < -self.angle / 2.0 || angle > self.angle / 2.0 {
            //     continue;
            // }
            // let angle = angle + self.angle / 2.0;
            // let cell = angle / self.angle * (self.cells as f32) / 2.0 + (self.cells as f32) / 2.0;
            // let cell = (cell as usize).min(cells.len() - 1);

            // cells[cell] += (self.range - distance) / self.range;
        }
        cells
    }
}

impl Default for Vision {
    fn default() -> Self {
        Self {
            range: VISION_RANGE,
            angle: VISION_ANGLE,
            cells: CELLS,
        }
    }
}

impl Simulation {
    pub fn randomize(rng: &mut dyn RngCore) -> Self {
        let world = World::randomize(rng);
        let ga = GeneticAlgorithm::new(
            RouletteWheel::default(),
            UniformCrossOver::default(),
            GaussianMutation::new(0.01, 0.3),
        );
        let age = 0;
        Self { world, ga, age }
    }
    pub fn world(&self) -> &World {
        &self.world
    }
    pub fn step_forward(&mut self, rng: &mut dyn RngCore) -> Option<Statistics> {
        self.process_motion();
        self.process_collision(rng);
        self.process_brain();

        self.age += 1;

        if self.age > GENERATION_LENGTH {
            Some(self.evolve(rng))
        } else {
            None
        }
    }
    pub fn process_motion(&mut self) {
        for animal in &mut self.world.animals {
            animal.pos += animal.rotation * nalgebra::Vector2::new(animal.speed, 0.0);

            animal.pos.x = na::wrap(animal.pos.x, 0.0, 1.0);
            animal.pos.y = na::wrap(animal.pos.y, 0.0, 1.0);
        }
    }
    pub fn process_collision(&mut self, rng: &mut dyn RngCore) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.food {
                let dist = nalgebra::distance(&animal.pos, &food.pos);
                if dist <= 0.009 {
                    food.pos = rng.gen();
                    animal.score += 1;
                }
            }
            for obstacle in &mut self.world.obstacles {
                let dist = nalgebra::distance(&animal.pos, &obstacle.pos);
                if dist <= 0.009 {}
            }
        }
    }
    pub fn process_brain(&mut self) {
        for animal in &mut self.world.animals {
            let vision = animal.vision.process_vision(
                animal.pos,
                animal.rotation,
                &self.world.food,
                &self.world.obstacles,
            );
            let brain_response = animal.brain.nn.propagate(vision.clone());
            let speed = brain_response[0].clamp(-SPEED_ACCEL, SPEED_ACCEL);
            let rotation = brain_response[1].clamp(-ROTATION_ACCEL, ROTATION_ACCEL);

            animal.speed = (animal.speed + speed).clamp(SPEED_MIN, SPEED_MAX);
            animal.rotation = na::Rotation2::new(animal.rotation.angle() + rotation);
            animal.vision_input = vision;
        }
    }
    pub fn evolve(&mut self, rng: &mut dyn RngCore) -> Statistics {
        self.age = 0;

        let current_population: Vec<_> = self
            .world
            .animals()
            .iter()
            .map(AnimalIndividual::from_animal)
            .collect();

        // let mut animal_individuals = Vec::new();
        // for animal in self.world.animals.iter() {
        //     let animal_individual = AnimalIndividual::from_animal(animal);

        //     animal_individuals.push(animal_individual);
        // }

        let (evolved_population, statistics) = self.ga.evolve(rng, &current_population);
        self.world.animals = evolved_population
            .into_iter()
            .map(|individual| individual.to_animal(rng))
            .collect();

        for food in &mut self.world.food {
            food.pos = rng.gen();
        }
        statistics
    }
}

impl World {
    pub fn randomize(rng: &mut dyn RngCore) -> Self {
        let animals = (0..50).map(|_| Animal::randomize(rng)).collect();
        let food = (0..30).map(|_| Food::randomize(rng)).collect();
        let mut obstacles = Vec::with_capacity(NUM_OBSTACLES / 4);

        for i in 0..99 {
            let point = Point2::new((i as f32) / (NUM_OBSTACLES as f32 / 4.0), 0.0);
            obstacles.push(Obstacle { pos: point });
        }
        for i in 0..99 {
            let point = Point2::new((i as f32) / (NUM_OBSTACLES as f32 / 4.0), 1.0);
            obstacles.push(Obstacle { pos: point });
        }
        for i in 0..99 {
            let point = Point2::new(0.0, (i as f32) / (NUM_OBSTACLES as f32 / 4.0));
            obstacles.push(Obstacle { pos: point });
        }
        for i in 0..99 {
            let point = Point2::new(1.0, (i as f32) / (NUM_OBSTACLES as f32 / 4.0));
            obstacles.push(Obstacle { pos: point });
        }

        Self {
            animals,
            food,
            obstacles,
        }
    }
    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn food(&self) -> &[Food] {
        &self.food
    }
    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }
}

impl Animal {
    pub fn randomize(rng: &mut dyn RngCore) -> Self {
        let vision = Vision::default();
        let brain = Brain::randomize(&vision);
        Self {
            pos: rng.gen(),
            rotation: rng.gen(),
            speed: 0.0005,
            vision,
            brain,
            score: 0,
            vision_input: vec![0.0; CELLS],
        }
    }
    pub fn position(&self) -> &Point2<f32> {
        &self.pos
    }
    pub fn rotation(&self) -> &Rotation2<f32> {
        &self.rotation
    }
    pub fn speed(&self) -> f32 {
        self.speed
    }
    pub fn as_chromosome(&self) -> Chromosome {
        self.brain.as_chromosome()
    }
    pub fn from_chromosome(chromosome: Chromosome, rng: &mut dyn RngCore) -> Self {
        let vision = Vision::default();
        let brain = Brain::from_chromosome(chromosome, &vision);
        Self {
            pos: rng.gen(),
            rotation: rng.gen(),
            speed: 0.001,
            vision,
            brain,
            score: 0,
            vision_input: Vec::with_capacity(CELLS),
        }
    }
}

impl Food {
    pub fn randomize(rng: &mut dyn RngCore) -> Self {
        Self { pos: rng.gen() }
    }
    pub fn position(&self) -> &Point2<f32> {
        &self.pos
    }
}
impl Obstacle {
    pub fn position(&self) -> &Point2<f32> {
        &self.pos
    }
}
