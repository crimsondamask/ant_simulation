use crate::*;

#[derive(Debug, Clone)]
pub struct Brain {
    pub nn: lib_neuralnetwork::Network,
}

impl Brain {
    pub fn randomize(vision: &Vision) -> Self {
        Self {
            nn: lib_neuralnetwork::Network::randomize(&Self::topology(vision)),
        }
    }
    pub fn as_chromosome(&self) -> lib_genetic_algo::Chromosome {
        self.nn.weights().into_iter().collect()
    }
    pub fn topology(vision: &Vision) -> [lib_neuralnetwork::NetworkTopology; 3] {
        [
            lib_neuralnetwork::NetworkTopology {
                neurons: vision.cells(),
            },
            lib_neuralnetwork::NetworkTopology {
                neurons: vision.cells() * 2,
            },
            lib_neuralnetwork::NetworkTopology { neurons: 2 },
        ]
    }
    pub fn from_chromosome(chromosome: Chromosome, vision: &Vision) -> Self {
        Self {
            nn: Network::from_weights(&Self::topology(vision), chromosome),
        }
    }
}
