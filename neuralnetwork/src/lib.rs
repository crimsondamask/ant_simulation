use rand::Rng;

#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug, Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug, Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Debug, Clone)]
pub struct NetworkTopology {
    pub neurons: usize,
}

impl Network {
    pub fn randomize(layers: &[NetworkTopology]) -> Self {
        assert!(layers.len() > 1);
        let mut created_layers = Vec::new();

        for i in 0..(layers.len() - 1) {
            let input_neurons = layers[i].neurons;
            let output_neurons = layers[i + 1].neurons;

            created_layers.push(Layer::randomize(input_neurons, output_neurons));
        }

        Self {
            layers: created_layers,
        }
    }

    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propagate(inputs);
        }

        inputs
    }
    pub fn weights(&self) -> Vec<f32> {
        let mut weights = Vec::new();

        for layer in &self.layers {
            for neuron in &layer.neurons {
                weights.push(neuron.bias);

                for weight in &neuron.weights {
                    weights.push(*weight);
                }
            }
        }
        weights
    }
    pub fn from_weights(
        layers: &[NetworkTopology],
        weights: impl IntoIterator<Item = f32>,
    ) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| Layer::from_weights(layers[0].neurons, layers[1].neurons, &mut weights))
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }
}

impl Layer {
    pub fn randomize(input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::randomize(input_neurons))
            .collect();

        Self { neurons }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
    pub fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }
}

impl Neuron {
    pub fn randomize(output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let bias = rng.gen_range(-1.0..=1.0);

        let weights: Vec<f32> = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { weights, bias }
    }

    pub fn propagate(&self, inputs: &[f32]) -> f32 {
        // assert!(inputs.len() == self.weights.len());
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
    pub fn from_weights(output_neurons: usize, weights: &mut dyn Iterator<Item = f32>) -> Self {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..output_neurons)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self { bias, weights }
    }
}

//-------------------------------------------
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
