pub struct Chromosome {
    genome: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genome.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genome.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genome.iter_mut()
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genome: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = impl Iterator<Item = f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genome.into_iter()
    }
}
