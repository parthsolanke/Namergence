use std::fs;
use std::io;
use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single-layer recurrent network mirroring the Python `RNN` module:
/// the hidden state is a linear projection of the concatenated input and
/// previous hidden state, and the output is a log-softmax over that projection.
#[derive(Serialize, Deserialize, Clone)]
pub struct Rnn {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub w_ih: Vec<f64>,
    pub b_ih: Vec<f64>,
    pub w_io: Vec<f64>,
    pub b_io: Vec<f64>,
}

impl Rnn {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let combined = input_size + hidden_size;
        let mut rng = rand::thread_rng();
        let bound = 1.0 / (combined as f64).sqrt();

        let mut w_ih = vec![0.0; hidden_size * combined];
        let mut b_ih = vec![0.0; hidden_size];
        let mut w_io = vec![0.0; output_size * combined];
        let mut b_io = vec![0.0; output_size];

        for value in w_ih.iter_mut() {
            *value = rng.gen_range(-bound..bound);
        }
        for value in b_ih.iter_mut() {
            *value = rng.gen_range(-bound..bound);
        }
        for value in w_io.iter_mut() {
            *value = rng.gen_range(-bound..bound);
        }
        for value in b_io.iter_mut() {
            *value = rng.gen_range(-bound..bound);
        }

        Rnn {
            input_size,
            hidden_size,
            output_size,
            w_ih,
            b_ih,
            w_io,
            b_io,
        }
    }

    pub fn init_hidden(&self) -> Vec<f64> {
        vec![0.0; self.hidden_size]
    }

    fn combined_size(&self) -> usize {
        self.input_size + self.hidden_size
    }

    /// Runs one timestep and returns (softmax probabilities, next hidden state).
    pub fn forward(&self, input: &[f64], hidden: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let d = self.combined_size();
        let mut combined = Vec::with_capacity(d);
        combined.extend_from_slice(input);
        combined.extend_from_slice(hidden);

        let mut new_hidden = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let row = &self.w_ih[j * d..(j + 1) * d];
            let mut sum = self.b_ih[j];
            for k in 0..d {
                sum += row[k] * combined[k];
            }
            new_hidden[j] = sum;
        }

        let mut probs = vec![0.0; self.output_size];
        let mut max_logit = f64::NEG_INFINITY;
        for o in 0..self.output_size {
            let row = &self.w_io[o * d..(o + 1) * d];
            let mut sum = self.b_io[o];
            for k in 0..d {
                sum += row[k] * combined[k];
            }
            probs[o] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        let mut denom = 0.0;
        for value in probs.iter_mut() {
            *value = (*value - max_logit).exp();
            denom += *value;
        }
        for value in probs.iter_mut() {
            *value /= denom;
        }

        (probs, new_hidden)
    }

    /// Runs a full sequence, backpropagates through time and applies one SGD
    /// step. Returns the negative log-likelihood loss for the target category.
    pub fn train_sequence(
        &mut self,
        inputs: &[Vec<f64>],
        target: usize,
        learning_rate: f64,
    ) -> f64 {
        let d = self.combined_size();
        let mut hidden = self.init_hidden();
        let mut caches: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(inputs.len());

        for input in inputs {
            let mut combined = Vec::with_capacity(d);
            combined.extend_from_slice(input);
            combined.extend_from_slice(&hidden);

            let mut new_hidden = vec![0.0; self.hidden_size];
            for j in 0..self.hidden_size {
                let row = &self.w_ih[j * d..(j + 1) * d];
                let mut sum = self.b_ih[j];
                for k in 0..d {
                    sum += row[k] * combined[k];
                }
                new_hidden[j] = sum;
            }

            let mut probs = vec![0.0; self.output_size];
            let mut max_logit = f64::NEG_INFINITY;
            for o in 0..self.output_size {
                let row = &self.w_io[o * d..(o + 1) * d];
                let mut sum = self.b_io[o];
                for k in 0..d {
                    sum += row[k] * combined[k];
                }
                probs[o] = sum;
                if sum > max_logit {
                    max_logit = sum;
                }
            }
            let mut denom = 0.0;
            for value in probs.iter_mut() {
                *value = (*value - max_logit).exp();
                denom += *value;
            }
            for value in probs.iter_mut() {
                *value /= denom;
            }

            caches.push((combined, probs));
            hidden = new_hidden;
        }

        if caches.is_empty() {
            return 0.0;
        }

        let last = caches.len() - 1;
        let target_prob = caches[last].1[target].max(1e-12);
        let loss = -target_prob.ln();

        let mut w_ih_grad = vec![0.0; self.w_ih.len()];
        let mut b_ih_grad = vec![0.0; self.b_ih.len()];
        let mut w_io_grad = vec![0.0; self.w_io.len()];
        let mut b_io_grad = vec![0.0; self.b_io.len()];

        let mut dh_next = vec![0.0; self.hidden_size];

        for t in (0..caches.len()).rev() {
            let (combined, probs) = &caches[t];

            let mut dz = vec![0.0; self.output_size];
            if t == last {
                dz.copy_from_slice(probs);
                dz[target] -= 1.0;
            }

            let mut dc = vec![0.0; d];
            for o in 0..self.output_size {
                let dz_o = dz[o];
                if dz_o == 0.0 {
                    continue;
                }
                let row = &self.w_io[o * d..(o + 1) * d];
                for k in 0..d {
                    dc[k] += row[k] * dz_o;
                    w_io_grad[o * d + k] += dz_o * combined[k];
                }
                b_io_grad[o] += dz_o;
            }

            let mut dh = vec![0.0; self.hidden_size];
            for j in 0..self.hidden_size {
                dh[j] = dc[self.input_size + j] + dh_next[j];
            }

            let mut dc_ih = vec![0.0; d];
            for j in 0..self.hidden_size {
                let dh_j = dh[j];
                if dh_j == 0.0 {
                    continue;
                }
                let row = &self.w_ih[j * d..(j + 1) * d];
                for k in 0..d {
                    dc_ih[k] += row[k] * dh_j;
                    w_ih_grad[j * d + k] += dh_j * combined[k];
                }
                b_ih_grad[j] += dh_j;
            }

            for j in 0..self.hidden_size {
                dh_next[j] = dc_ih[self.input_size + j];
            }
        }

        for i in 0..self.w_ih.len() {
            self.w_ih[i] -= learning_rate * w_ih_grad[i];
        }
        for i in 0..self.b_ih.len() {
            self.b_ih[i] -= learning_rate * b_ih_grad[i];
        }
        for i in 0..self.w_io.len() {
            self.w_io[i] -= learning_rate * w_io_grad[i];
        }
        for i in 0..self.b_io.len() {
            self.b_io[i] -= learning_rate * b_io_grad[i];
        }

        loss
    }
}

/// A trained network together with the ordered category labels.
#[derive(Serialize, Deserialize, Clone)]
pub struct SavedModel {
    pub rnn: Rnn,
    pub categories: Vec<String>,
}

impl SavedModel {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
        fs::write(path, json)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|error| io::Error::new(io::ErrorKind::Other, error))
    }
}
