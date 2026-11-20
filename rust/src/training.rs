use std::collections::BTreeMap;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::loader;
use crate::model::Rnn;

pub struct HyperParams {
    pub learning_rate: f64,
    pub hidden_size: usize,
    pub n_iters: usize,
    pub print_every: usize,
}

impl Default for HyperParams {
    fn default() -> Self {
        let n_iters = std::env::var("NAMERGENCE_TRAIN_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(100_000);

        HyperParams {
            learning_rate: 0.001,
            hidden_size: 128,
            n_iters,
            print_every: 5_000,
        }
    }
}

pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

/// Runs inference for a name and returns the category probabilities.
pub fn predict_probs(rnn: &Rnn, name: &str) -> Vec<f64> {
    let sanitized = loader::unicode_to_ascii(name);
    let tensor = loader::line_to_tensor(&sanitized);

    if tensor.is_empty() {
        let n = rnn.output_size.max(1);
        return vec![1.0 / n as f64; n];
    }

    let mut hidden = rnn.init_hidden();
    let mut probs = vec![0.0; rnn.output_size];
    for input in &tensor {
        let (next_probs, next_hidden) = rnn.forward(input, &hidden);
        probs = next_probs;
        hidden = next_hidden;
    }
    probs
}

pub fn train(
    names: &BTreeMap<String, Vec<String>>,
    categories: &[String],
    hyper: &HyperParams,
) -> Rnn {
    let n_letters = loader::n_letters();
    let mut rnn = Rnn::new(n_letters, hyper.hidden_size, categories.len());
    let mut rng = rand::thread_rng();

    for iter in 1..=hyper.n_iters {
        let category_index = rng.gen_range(0..categories.len());
        let category = &categories[category_index];
        let lines = &names[category];
        if lines.is_empty() {
            continue;
        }

        let line = match lines.choose(&mut rng) {
            Some(line) => line,
            None => continue,
        };
        let tensor = loader::line_to_tensor(line);
        if tensor.is_empty() {
            continue;
        }

        let loss = rnn.train_sequence(&tensor, category_index, hyper.learning_rate);

        if hyper.print_every > 0 && iter % hyper.print_every == 0 {
            let probs = predict_probs(&rnn, line);
            let guess = argmax(&probs);
            let mark = if guess == category_index {
                "OK".to_string()
            } else {
                format!("MISS ({})", category)
            };
            println!(
                "{} {:.2}% ({}) loss={:.4} / {} {}",
                iter,
                iter as f64 / hyper.n_iters as f64 * 100.0,
                line,
                loss,
                categories[guess],
                mark
            );
        }
    }

    rnn
}

pub fn accuracy(
    rnn: &Rnn,
    names: &BTreeMap<String, Vec<String>>,
    categories: &[String],
) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;

    for (category_index, category) in categories.iter().enumerate() {
        for name in &names[category] {
            let probs = predict_probs(rnn, name);
            if argmax(&probs) == category_index {
                correct += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}
