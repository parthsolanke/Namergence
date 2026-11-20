use std::path::PathBuf;

use namergence::loader;
use namergence::model::SavedModel;
use namergence::training::{self, HyperParams};

fn model_path() -> PathBuf {
    std::env::var("NAMERGENCE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("model/rnn.json"))
}

fn main() {
    let (names, categories) = loader::load_data().expect("failed to load names data");
    let total_names: usize = names.values().map(|lines| lines.len()).sum();
    println!(
        "loaded {} names across {} categories",
        total_names,
        categories.len()
    );

    let hyper = HyperParams::default();
    println!("training for {} iterations", hyper.n_iters);
    let rnn = training::train(&names, &categories, &hyper);

    let accuracy = training::accuracy(&rnn, &names, &categories);
    println!("accuracy: {:.4}", accuracy);

    let model = SavedModel { rnn, categories };
    let path = model_path();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    model.save(&path).expect("failed to save model");
    println!("saved model to {}", path.display());
}
