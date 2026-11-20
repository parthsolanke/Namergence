use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use namergence::loader;
use namergence::model::SavedModel;
use namergence::training::{self, HyperParams};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

#[derive(Deserialize)]
struct InputName {
    name: String,
}

#[derive(Serialize)]
struct Prediction {
    name: String,
    label: String,
    confidences: Vec<f64>,
    categories: Vec<String>,
}

struct AppState {
    model: SavedModel,
}

fn model_path() -> PathBuf {
    std::env::var("NAMERGENCE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("model/rnn.json"))
}

fn load_or_train(path: &Path) -> SavedModel {
    if path.is_file() {
        match SavedModel::load(path) {
            Ok(model) => return model,
            Err(error) => eprintln!("warning: could not load {}: {}", path.display(), error),
        }
    }

    eprintln!("no saved model at {}, training a new one", path.display());
    let (names, categories) = loader::load_data().expect("failed to load names data");
    let hyper = HyperParams::default();
    let rnn = training::train(&names, &categories, &hyper);
    let model = SavedModel { rnn, categories };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).ok();
        }
    }
    if let Err(error) = model.save(path) {
        eprintln!("warning: could not save model to {}: {}", path.display(), error);
    }

    model
}

async fn predict(
    State(state): State<Arc<AppState>>,
    Json(input): Json<InputName>,
) -> Json<Prediction> {
    let probs = training::predict_probs(&state.model.rnn, &input.name);
    let index = training::argmax(&probs);

    Json(Prediction {
        name: input.name,
        label: state.model.categories[index].clone(),
        confidences: probs,
        categories: state.model.categories.clone(),
    })
}

#[tokio::main]
async fn main() {
    let path = model_path();
    let model = load_or_train(&path);
    let state = Arc::new(AppState { model });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/predict", post(predict))
        .layer(cors)
        .with_state(state);

    let addr = std::env::var("NAMERGENCE_ADDR").unwrap_or_else(|_| "127.0.0.1:8000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|error| panic!("failed to bind {}: {}", addr, error));
    println!("listening on http://{}", addr);

    axum::serve(listener, app).await.expect("server error");
}
