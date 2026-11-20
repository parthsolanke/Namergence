use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

pub const ALL_LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'";

pub fn n_letters() -> usize {
    ALL_LETTERS.chars().count()
}

pub fn unicode_to_ascii(s: &str) -> String {
    s.nfd()
        .filter(|c| !is_combining_mark(*c) && ALL_LETTERS.contains(*c))
        .collect()
}

pub fn letter_to_index(letter: char) -> Option<usize> {
    ALL_LETTERS.find(letter)
}

pub fn line_to_tensor(line: &str) -> Vec<Vec<f64>> {
    let n = n_letters();
    let mut tensor = Vec::new();
    for ch in line.chars() {
        if let Some(idx) = letter_to_index(ch) {
            let mut one_hot = vec![0.0; n];
            one_hot[idx] = 1.0;
            tensor.push(one_hot);
        }
    }
    tensor
}

pub fn resolve_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAMERGENCE_DATA_DIR") {
        return PathBuf::from(dir);
    }
    for candidate in ["data/names", "../data/names", "../../data/names"] {
        if Path::new(candidate).is_dir() {
            return PathBuf::from(candidate);
        }
    }
    PathBuf::from("data/names")
}

pub fn load_data() -> io::Result<(BTreeMap<String, Vec<String>>, Vec<String>)> {
    let dir = resolve_data_dir();
    let mut names: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("txt") {
            continue;
        }

        let category = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(stem) => stem.to_string(),
            None => continue,
        };

        let content = fs::read_to_string(&path)?;
        let lines: Vec<String> = content
            .trim()
            .split('\n')
            .map(|line| unicode_to_ascii(line.trim_end_matches('\r')))
            .collect();

        names.insert(category, lines);
    }

    let categories: Vec<String> = names.keys().cloned().collect();
    Ok((names, categories))
}
