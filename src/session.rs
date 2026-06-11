use crate::app::Entry;
use crate::providers::{Message, ModelEntry};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const INPUT_HISTORY_FILE: &str = "input_history.jsonl";
const INPUT_HISTORY_MAX: usize = 500;

/// A persisted conversation. Saved after every completed turn, so a crash
/// loses at most the in-flight exchange.
#[derive(Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    /// Working directory the session belongs to. `None` on sessions saved by
    /// older versions; treated as belonging everywhere.
    #[serde(default)]
    pub cwd: Option<String>,
    pub model: Option<ModelEntry>,
    pub history: Vec<Message>,
    pub transcript: Vec<Entry>,
}

pub struct Meta {
    pub path: PathBuf,
    pub title: String,
    pub updated_at: u64,
    pub cwd: Option<String>,
}

/// Unique even when called repeatedly within the same millisecond or from
/// concurrent instances: millis + pid + per-process counter.
pub fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        now_millis(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn data_root() -> Result<PathBuf> {
    let base = match std::env::var_os("SHALTAIBOLTAI_DATA_DIR") {
        Some(p) => PathBuf::from(p),
        None => dirs::data_dir()
            .context("no data directory on this platform")?
            .join("shaltaiboltai"),
    };
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

fn sessions_dir() -> Result<PathBuf> {
    let d = data_root()?.join("sessions");
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

pub fn save(session: &Session) -> Result<()> {
    let path = sessions_dir()?.join(format!("{}.json", session.id));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(session)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Session> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// All saved sessions, newest first. Unreadable files are skipped.
pub fn list() -> Vec<Meta> {
    let Ok(dir) = sessions_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut metas: Vec<Meta> = entries
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                return None;
            }
            let s = load(&path).ok()?;
            Some(Meta {
                path,
                title: s.title,
                updated_at: s.updated_at,
                cwd: s.cwd,
            })
        })
        .collect();
    metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    metas
}

/// Compact "how long ago" label for the session picker.
pub fn ago(updated_at: u64) -> String {
    let delta = now_secs().saturating_sub(updated_at);
    match delta {
        0..=59 => "just now".into(),
        60..=3599 => format!("{}m ago", delta / 60),
        3600..=86_399 => format!("{}h ago", delta / 3600),
        _ => format!("{}d ago", delta / 86_400),
    }
}

// ---- persisted UI state ----

const THEME_FILE: &str = "theme";

/// Theme chosen at runtime via /theme; takes precedence over config.toml.
pub fn load_theme_name() -> Option<String> {
    let root = data_root().ok()?;
    let name = std::fs::read_to_string(root.join(THEME_FILE)).ok()?;
    let name = name.trim().to_owned();
    (!name.is_empty()).then_some(name)
}

pub fn save_theme_name(name: &str) {
    if let Ok(root) = data_root() {
        let _ = std::fs::write(root.join(THEME_FILE), name);
    }
}

// ---- prompt input history (shell-style Up-arrow recall) ----

/// Stored as JSON-encoded strings, one per line, so multi-line inputs
/// round-trip safely.
pub fn load_input_history() -> Vec<String> {
    let Ok(root) = data_root() else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(root.join(INPUT_HISTORY_FILE)) else {
        return Vec::new();
    };
    let entries: Vec<String> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<String>(l).ok())
        .collect();
    let skip = entries.len().saturating_sub(INPUT_HISTORY_MAX);
    entries.into_iter().skip(skip).collect()
}

pub fn append_input_history(entry: &str) {
    let Ok(root) = data_root() else { return };
    let Ok(line) = serde_json::to_string(entry) else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(INPUT_HISTORY_FILE))
    {
        let _ = writeln!(f, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_in_rapid_succession() {
        let ids: Vec<String> = (0..100).map(|_| new_id()).collect();
        let mut deduped = ids.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(deduped.len(), ids.len());
    }
}
