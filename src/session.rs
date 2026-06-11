use crate::app::Entry;
use crate::providers::{Message, ModelEntry};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// A persisted conversation. Saved after every completed turn, so a crash
/// loses at most the in-flight exchange.
#[derive(Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    pub model: Option<ModelEntry>,
    pub history: Vec<Message>,
    pub transcript: Vec<Entry>,
}

pub struct Meta {
    pub path: PathBuf,
    pub title: String,
    pub updated_at: u64,
}

pub fn new_id() -> String {
    format!("{}", now_secs() * 1000 + std::process::id() as u64 % 1000)
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn dir() -> Result<PathBuf> {
    let base = match std::env::var_os("SHALTAIBOLTAI_DATA_DIR") {
        Some(p) => PathBuf::from(p),
        None => dirs::data_dir()
            .context("no data directory on this platform")?
            .join("shaltaiboltai"),
    };
    let d = base.join("sessions");
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

pub fn save(session: &Session) -> Result<()> {
    let path = dir()?.join(format!("{}.json", session.id));
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
    let Ok(dir) = dir() else { return Vec::new() };
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
