use crate::providers::{ToolCall, ToolDef};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_SEARCH_RESULTS: usize = 200;
const MAX_SEARCH_FILE_BYTES: u64 = 1024 * 1024;
const MAX_DIFF_PREVIEW_LINES: usize = 40;

pub fn definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "read_file",
            description: "Read a UTF-8 text file and return its contents.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, absolute or relative to the working directory."}
                },
                "required": ["path"]
            }),
        },
        ToolDef {
            name: "write_file",
            description: "Write content to a file, creating it (and parent directories) if needed, overwriting if it exists. For changes to an existing file prefer edit_file.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Full file content to write."}
                },
                "required": ["path", "content"]
            }),
        },
        ToolDef {
            name: "edit_file",
            description: "Replace an exact string in a file. old_string must match exactly once unless replace_all is true; include surrounding lines to make it unique.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolDef {
            name: "list_directory",
            description: "List the entries of a directory.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to the working directory."}
                }
            }),
        },
        ToolDef {
            name: "grep",
            description: "Search file contents with a regular expression, recursively. Respects .gitignore and skips hidden/binary files. Returns path:line:text matches.",
            schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to the working directory."}
                },
                "required": ["pattern"]
            }),
        },
        ToolDef {
            name: "glob",
            description: "Find files by name with a glob pattern (e.g. **/*.rs). Respects .gitignore.",
            schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern matched against paths relative to the search root."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to the working directory."}
                },
                "required": ["pattern"]
            }),
        },
        ToolDef {
            name: "run_command",
            description: "Run a shell command in the working directory and return stdout/stderr. 60 second timeout.",
            schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["command"]
            }),
        },
    ]
}

/// Mutating tools always require approval. Read-only tools are auto-approved
/// only inside the working directory — reads outside it (dotfiles, keys,
/// other projects) must be confirmed by the user before their contents are
/// sent to a model provider.
pub fn requires_approval(call: &ToolCall) -> bool {
    match call.name.as_str() {
        "write_file" | "edit_file" | "run_command" => true,
        "read_file" | "list_directory" | "grep" | "glob" => {
            let path = call.arguments["path"].as_str().unwrap_or(".");
            !path_within_cwd(path)
        }
        _ => true,
    }
}

fn path_within_cwd(path: &str) -> bool {
    let Ok(cwd) = std::env::current_dir() else {
        return false;
    };
    let p = Path::new(path);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };
    normalize(&abs).starts_with(normalize(&cwd))
}

/// Lexical normalization (`a/b/../c` → `a/c`) — no filesystem access, so it
/// also works for paths that don't exist yet.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(Component::ParentDir);
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// One-line human-readable summary shown in the approval prompt and transcript.
pub fn describe(call: &ToolCall) -> String {
    let arg = |k: &str| call.arguments[k].as_str().unwrap_or("?");
    match call.name.as_str() {
        "read_file" => format!("read_file {}", arg("path")),
        "write_file" => format!(
            "write_file {} ({} bytes)",
            arg("path"),
            call.arguments["content"].as_str().map_or(0, str::len)
        ),
        "edit_file" => format!("edit_file {}", arg("path")),
        "list_directory" => {
            format!(
                "list_directory {}",
                call.arguments["path"].as_str().unwrap_or(".")
            )
        }
        "grep" => format!(
            "grep /{}/ in {}",
            arg("pattern"),
            call.arguments["path"].as_str().unwrap_or(".")
        ),
        "glob" => format!(
            "glob {} in {}",
            arg("pattern"),
            call.arguments["path"].as_str().unwrap_or(".")
        ),
        "run_command" => format!("run_command: {}", arg("command")),
        other => format!("{other} {}", call.arguments),
    }
}

/// Diff preview for the approval dialog: what the file change would do.
/// Tags: '+' insert, '-' delete, ' ' context, '@' hunk header, '!' problem.
pub fn approval_preview(call: &ToolCall) -> Option<Vec<(char, String)>> {
    let path = call.arguments["path"].as_str()?;
    match call.name.as_str() {
        "write_file" => {
            let old = std::fs::read_to_string(path).unwrap_or_default();
            let new = call.arguments["content"].as_str()?;
            Some(diff_lines(&old, new))
        }
        "edit_file" => {
            let old = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return Some(vec![('!', format!("cannot read {path}: {e}"))]),
            };
            match apply_edit(
                &old,
                call.arguments["old_string"].as_str()?,
                call.arguments["new_string"].as_str()?,
                call.arguments["replace_all"].as_bool().unwrap_or(false),
            ) {
                Ok(new) => Some(diff_lines(&old, &new)),
                Err(e) => Some(vec![('!', format!("{e:#}"))]),
            }
        }
        _ => None,
    }
}

fn diff_lines(old: &str, new: &str) -> Vec<(char, String)> {
    let diff = similar::TextDiff::from_lines(old, new);
    let mut out = Vec::new();
    for hunk in diff.unified_diff().context_radius(2).iter_hunks() {
        out.push(('@', hunk.header().to_string()));
        for change in hunk.iter_changes() {
            let tag = match change.tag() {
                similar::ChangeTag::Insert => '+',
                similar::ChangeTag::Delete => '-',
                similar::ChangeTag::Equal => ' ',
            };
            out.push((tag, change.value().trim_end_matches('\n').to_owned()));
            if out.len() >= MAX_DIFF_PREVIEW_LINES {
                out.push(('@', "… diff truncated".into()));
                return out;
            }
        }
    }
    if out.is_empty() {
        out.push((' ', "(no changes)".into()));
    }
    out
}

pub async fn execute(call: &ToolCall) -> (String, bool) {
    match run(call).await {
        Ok(output) => (truncate(output), false),
        Err(e) => (format!("{e:#}"), true),
    }
}

async fn run(call: &ToolCall) -> Result<String> {
    let args = &call.arguments;
    match call.name.as_str() {
        "read_file" => {
            let path = str_arg(args, "path")?;
            tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("failed to read {path}"))
        }
        "write_file" => {
            let path = str_arg(args, "path")?;
            let content = str_arg(args, "content")?;
            if let Some(parent) = Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await?;
                }
            }
            tokio::fs::write(path, content)
                .await
                .with_context(|| format!("failed to write {path}"))?;
            Ok(format!("wrote {} bytes to {path}", content.len()))
        }
        "edit_file" => {
            let path = str_arg(args, "path")?;
            let content = tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("failed to read {path}"))?;
            let updated = apply_edit(
                &content,
                str_arg(args, "old_string")?,
                str_arg(args, "new_string")?,
                args["replace_all"].as_bool().unwrap_or(false),
            )?;
            tokio::fs::write(path, &updated)
                .await
                .with_context(|| format!("failed to write {path}"))?;
            Ok(format!("edited {path}"))
        }
        "list_directory" => {
            let path = args["path"].as_str().unwrap_or(".");
            let mut entries = tokio::fs::read_dir(path)
                .await
                .with_context(|| format!("failed to list {path}"))?;
            let mut names = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let suffix = if entry.file_type().await?.is_dir() {
                    "/"
                } else {
                    ""
                };
                names.push(format!("{}{suffix}", entry.file_name().to_string_lossy()));
            }
            names.sort();
            Ok(names.join("\n"))
        }
        "grep" => {
            let pattern = str_arg(args, "pattern")?.to_owned();
            let root = args["path"].as_str().unwrap_or(".").to_owned();
            tokio::task::spawn_blocking(move || grep_files(&pattern, &root)).await?
        }
        "glob" => {
            let pattern = str_arg(args, "pattern")?.to_owned();
            let root = args["path"].as_str().unwrap_or(".").to_owned();
            tokio::task::spawn_blocking(move || glob_files(&pattern, &root)).await?
        }
        "run_command" => {
            let command = str_arg(args, "command")?;
            let output = tokio::time::timeout(
                COMMAND_TIMEOUT,
                tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .context("command timed out after 60s")??;

            let mut result = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.trim().is_empty() {
                result.push_str("\n[stderr]\n");
                result.push_str(&stderr);
            }
            if !output.status.success() {
                anyhow::bail!("exit status {}\n{}", output.status, truncate(result));
            }
            Ok(if result.trim().is_empty() {
                "(no output)".into()
            } else {
                result
            })
        }
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

fn apply_edit(content: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
    let matches = content.matches(old).count();
    match matches {
        0 => anyhow::bail!("old_string not found in file"),
        1 => Ok(content.replacen(old, new, 1)),
        n if replace_all => {
            let _ = n;
            Ok(content.replace(old, new))
        }
        n => anyhow::bail!(
            "old_string matches {n} times — include more surrounding context to make it unique, or set replace_all"
        ),
    }
}

fn grep_files(pattern: &str, root: &str) -> Result<String> {
    let re = regex::Regex::new(pattern).context("invalid regex")?;
    let mut out = Vec::new();

    for entry in ignore::WalkBuilder::new(root).build().flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if entry
            .metadata()
            .map_or(true, |m| m.len() > MAX_SEARCH_FILE_BYTES)
        {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue; // binary or unreadable
        };
        for (no, line) in content.lines().enumerate() {
            if re.is_match(line) {
                out.push(format!(
                    "{}:{}:{}",
                    entry.path().display(),
                    no + 1,
                    line.trim_end()
                ));
                if out.len() >= MAX_SEARCH_RESULTS {
                    out.push(format!("… stopped at {MAX_SEARCH_RESULTS} matches"));
                    return Ok(out.join("\n"));
                }
            }
        }
    }
    Ok(if out.is_empty() {
        "no matches".into()
    } else {
        out.join("\n")
    })
}

fn glob_files(pattern: &str, root: &str) -> Result<String> {
    let glob = globset::GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .context("invalid glob pattern")?
        .compile_matcher();
    let mut out = Vec::new();

    for entry in ignore::WalkBuilder::new(root).build().flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let relative = entry.path().strip_prefix(root).unwrap_or(entry.path());
        if glob.is_match(relative) || glob.is_match(entry.path()) {
            out.push(entry.path().display().to_string());
            if out.len() >= MAX_SEARCH_RESULTS {
                out.push(format!("… stopped at {MAX_SEARCH_RESULTS} files"));
                break;
            }
        }
    }
    out.sort();
    Ok(if out.is_empty() {
        "no files matched".into()
    } else {
        out.join("\n")
    })
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key]
        .as_str()
        .with_context(|| format!("missing required argument: {key}"))
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_OUTPUT_BYTES {
        let mut cut = MAX_OUTPUT_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n[output truncated]");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "t".into(),
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn reads_inside_cwd_are_auto_approved() {
        assert!(!requires_approval(&call(
            "read_file",
            json!({"path": "src/main.rs"})
        )));
        assert!(!requires_approval(&call("grep", json!({"pattern": "x"}))));
        assert!(!requires_approval(&call("list_directory", json!({}))));
    }

    #[test]
    fn reads_outside_cwd_require_approval() {
        assert!(requires_approval(&call(
            "read_file",
            json!({"path": "/etc/passwd"})
        )));
        assert!(requires_approval(&call(
            "read_file",
            json!({"path": "../secrets.txt"})
        )));
        assert!(requires_approval(&call(
            "list_directory",
            json!({"path": "/"})
        )));
    }

    #[test]
    fn mutations_always_require_approval() {
        assert!(requires_approval(&call(
            "write_file",
            json!({"path": "x", "content": ""})
        )));
        assert!(requires_approval(&call("edit_file", json!({"path": "x"}))));
        assert!(requires_approval(&call(
            "run_command",
            json!({"command": "ls"})
        )));
    }

    #[test]
    fn apply_edit_enforces_unique_match() {
        assert_eq!(apply_edit("a b a", "b", "c", false).unwrap(), "a c a");
        assert!(apply_edit("a b a", "a", "c", false).is_err());
        assert_eq!(apply_edit("a b a", "a", "c", true).unwrap(), "c b c");
        assert!(apply_edit("a b a", "z", "c", false).is_err());
    }

    #[tokio::test]
    async fn edit_file_round_trip() {
        let path = std::env::temp_dir().join(format!("shaltai-edit-{}.txt", std::process::id()));
        std::fs::write(&path, "hello world\n").unwrap();
        let (out, is_error) = execute(&call(
            "edit_file",
            json!({"path": path.to_str().unwrap(), "old_string": "world", "new_string": "rust"}),
        ))
        .await;
        assert!(!is_error, "{out}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello rust\n");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn diff_preview_shows_changes() {
        let lines = diff_lines("a\nb\nc\n", "a\nB\nc\n");
        assert!(lines.iter().any(|(t, l)| *t == '-' && l == "b"));
        assert!(lines.iter().any(|(t, l)| *t == '+' && l == "B"));
    }
}
