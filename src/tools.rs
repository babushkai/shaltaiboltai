use crate::providers::{ToolCall, ToolDef};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_OUTPUT_BYTES: usize = 32 * 1024;

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
            description: "Write content to a file, creating it (and parent directories) if needed, overwriting if it exists.",
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

/// Read-only tools run without confirmation; anything that mutates the
/// filesystem or executes code requires user approval.
pub fn requires_approval(name: &str) -> bool {
    matches!(name, "write_file" | "run_command")
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
        "list_directory" => format!(
            "list_directory {}",
            call.arguments["path"].as_str().unwrap_or(".")
        ),
        "run_command" => format!("run_command: {}", arg("command")),
        other => format!("{other} {}", call.arguments),
    }
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
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await?;
                }
            }
            tokio::fs::write(path, content)
                .await
                .with_context(|| format!("failed to write {path}"))?;
            Ok(format!("wrote {} bytes to {path}", content.len()))
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
        "run_command" => {
            let command = str_arg(args, "command")?;
            let output = tokio::time::timeout(
                COMMAND_TIMEOUT,
                tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
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
