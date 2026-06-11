use crate::providers::ImageData;
use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use std::path::{Path, PathBuf};

/// Provider limits sit around 5MB per image; reject earlier with a clear error.
const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;
const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// Find existing image files referenced in a message. Handles the forms
/// terminals produce on drag-and-drop: backslash-escaped spaces and
/// quoted paths, plus `~` expansion.
pub fn extract_image_paths(text: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut token = String::new();
    let mut quote: Option<char> = None;
    let mut chars = text.chars().peekable();

    let finish = |token: &mut String, out: &mut Vec<PathBuf>| {
        if token.is_empty() {
            return;
        }
        let path = expand_home(token);
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
            && path.is_file()
        {
            out.push(path);
        }
        token.clear();
    };

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(&next) = chars.peek() {
                    token.push(next);
                    chars.next();
                }
            }
            '"' | '\'' => match quote {
                Some(q) if q == c => quote = None,
                Some(_) => token.push(c),
                None => quote = Some(c),
            },
            c if c.is_whitespace() && quote.is_none() => finish(&mut token, &mut out),
            c => token.push(c),
        }
    }
    finish(&mut token, &mut out);
    out
}

fn expand_home(token: &str) -> PathBuf {
    if let Some(rest) = token.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(token)
}

pub fn load_image(path: &Path) -> Result<(String, ImageData)> {
    let size = std::fs::metadata(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .len();
    anyhow::ensure!(
        size <= MAX_IMAGE_BYTES,
        "{} is {:.1}MB — images are limited to 5MB",
        path.display(),
        size as f64 / (1024.0 * 1024.0)
    );
    let bytes = std::fs::read(path)?;
    let media_type = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        other => anyhow::bail!("unsupported image type: {other:?}"),
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    Ok((
        name,
        ImageData {
            media_type: media_type.into(),
            data: BASE64.encode(&bytes),
        },
    ))
}

/// Grab an image from the system clipboard (macOS): screenshots and most
/// "Copy image" actions land as PNG; browser copies often land as TIFF,
/// which `sips` (ships with macOS) converts.
#[cfg(target_os = "macos")]
pub fn clipboard_image() -> Result<ImageData> {
    let stem = std::env::temp_dir().join(format!("shaltai-clip-{}", std::process::id()));

    let grab = |class: &str, path: &Path| -> Vec<u8> {
        let script = format!(
            "set f to (open for access POSIX file \"{}\" with write permission)\n\
             set eof f to 0\n\
             try\n\
             write (the clipboard as {class}) to f\n\
             end try\n\
             close access f",
            path.display()
        );
        let _ = std::process::Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .output();
        std::fs::read(path).unwrap_or_default()
    };

    let png_path = stem.with_extension("png");
    let mut bytes = grab("«class PNGf»", &png_path);

    if bytes.is_empty() {
        let tiff_path = stem.with_extension("tiff");
        if !grab("«class TIFF»", &tiff_path).is_empty() {
            let _ = std::process::Command::new("sips")
                .args(["-s", "format", "png"])
                .arg(&tiff_path)
                .arg("--out")
                .arg(&png_path)
                .output();
            bytes = std::fs::read(&png_path).unwrap_or_default();
        }
        let _ = std::fs::remove_file(&tiff_path);
    }
    let _ = std::fs::remove_file(&png_path);

    anyhow::ensure!(!bytes.is_empty(), "no image in the clipboard");
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_IMAGE_BYTES,
        "clipboard image exceeds the 5MB limit"
    );
    Ok(ImageData {
        media_type: "image/png".into(),
        data: BASE64.encode(&bytes),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn clipboard_image() -> Result<ImageData> {
    anyhow::bail!("clipboard image capture is only supported on macOS for now")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_image(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("shaltai-img-{}-{name}", std::process::id()));
        std::fs::write(&path, b"fake-png-bytes").unwrap();
        path
    }

    #[test]
    fn finds_plain_escaped_and_quoted_paths() {
        let plain = temp_image("a.png");
        let spaced = std::env::temp_dir().join(format!("shaltai img {}.jpg", std::process::id()));
        std::fs::write(&spaced, b"x").unwrap();

        let text = format!(
            "look at {} and \"{}\" please",
            plain.display(),
            spaced.display()
        );
        let found = extract_image_paths(&text);
        assert_eq!(found, vec![plain.clone(), spaced.clone()]);

        // Backslash-escaped spaces, as macOS Terminal produces on drag-drop.
        let escaped = spaced.display().to_string().replace(' ', "\\ ");
        assert_eq!(extract_image_paths(&escaped), vec![spaced.clone()]);

        std::fs::remove_file(plain).ok();
        std::fs::remove_file(spaced).ok();
    }

    #[test]
    fn ignores_missing_files_and_other_extensions() {
        assert!(extract_image_paths("/nonexistent/x.png describe it").is_empty());
        let rs = temp_image("c.rs");
        assert!(extract_image_paths(&rs.display().to_string()).is_empty());
        std::fs::remove_file(rs).ok();
    }

    #[test]
    fn loads_and_encodes_an_image() {
        let path = temp_image("d.png");
        let (name, image) = load_image(&path).unwrap();
        assert!(name.ends_with("d.png"));
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.data, BASE64.encode(b"fake-png-bytes"));
        std::fs::remove_file(path).ok();
    }
}
