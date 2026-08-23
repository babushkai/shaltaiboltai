//! Terminal-native Shaltaiboltai frames generated from the character artwork.
//!
//! `build.rs` isolates the four connected poses from the sprite sheet and keeps
//! their shared baseline. Kitty-capable terminals receive the normalized
//! high-resolution poses directly; paired source-color pixels provide the
//! portable half-block fallback without substituting another character.

use anyhow::Context;
use image::{imageops, DynamicImage, ImageFormat, RgbaImage};
use ratatui::layout::Rect;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::kitty::Kitty;
use ratatui_image::protocol::Protocol;
use std::fmt::Write as _;
use std::io::{self, Write as _};

const OPAQUE_FLAG: u32 = 0x0100_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MascotCell {
    top: u32,
    bottom: u32,
}

impl MascotCell {
    pub const fn new(top: u32, bottom: u32) -> Self {
        Self { top, bottom }
    }

    pub const fn top(self) -> u32 {
        self.top
    }

    pub const fn bottom(self) -> u32 {
        self.bottom
    }
}

include!(concat!(env!("OUT_DIR"), "/mascot_frames.rs"));

const NATIVE_ART_AREA: Rect = Rect::new(0, 0, 28, 17);
const NATIVE_POSES: [&[u8]; 4] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/mascot_pose_0.png")),
    include_bytes!(concat!(env!("OUT_DIR"), "/mascot_pose_1.png")),
    include_bytes!(concat!(env!("OUT_DIR"), "/mascot_pose_2.png")),
    include_bytes!(concat!(env!("OUT_DIR"), "/mascot_pose_3.png")),
];

/// High-resolution pose cache for terminals that implement Kitty graphics.
///
/// Every pose is decoded and encoded once during startup. Animation only
/// changes which fixed protocol is placed in the transcript, so no image work
/// runs in the input/render loop.
pub struct NativeMascot {
    poses: Vec<Protocol>,
    image_ids: [u32; 4],
}

impl NativeMascot {
    /// Detect direct Ghostty/Kitty sessions from their terminal identity and
    /// derive physical cell size without starting a competing stdin reader.
    /// Unsupported terminals and multiplexers keep the cell fallback.
    pub fn detect() -> anyhow::Result<Option<Self>> {
        if !direct_kitty_terminal() {
            return Ok(None);
        }
        let window = crossterm::terminal::window_size().context("read terminal pixel geometry")?;
        if window.columns == 0 || window.rows == 0 || window.width == 0 || window.height == 0 {
            return Ok(None);
        }
        let font_size = (window.width / window.columns, window.height / window.rows);
        if font_size.0 == 0 || font_size.1 == 0 {
            return Ok(None);
        }
        #[allow(deprecated)]
        let mut picker = Picker::from_fontsize(font_size);
        picker.set_protocol_type(ProtocolType::Kitty);
        Self::from_picker(picker)
    }

    pub(crate) fn from_picker(picker: Picker) -> anyhow::Result<Option<Self>> {
        if picker.protocol_type() != ProtocolType::Kitty {
            return Ok(None);
        }

        let image_id_base = (rand::random::<u32>() & 0x7fff_fffc).saturating_add(4);
        let image_ids = std::array::from_fn(|index| image_id_base + index as u32);
        let poses = NATIVE_POSES
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                let image = image::load_from_memory_with_format(bytes, ImageFormat::Png)
                    .with_context(|| format!("decode native mascot pose {index}"))?;
                kitty_protocol(image, picker.font_size(), image_ids[index])
                    .with_context(|| format!("encode native mascot pose {index}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Some(Self { poses, image_ids }))
    }

    pub fn protocol(&self, state: MascotState, tick: u64) -> &Protocol {
        &self.poses[frame_index(state, tick)]
    }

    /// Delete only the terminal-side images allocated by this process.
    pub fn clear(&self) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        stdout.write_all(delete_sequence(self.image_ids).as_bytes())?;
        stdout.flush()
    }
}

fn direct_kitty_terminal() -> bool {
    let term = std::env::var("TERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let program = std::env::var("TERM_PROGRAM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    direct_kitty_terminal_identity(&term, &program, std::env::var_os("TMUX").is_some())
}

fn direct_kitty_terminal_identity(term: &str, program: &str, in_tmux: bool) -> bool {
    if in_tmux || term.starts_with("tmux") || term.starts_with("screen") || program == "tmux" {
        return false;
    }
    term.contains("ghostty")
        || term.contains("kitty")
        || program.contains("ghostty")
        || program.contains("kitty")
}

fn kitty_protocol(
    image: DynamicImage,
    font_size: (u16, u16),
    image_id: u32,
) -> anyhow::Result<Protocol> {
    let pixel_width = u32::from(NATIVE_ART_AREA.width) * u32::from(font_size.0);
    let pixel_height = u32::from(NATIVE_ART_AREA.height) * u32::from(font_size.1);
    let resized = image.resize(pixel_width, pixel_height, imageops::FilterType::Lanczos3);
    let mut canvas = RgbaImage::new(pixel_width, pixel_height);
    let x = i64::from((pixel_width - resized.width()) / 2);
    let y = i64::from((pixel_height - resized.height()) / 2);
    imageops::overlay(&mut canvas, &resized, x, y);
    Ok(Protocol::Kitty(Kitty::new(
        DynamicImage::ImageRgba8(canvas),
        NATIVE_ART_AREA,
        image_id,
        false,
    )?))
}

fn delete_sequence(image_ids: [u32; 4]) -> String {
    let mut sequence = String::new();
    for image_id in image_ids {
        write!(sequence, "\x1b_Ga=d,d=I,i={image_id};\x1b\\").unwrap();
    }
    sequence
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MascotState {
    Idle,
    Working,
    Waiting,
    Thinking,
}

impl MascotState {
    pub const fn is_animated(self) -> bool {
        matches!(self, Self::Working | Self::Thinking)
    }
}

pub fn color(pixel: u32) -> Option<(u8, u8, u8)> {
    (pixel & OPAQUE_FLAG != 0).then_some((
        ((pixel >> 16) & 0xff) as u8,
        ((pixel >> 8) & 0xff) as u8,
        (pixel & 0xff) as u8,
    ))
}

pub fn frame_index(state: MascotState, tick: u64) -> usize {
    match state {
        MascotState::Idle | MascotState::Waiting => 0,
        MascotState::Thinking => {
            if (tick / 2).is_multiple_of(2) {
                0
            } else {
                2
            }
        }
        MascotState::Working => (tick / 2) as usize % GENERATED_FRAMES.len(),
    }
}

pub fn frame(state: MascotState, tick: u64) -> &'static MascotFrame {
    &GENERATED_FRAMES[frame_index(state, tick)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_frames_keep_one_stable_canvas_and_humpty_dumpty_colors() {
        assert_eq!(GENERATED_FRAMES.len(), 4);
        assert_eq!(FRAME_WIDTH, 26);
        assert_eq!(FRAME_HEIGHT, 17);

        for frame in &GENERATED_FRAMES {
            let colors = frame
                .cells
                .iter()
                .flatten()
                .flat_map(|cell| [color(cell.top()), color(cell.bottom())])
                .flatten()
                .collect::<Vec<_>>();
            assert!(colors.len() > 180, "pose lost too much of its silhouette");
            assert!(
                colors
                    .iter()
                    .any(|&(r, g, b)| r > 190 && g > 155 && b > 105),
                "pose is missing the warm egg shell"
            );
            assert!(
                colors.iter().any(|&(r, g, b)| b > 45 && b > r && b > g),
                "pose is missing the navy coat"
            );
            assert!(
                colors
                    .iter()
                    .any(|&(r, g, _)| r > 120 && r > g.saturating_add(35)),
                "pose is missing the red waistcoat"
            );
            assert!(
                colors
                    .iter()
                    .any(|&(r, g, b)| b > 65 && g > r.saturating_add(20)),
                "pose is missing the teal breeches"
            );

            let edge_pixels = frame
                .cells
                .iter()
                .flat_map(|row| [row[0], row[FRAME_WIDTH - 1]])
                .flat_map(|cell| [cell.top(), cell.bottom()])
                .filter(|pixel| color(*pixel).is_some())
                .count();
            assert!(edge_pixels < 3, "pose touches the normalized canvas edge");
        }
    }

    #[test]
    fn working_moves_while_idle_and_waiting_stay_still() {
        assert_ne!(
            frame(MascotState::Working, 0).cells,
            frame(MascotState::Working, 2).cells
        );
        assert_eq!(
            frame(MascotState::Idle, 0).cells,
            frame(MascotState::Idle, 999).cells
        );
        assert_eq!(
            frame(MascotState::Waiting, 0).cells,
            frame(MascotState::Waiting, 999).cells
        );
    }

    #[test]
    fn native_poses_keep_the_full_resolution_normalized_canvas() {
        for bytes in NATIVE_POSES {
            let pose = image::load_from_memory_with_format(bytes, ImageFormat::Png).unwrap();
            assert_eq!((pose.width(), pose.height()), (600, 736));
            assert_eq!(pose.to_rgba8().get_pixel(0, 0).0[3], 0);
        }
    }

    #[test]
    fn kitty_cache_builds_four_fixed_high_resolution_protocols() {
        let mut picker = Picker::halfblocks();
        picker.set_protocol_type(ProtocolType::Kitty);
        let native = NativeMascot::from_picker(picker)
            .unwrap()
            .expect("forced Kitty renderer");

        assert_eq!(native.poses.len(), NATIVE_POSES.len());
        for pose in &native.poses {
            assert_eq!(pose.area(), NATIVE_ART_AREA);
        }
    }

    #[test]
    fn non_kitty_protocol_uses_the_existing_cell_fallback() {
        assert!(NativeMascot::from_picker(Picker::halfblocks())
            .unwrap()
            .is_none());
    }

    #[test]
    fn kitty_cleanup_targets_every_owned_image_id() {
        let sequence = delete_sequence([11, 12, 13, 14]);
        for image_id in [11, 12, 13, 14] {
            assert!(sequence.contains(&format!("a=d,d=I,i={image_id}")));
        }
        assert_eq!(sequence.matches("\x1b_G").count(), 4);
    }

    #[test]
    fn native_detection_is_limited_to_direct_ghostty_and_kitty_sessions() {
        assert!(direct_kitty_terminal_identity(
            "xterm-ghostty",
            "ghostty",
            false
        ));
        assert!(direct_kitty_terminal_identity("xterm-kitty", "", false));
        assert!(!direct_kitty_terminal_identity(
            "tmux-256color",
            "ghostty",
            true
        ));
        assert!(!direct_kitty_terminal_identity(
            "screen-256color",
            "ghostty",
            true
        ));
        assert!(!direct_kitty_terminal_identity(
            "xterm-256color",
            "apple_terminal",
            false
        ));
    }
}
