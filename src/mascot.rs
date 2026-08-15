//! Terminal-native Shaltaiboltai frames generated from the character artwork.
//!
//! `build.rs` isolates the four connected poses from the sprite sheet, keeps
//! their shared baseline, and samples their source colors into paired vertical
//! pixels. The TUI renders each pair as one half-block cell without replacing
//! the character with a separately drawn terminal icon.

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

pub fn frame(state: MascotState, tick: u64) -> &'static MascotFrame {
    match state {
        MascotState::Idle | MascotState::Waiting => &GENERATED_FRAMES[0],
        MascotState::Thinking => {
            if (tick / 2).is_multiple_of(2) {
                &GENERATED_FRAMES[0]
            } else {
                &GENERATED_FRAMES[2]
            }
        }
        MascotState::Working => &GENERATED_FRAMES[(tick / 2) as usize % GENERATED_FRAMES.len()],
    }
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
}
