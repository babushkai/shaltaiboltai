//! Terminal-native Shaltaiboltai sprite frames.
//!
//! The reference character lives in `assets/mascot/shaltaiboltai-reference.png`.
//! These fixed-cell poses preserve its silhouette and color regions without
//! depending on terminal-specific image protocols.

pub const FRAME_WIDTH: usize = 25;
pub const FRAME_HEIGHT: usize = 6;

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

#[derive(Debug)]
pub struct MascotFrame {
    pub rows: [&'static str; FRAME_HEIGHT],
}

const REST: MascotFrame = MascotFrame {
    rows: [
        "       ╭─────────╮       ",
        "   ╭───┤  ⌒   ⌒  ├───╮   ",
        "   ╰───┤    ▾    ├───╯   ",
        "        ╰━≋≋≋≋≋━╯        ",
        "        ╭╯  ●  ╰╮        ",
        "       ▟▛       ▜▙       ",
    ],
};

const LEFT_STEP: MascotFrame = MascotFrame {
    rows: [
        "   ╭──╮╭─────────╮       ",
        "    ╰──┤  ⌒   ⌒  ├───╮   ",
        "       ┤    ▾    ├───╯   ",
        "        ╰━≋≋≋≋≋━╯        ",
        "        ╭╯  ●  ╰╮        ",
        "       ▟▛       ▜▙       ",
    ],
};

const HOP: MascotFrame = MascotFrame {
    rows: [
        "   ╭──╮╭─────────╮╭──╮   ",
        "    ╰──┤  ⌒   ⌒  ├──╯    ",
        "       │    ▾    │       ",
        "        ╰━≋≋≋≋≋━╯        ",
        "        ╭╯  ●  ╰╮        ",
        "        ▟▛     ▜▙        ",
    ],
};

const RIGHT_STEP: MascotFrame = MascotFrame {
    rows: [
        "       ╭─────────╮╭──╮   ",
        "   ╭───┤  ⌒   ⌒  ├──╯    ",
        "   ╰───┤    ▾    │       ",
        "        ╰━≋≋≋≋≋━╯        ",
        "        ╭╯  ●  ╰╮        ",
        "       ▟▛       ▜▙       ",
    ],
};

pub fn frame(state: MascotState, tick: u64) -> &'static MascotFrame {
    match state {
        MascotState::Idle | MascotState::Waiting => &REST,
        MascotState::Thinking => {
            if (tick / 2).is_multiple_of(2) {
                &REST
            } else {
                &HOP
            }
        }
        MascotState::Working => {
            const DANCE: [&MascotFrame; 4] = [&REST, &LEFT_STEP, &HOP, &RIGHT_STEP];
            DANCE[(tick / 2) as usize % DANCE.len()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn every_pose_has_the_same_terminal_geometry() {
        for pose in [&REST, &LEFT_STEP, &HOP, &RIGHT_STEP] {
            assert_eq!(pose.rows.len(), FRAME_HEIGHT);
            for row in pose.rows {
                assert_eq!(UnicodeWidthStr::width(row), FRAME_WIDTH, "{row:?}");
                assert!(!row.chars().any(char::is_control), "{row:?}");
            }
        }
    }

    #[test]
    fn working_dances_while_idle_and_waiting_stay_still() {
        assert_ne!(
            frame(MascotState::Working, 0).rows,
            frame(MascotState::Working, 2).rows
        );
        assert_eq!(
            frame(MascotState::Idle, 0).rows,
            frame(MascotState::Idle, 999).rows
        );
        assert_eq!(
            frame(MascotState::Waiting, 0).rows,
            frame(MascotState::Waiting, 999).rows
        );
    }
}
