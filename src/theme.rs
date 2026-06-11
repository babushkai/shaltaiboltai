use ratatui::style::Color;

/// A complete palette with tonal depth, not just accent colors:
/// `bg` is the base canvas, `surface` is one elevation above it (cards,
/// input field, status bar, overlays), `border` sits between the two so
/// outlines read as structure rather than content. `bg`/`surface` of `None`
/// keep the terminal's own colors (the `terminal` theme).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theme {
    pub name: &'static str,
    pub bg: Option<Color>,
    pub surface: Option<Color>,
    pub border: Color,
    pub fg: Color,
    /// Secondary text: hints, timestamps, tool output, quotes.
    pub dim: Color,
    /// Primary accent: user gutter, focused input, model chip, selection.
    pub accent: Color,
    /// Secondary accent: headings, bullets, provider labels, hunk headers.
    pub accent2: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub code: Color,
}

const fn rgb(hex: u32) -> Color {
    Color::Rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

pub const MOCHA: Theme = Theme {
    name: "mocha",
    bg: Some(rgb(0x1e1e2e)),
    surface: Some(rgb(0x313244)),
    border: rgb(0x45475a),
    fg: rgb(0xcdd6f4),
    dim: rgb(0x9399b2),
    accent: rgb(0xcba6f7),
    accent2: rgb(0x89b4fa),
    success: rgb(0xa6e3a1),
    warning: rgb(0xf9e2af),
    error: rgb(0xf38ba8),
    code: rgb(0x94e2d5),
};

pub const TOKYO_NIGHT: Theme = Theme {
    name: "tokyo-night",
    bg: Some(rgb(0x1a1b26)),
    surface: Some(rgb(0x292e42)),
    border: rgb(0x3b4261),
    fg: rgb(0xc0caf5),
    dim: rgb(0x787c99),
    accent: rgb(0x7aa2f7),
    accent2: rgb(0xbb9af7),
    success: rgb(0x9ece6a),
    warning: rgb(0xe0af68),
    error: rgb(0xf7768e),
    code: rgb(0x7dcfff),
};

pub const ROSE_PINE: Theme = Theme {
    name: "rose-pine",
    bg: Some(rgb(0x191724)),
    surface: Some(rgb(0x26233a)),
    border: rgb(0x403d52),
    fg: rgb(0xe0def4),
    dim: rgb(0x908caa),
    accent: rgb(0xebbcba),
    accent2: rgb(0xc4a7e7),
    success: rgb(0x9ccfd8),
    warning: rgb(0xf6c177),
    error: rgb(0xeb6f92),
    code: rgb(0x9ccfd8),
};

pub const NORD: Theme = Theme {
    name: "nord",
    bg: Some(rgb(0x2e3440)),
    surface: Some(rgb(0x3b4252)),
    border: rgb(0x4c566a),
    fg: rgb(0xd8dee9),
    dim: rgb(0x8a92a5),
    accent: rgb(0x88c0d0),
    accent2: rgb(0x81a1c1),
    success: rgb(0xa3be8c),
    warning: rgb(0xebcb8b),
    error: rgb(0xbf616a),
    code: rgb(0x8fbcbb),
};

pub const GRUVBOX: Theme = Theme {
    name: "gruvbox",
    bg: Some(rgb(0x282828)),
    surface: Some(rgb(0x3c3836)),
    border: rgb(0x504945),
    fg: rgb(0xebdbb2),
    dim: rgb(0xa89984),
    accent: rgb(0x83a598),
    accent2: rgb(0xd3869b),
    success: rgb(0xb8bb26),
    warning: rgb(0xfabd2f),
    error: rgb(0xfb4934),
    code: rgb(0x8ec07c),
};

pub const LATTE: Theme = Theme {
    name: "latte",
    bg: Some(rgb(0xeff1f5)),
    surface: Some(rgb(0xe6e9ef)),
    border: rgb(0xbcc0cc),
    fg: rgb(0x4c4f69),
    dim: rgb(0x8c8fa1),
    accent: rgb(0x8839ef),
    accent2: rgb(0x1e66f5),
    success: rgb(0x40a02b),
    warning: rgb(0xdf8e1d),
    error: rgb(0xd20f39),
    code: rgb(0x179299),
};

/// Plain ANSI colors and no backgrounds — for terminals without truecolor.
pub const TERMINAL: Theme = Theme {
    name: "terminal",
    bg: None,
    surface: None,
    border: Color::DarkGray,
    fg: Color::White,
    dim: Color::DarkGray,
    accent: Color::Cyan,
    accent2: Color::Magenta,
    success: Color::Green,
    warning: Color::Yellow,
    error: Color::Red,
    code: Color::Yellow,
};

pub const DEFAULT: Theme = MOCHA;

pub fn all() -> &'static [Theme] {
    &[
        MOCHA,
        TOKYO_NIGHT,
        ROSE_PINE,
        NORD,
        GRUVBOX,
        LATTE,
        TERMINAL,
    ]
}

pub fn by_name(name: &str) -> Option<Theme> {
    all().iter().find(|t| t.name == name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_names_are_unique_and_resolvable() {
        let mut names: Vec<_> = all().iter().map(|t| t.name).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
        for t in all() {
            assert_eq!(by_name(t.name).map(|x| x.name), Some(t.name));
        }
        assert!(by_name("nonexistent").is_none());
    }

    #[test]
    fn default_theme_is_listed() {
        assert!(all().iter().any(|t| t.name == DEFAULT.name));
    }

    #[test]
    fn themed_palettes_define_both_elevations() {
        for t in all() {
            // A theme either keeps the terminal's colors entirely or defines
            // both the base and the elevated surface.
            assert_eq!(t.bg.is_some(), t.surface.is_some(), "{}", t.name);
        }
    }
}
