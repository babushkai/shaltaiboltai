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
    /// Raised controls, cards, and selected rows.
    pub elevated: Option<Color>,
    /// Hover/selection emphasis one step above `elevated`.
    pub hover: Option<Color>,
    pub border: Color,
    pub fg: Color,
    /// Normal secondary copy, distinct from genuinely muted metadata.
    pub secondary: Color,
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

/// Shaltaiboltai's authored dark identity, transferred from the original
/// TypeScript Ink & Paper interface.
pub const INK: Theme = Theme {
    name: "ink",
    bg: Some(rgb(0x141419)),
    surface: Some(rgb(0x1b1b21)),
    elevated: Some(rgb(0x232329)),
    hover: Some(rgb(0x2b2b33)),
    border: rgb(0x34333b),
    fg: rgb(0xece9e2),
    secondary: rgb(0xc5c1b8),
    dim: rgb(0x8f8b81),
    accent: rgb(0xf2765d),
    accent2: rgb(0x8fb0d1),
    success: rgb(0xa8bd8e),
    warning: rgb(0xd6b571),
    error: rgb(0xe8707c),
    code: rgb(0xc5a3c9),
};

/// Warm washi-paper companion to [`INK`].
pub const PAPER: Theme = Theme {
    name: "paper",
    bg: Some(rgb(0xfaf6ef)),
    surface: Some(rgb(0xf2ecdf)),
    elevated: Some(rgb(0xe9e1d0)),
    hover: Some(rgb(0xe0d6c1)),
    border: rgb(0xdbd2bf),
    fg: rgb(0x26231e),
    secondary: rgb(0x4a463e),
    dim: rgb(0x7d7668),
    accent: rgb(0xbd3a1c),
    accent2: rgb(0x2e5f8a),
    success: rgb(0x55703a),
    warning: rgb(0x85681c),
    error: rgb(0xab2330),
    code: rgb(0x7c5183),
};

pub const MOCHA: Theme = Theme {
    name: "mocha",
    bg: Some(rgb(0x1e1e2e)),
    surface: Some(rgb(0x313244)),
    elevated: Some(rgb(0x3b3d52)),
    hover: Some(rgb(0x45475a)),
    border: rgb(0x45475a),
    fg: rgb(0xcdd6f4),
    secondary: rgb(0xbac2de),
    dim: rgb(0xa6adc8),
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
    elevated: Some(rgb(0x33384f)),
    hover: Some(rgb(0x414868)),
    border: rgb(0x3b4261),
    fg: rgb(0xc0caf5),
    secondary: rgb(0xa9b1d6),
    dim: rgb(0x9aa5ce),
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
    elevated: Some(rgb(0x312e46)),
    hover: Some(rgb(0x403d52)),
    border: rgb(0x403d52),
    fg: rgb(0xe0def4),
    secondary: rgb(0xc4c0dc),
    dim: rgb(0xa19bbd),
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
    elevated: Some(rgb(0x434c5e)),
    hover: Some(rgb(0x4c566a)),
    border: rgb(0x4c566a),
    fg: rgb(0xd8dee9),
    secondary: rgb(0xc2c8d0),
    dim: rgb(0xaeb6c4),
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
    elevated: Some(rgb(0x504945)),
    hover: Some(rgb(0x665c54)),
    border: rgb(0x504945),
    fg: rgb(0xebdbb2),
    secondary: rgb(0xd5c4a1),
    dim: rgb(0xbdae93),
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
    elevated: Some(rgb(0xdce0e8)),
    hover: Some(rgb(0xcdd0da)),
    border: rgb(0xbcc0cc),
    fg: rgb(0x4c4f69),
    secondary: rgb(0x5c5f77),
    dim: rgb(0x5c5f77),
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
    elevated: None,
    hover: None,
    border: Color::DarkGray,
    fg: Color::White,
    secondary: Color::Gray,
    dim: Color::DarkGray,
    accent: Color::Cyan,
    accent2: Color::Magenta,
    success: Color::Green,
    warning: Color::Yellow,
    error: Color::Red,
    code: Color::Yellow,
};

pub const DEFAULT: Theme = INK;

pub fn all() -> &'static [Theme] {
    &[
        INK,
        PAPER,
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
            // the complete elevation ladder.
            assert_eq!(t.bg.is_some(), t.surface.is_some(), "{}", t.name);
            assert_eq!(t.bg.is_some(), t.elevated.is_some(), "{}", t.name);
            assert_eq!(t.bg.is_some(), t.hover.is_some(), "{}", t.name);
        }
    }

    #[test]
    fn ink_and_paper_match_the_original_product_tokens() {
        assert_eq!(INK.bg, Some(rgb(0x141419)));
        assert_eq!(INK.accent, rgb(0xf2765d));
        assert_eq!(INK.fg, rgb(0xece9e2));
        assert_eq!(PAPER.bg, Some(rgb(0xfaf6ef)));
        assert_eq!(PAPER.accent, rgb(0xbd3a1c));
        assert_eq!(PAPER.fg, rgb(0x26231e));
    }

    #[test]
    fn text_hierarchy_remains_readable_on_elevated_surfaces() {
        fn luminance(color: Color) -> Option<f64> {
            let Color::Rgb(r, g, b) = color else {
                return None;
            };
            let channel = |value: u8| {
                let value = f64::from(value) / 255.0;
                if value <= 0.039_28 {
                    value / 12.92
                } else {
                    ((value + 0.055) / 1.055).powf(2.4)
                }
            };
            Some(0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b))
        }

        for theme in all() {
            let (Some(surface), Some(secondary), Some(dim)) = (
                theme.surface.and_then(luminance),
                luminance(theme.secondary),
                luminance(theme.dim),
            ) else {
                continue;
            };
            let secondary_ratio = (surface.max(secondary) + 0.05) / (surface.min(secondary) + 0.05);
            assert!(
                secondary_ratio >= 4.5,
                "{} secondary contrast is {secondary_ratio:.2}:1",
                theme.name
            );
            // Muted copy is reserved for non-essential metadata and follows
            // the authored Ink & Paper token exactly; 3:1 keeps that quieter
            // tier legible without flattening it into body text.
            let muted_ratio = (surface.max(dim) + 0.05) / (surface.min(dim) + 0.05);
            assert!(
                muted_ratio >= 3.0,
                "{} muted contrast is {muted_ratio:.2}:1",
                theme.name
            );
        }
    }
}
