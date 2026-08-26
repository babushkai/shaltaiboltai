//! Dependency-free parsing for the interactive TUI's startup options.
//!
//! Parsing deliberately stops short of resolving or canonicalizing paths. That
//! work belongs to startup policy construction, where the selected working
//! directory and the host filesystem are both available.

use crate::sandbox::LINUX_SECCOMP_EXEC_OPTION;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

const DANGEROUS_BYPASS: &str = "--dangerously-bypass-approvals-and-sandbox";

/// Sandbox selection supplied on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }

    fn parse(value: &OsStr) -> Option<Self> {
        match value.to_str()? {
            "read-only" => Some(Self::ReadOnly),
            "workspace-write" => Some(Self::WorkspaceWrite),
            "danger-full-access" => Some(Self::DangerFullAccess),
            _ => None,
        }
    }
}

impl fmt::Display for SandboxMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Human-approval policy supplied on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalPolicy {
    OnRequest,
    Never,
}

impl ApprovalPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OnRequest => "on-request",
            Self::Never => "never",
        }
    }

    fn parse(value: &OsStr) -> Option<Self> {
        match value.to_str()? {
            "on-request" => Some(Self::OnRequest),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

impl fmt::Display for ApprovalPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Raw, validated command-line options.
///
/// Path values retain their original spelling. In particular, relative paths
/// are not interpreted until the caller has applied `cwd`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LaunchOptions {
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub cwd: Option<PathBuf>,
    pub additional_writable_dirs: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
    pub sandbox_mode: Option<SandboxMode>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub no_alt_screen: bool,
    pub full_auto: bool,
    pub dangerously_bypass_approvals_and_sandbox: bool,
    /// Internal Linux child stage used after Bubblewrap has established the
    /// filesystem namespace. Intentionally omitted from public help.
    pub sandbox_seccomp_command: Option<OsString>,
    pub help: bool,
    pub version: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    UnknownArgument(OsString),
    UnexpectedPositional(OsString),
    MissingValue {
        option: &'static str,
    },
    EmptyValue {
        option: &'static str,
    },
    NonUtf8Value {
        option: &'static str,
        value: OsString,
    },
    InvalidValue {
        option: &'static str,
        value: OsString,
        expected: &'static str,
    },
    UnexpectedValue {
        option: &'static str,
    },
    DuplicateOption {
        option: &'static str,
    },
    ConflictingOptions {
        first: &'static str,
        second: &'static str,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArgument(argument) => write!(
                formatter,
                "unrecognized argument `{}`",
                argument.to_string_lossy()
            ),
            Self::UnexpectedPositional(argument) => write!(
                formatter,
                "unexpected positional argument `{}`; at most one PROMPT is accepted",
                argument.to_string_lossy()
            ),
            Self::MissingValue { option } => {
                write!(formatter, "option `{option}` requires a value")
            }
            Self::EmptyValue { option } => {
                write!(formatter, "option `{option}` requires a non-empty value")
            }
            Self::NonUtf8Value { option, value } => write!(
                formatter,
                "value `{}` for `{option}` must be valid UTF-8",
                value.to_string_lossy()
            ),
            Self::InvalidValue {
                option,
                value,
                expected,
            } => write!(
                formatter,
                "invalid value `{}` for `{option}`; expected one of: {expected}",
                value.to_string_lossy()
            ),
            Self::UnexpectedValue { option } => {
                write!(formatter, "option `{option}` does not take a value")
            }
            Self::DuplicateOption { option } => {
                write!(formatter, "option `{option}` may only be specified once")
            }
            Self::ConflictingOptions { first, second } => {
                write!(formatter, "option `{first}` cannot be used with `{second}`")
            }
        }
    }
}

impl Error for ParseError {}

/// Parse arguments excluding the binary name.
///
/// `OsString` values are retained for separated path arguments, allowing paths
/// that are not valid UTF-8 on platforms that support them. Inline
/// `--option=value` arguments necessarily use UTF-8 because option names are
/// textual.
pub fn parse_args<I, T>(arguments: I) -> Result<LaunchOptions, ParseError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let arguments: Vec<OsString> = arguments.into_iter().map(Into::into).collect();
    let mut options = LaunchOptions::default();
    let mut index = 0;

    while index < arguments.len() {
        let argument = &arguments[index];
        index += 1;

        if argument == "--" {
            for positional in &arguments[index..] {
                set_positional_prompt(&mut options, positional)?;
            }
            break;
        }

        let Some(text) = argument.to_str() else {
            return Err(ParseError::UnknownArgument(argument.clone()));
        };

        match text {
            "-C" => {
                let value = required_value(&arguments, &mut index, "--cd", None)?;
                set_once(&mut options.cwd, PathBuf::from(value), "--cd")?;
            }
            "-m" => {
                let value = required_text_value(&arguments, &mut index, "--model", None)?;
                set_once(&mut options.model, value, "--model")?;
            }
            "-s" => {
                let value = required_value(&arguments, &mut index, "--sandbox", None)?;
                let mode = SandboxMode::parse(&value).ok_or(ParseError::InvalidValue {
                    option: "--sandbox",
                    value,
                    expected: "read-only, workspace-write, danger-full-access",
                })?;
                set_once(&mut options.sandbox_mode, mode, "--sandbox")?;
            }
            "-a" => {
                let value = required_value(&arguments, &mut index, "--ask-for-approval", None)?;
                let policy = ApprovalPolicy::parse(&value).ok_or(ParseError::InvalidValue {
                    option: "--ask-for-approval",
                    value,
                    expected: "on-request, never",
                })?;
                set_once(&mut options.approval_policy, policy, "--ask-for-approval")?;
            }
            "-i" => {
                let value = required_value(&arguments, &mut index, "--image", None)?;
                append_path_list(&mut options.images, value, "--image")?;
            }
            "-h" => set_flag(&mut options.help, "--help")?,
            "-V" => set_flag(&mut options.version, "--version")?,
            _ if text.starts_with("--") => {
                parse_long_option(text, &arguments, &mut index, &mut options)?;
            }
            _ if text.starts_with('-') => {
                return Err(ParseError::UnknownArgument(argument.clone()));
            }
            _ => set_positional_prompt(&mut options, argument)?,
        }
    }

    validate_conflicts(&options)?;
    Ok(options)
}

fn parse_long_option(
    argument: &str,
    arguments: &[OsString],
    index: &mut usize,
    options: &mut LaunchOptions,
) -> Result<(), ParseError> {
    let (name, inline_value) = argument[2..]
        .split_once('=')
        .map_or((&argument[2..], None), |(name, value)| (name, Some(value)));

    match name {
        "cd" => {
            let value = required_value(arguments, index, "--cd", inline_value)?;
            set_once(&mut options.cwd, PathBuf::from(value), "--cd")
        }
        "model" => {
            let value = required_text_value(arguments, index, "--model", inline_value)?;
            set_once(&mut options.model, value, "--model")
        }
        "add-dir" => {
            let value = required_value(arguments, index, "--add-dir", inline_value)?;
            options.additional_writable_dirs.push(PathBuf::from(value));
            Ok(())
        }
        "image" => {
            let value = required_value(arguments, index, "--image", inline_value)?;
            append_path_list(&mut options.images, value, "--image")
        }
        "sandbox" => {
            let value = required_value(arguments, index, "--sandbox", inline_value)?;
            let mode = SandboxMode::parse(&value).ok_or(ParseError::InvalidValue {
                option: "--sandbox",
                value,
                expected: "read-only, workspace-write, danger-full-access",
            })?;
            set_once(&mut options.sandbox_mode, mode, "--sandbox")
        }
        "ask-for-approval" => {
            let value = required_value(arguments, index, "--ask-for-approval", inline_value)?;
            let policy = ApprovalPolicy::parse(&value).ok_or(ParseError::InvalidValue {
                option: "--ask-for-approval",
                value,
                expected: "on-request, never",
            })?;
            set_once(&mut options.approval_policy, policy, "--ask-for-approval")
        }
        "no-alt-screen" => {
            reject_inline_value("--no-alt-screen", inline_value)?;
            set_flag(&mut options.no_alt_screen, "--no-alt-screen")
        }
        "full-auto" => {
            reject_inline_value("--full-auto", inline_value)?;
            set_flag(&mut options.full_auto, "--full-auto")
        }
        "dangerously-bypass-approvals-and-sandbox" => {
            reject_inline_value(DANGEROUS_BYPASS, inline_value)?;
            set_flag(
                &mut options.dangerously_bypass_approvals_and_sandbox,
                DANGEROUS_BYPASS,
            )
        }
        "__sandbox-seccomp-exec" => {
            let value = required_value(arguments, index, LINUX_SECCOMP_EXEC_OPTION, inline_value)?;
            set_once(
                &mut options.sandbox_seccomp_command,
                value,
                LINUX_SECCOMP_EXEC_OPTION,
            )
        }
        "help" => {
            reject_inline_value("--help", inline_value)?;
            set_flag(&mut options.help, "--help")
        }
        "version" => {
            reject_inline_value("--version", inline_value)?;
            set_flag(&mut options.version, "--version")
        }
        _ => Err(ParseError::UnknownArgument(OsString::from(argument))),
    }
}

fn required_value(
    arguments: &[OsString],
    index: &mut usize,
    option: &'static str,
    inline_value: Option<&str>,
) -> Result<OsString, ParseError> {
    let value = if let Some(inline_value) = inline_value {
        OsString::from(inline_value)
    } else {
        let Some(value) = arguments.get(*index) else {
            return Err(ParseError::MissingValue { option });
        };
        if value.to_str().is_some_and(|value| value.starts_with('-')) {
            return Err(ParseError::MissingValue { option });
        }
        *index += 1;
        value.clone()
    };

    if value.is_empty() {
        return Err(ParseError::EmptyValue { option });
    }
    Ok(value)
}

fn required_text_value(
    arguments: &[OsString],
    index: &mut usize,
    option: &'static str,
    inline_value: Option<&str>,
) -> Result<String, ParseError> {
    let value = required_value(arguments, index, option, inline_value)?;
    value
        .into_string()
        .map_err(|value| ParseError::NonUtf8Value { option, value })
}

fn set_positional_prompt(
    options: &mut LaunchOptions,
    argument: &OsString,
) -> Result<(), ParseError> {
    if options.prompt.is_some() {
        return Err(ParseError::UnexpectedPositional(argument.clone()));
    }
    let prompt = argument
        .clone()
        .into_string()
        .map_err(|value| ParseError::NonUtf8Value {
            option: "PROMPT",
            value,
        })?;
    options.prompt = Some(prompt);
    Ok(())
}

fn reject_inline_value(option: &'static str, value: Option<&str>) -> Result<(), ParseError> {
    if value.is_some() {
        Err(ParseError::UnexpectedValue { option })
    } else {
        Ok(())
    }
}

fn append_path_list(
    destination: &mut Vec<PathBuf>,
    value: OsString,
    option: &'static str,
) -> Result<(), ParseError> {
    let Some(value) = value.to_str() else {
        destination.push(PathBuf::from(value));
        return Ok(());
    };

    let mut paths = Vec::new();
    for path in value.split(',') {
        if path.is_empty() {
            return Err(ParseError::EmptyValue { option });
        }
        paths.push(PathBuf::from(path));
    }
    destination.extend(paths);
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &'static str) -> Result<(), ParseError> {
    if slot.is_some() {
        return Err(ParseError::DuplicateOption { option });
    }
    *slot = Some(value);
    Ok(())
}

fn set_flag(flag: &mut bool, option: &'static str) -> Result<(), ParseError> {
    if *flag {
        return Err(ParseError::DuplicateOption { option });
    }
    *flag = true;
    Ok(())
}

fn validate_conflicts(options: &LaunchOptions) -> Result<(), ParseError> {
    if options.help && options.version {
        return conflict("--help", "--version");
    }
    if options.full_auto {
        if options.sandbox_mode.is_some() {
            return conflict("--full-auto", "--sandbox");
        }
        if options.approval_policy.is_some() {
            return conflict("--full-auto", "--ask-for-approval");
        }
        if options.dangerously_bypass_approvals_and_sandbox {
            return conflict("--full-auto", DANGEROUS_BYPASS);
        }
    }
    if options.dangerously_bypass_approvals_and_sandbox {
        if options.sandbox_mode.is_some() {
            return conflict(DANGEROUS_BYPASS, "--sandbox");
        }
        if options.approval_policy.is_some() {
            return conflict(DANGEROUS_BYPASS, "--ask-for-approval");
        }
    }
    Ok(())
}

fn conflict<T>(first: &'static str, second: &'static str) -> Result<T, ParseError> {
    Err(ParseError::ConflictingOptions { first, second })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<LaunchOptions, ParseError> {
        parse_args(arguments.iter().copied())
    }

    #[test]
    fn empty_arguments_use_launch_defaults() {
        assert_eq!(parse(&[]), Ok(LaunchOptions::default()));
    }

    #[test]
    fn parses_every_separated_value_and_repeatable_directory() {
        let options = parse(&[
            "-m",
            "codex:gpt-5",
            "-C",
            "workspace",
            "--add-dir",
            "../shared",
            "--add-dir",
            "../generated",
            "-i",
            "first.png,second.png",
            "--image",
            "third.png",
            "-s",
            "workspace-write",
            "-a",
            "on-request",
            "--no-alt-screen",
            "explain the changes",
        ])
        .expect("valid arguments");

        assert_eq!(options.model.as_deref(), Some("codex:gpt-5"));
        assert_eq!(options.prompt.as_deref(), Some("explain the changes"));
        assert_eq!(options.cwd, Some(PathBuf::from("workspace")));
        assert_eq!(
            options.additional_writable_dirs,
            vec![PathBuf::from("../shared"), PathBuf::from("../generated")]
        );
        assert_eq!(
            options.images,
            vec![
                PathBuf::from("first.png"),
                PathBuf::from("second.png"),
                PathBuf::from("third.png")
            ]
        );
        assert_eq!(options.sandbox_mode, Some(SandboxMode::WorkspaceWrite));
        assert_eq!(options.approval_policy, Some(ApprovalPolicy::OnRequest));
        assert!(options.no_alt_screen);
    }

    #[test]
    fn parses_long_inline_values_without_resolving_paths() {
        let options = parse(&[
            "--model=claude:sonnet",
            "--cd=relative/workspace",
            "--add-dir=../shared=cache",
            "--image=references/one.png,references/two.png",
            "--sandbox=danger-full-access",
            "--ask-for-approval=never",
        ])
        .expect("valid inline arguments");

        assert_eq!(options.model.as_deref(), Some("claude:sonnet"));
        assert_eq!(options.cwd, Some(PathBuf::from("relative/workspace")));
        assert_eq!(
            options.additional_writable_dirs,
            vec![PathBuf::from("../shared=cache")]
        );
        assert_eq!(
            options.images,
            vec![
                PathBuf::from("references/one.png"),
                PathBuf::from("references/two.png")
            ]
        );
        assert_eq!(options.sandbox_mode, Some(SandboxMode::DangerFullAccess));
        assert_eq!(options.approval_policy, Some(ApprovalPolicy::Never));
    }

    #[test]
    fn parses_all_sandbox_values() {
        for (value, expected) in [
            ("read-only", SandboxMode::ReadOnly),
            ("workspace-write", SandboxMode::WorkspaceWrite),
            ("danger-full-access", SandboxMode::DangerFullAccess),
        ] {
            assert_eq!(
                parse(&["--sandbox", value])
                    .expect("known sandbox")
                    .sandbox_mode,
                Some(expected)
            );
            assert_eq!(expected.to_string(), value);
        }
    }

    #[test]
    fn parses_all_approval_values() {
        for (value, expected) in [
            ("on-request", ApprovalPolicy::OnRequest),
            ("never", ApprovalPolicy::Never),
        ] {
            assert_eq!(
                parse(&["--ask-for-approval", value])
                    .expect("known approval policy")
                    .approval_policy,
                Some(expected)
            );
            assert_eq!(expected.to_string(), value);
        }
    }

    #[test]
    fn images_accumulate_across_short_long_and_comma_delimited_forms() {
        let options = parse(&[
            "-i",
            "one.png,two.png",
            "--image=three.png",
            "--image",
            "four.png,five.png",
        ])
        .expect("valid images");

        assert_eq!(
            options.images,
            ["one.png", "two.png", "three.png", "four.png", "five.png"].map(PathBuf::from)
        );
    }

    #[test]
    fn shorthand_flags_remain_raw_for_startup_resolution() {
        assert!(parse(&["--full-auto"]).expect("full auto").full_auto);
        assert!(
            parse(&[DANGEROUS_BYPASS])
                .expect("dangerous bypass")
                .dangerously_bypass_approvals_and_sandbox
        );
    }

    #[test]
    fn internal_seccomp_stage_preserves_the_exact_shell_command() {
        let command = "printf '%s\\n' -- --value=two";
        let argument = format!("{LINUX_SECCOMP_EXEC_OPTION}={command}");
        let options = parse_args([argument]).expect("internal child stage");
        assert_eq!(
            options.sandbox_seccomp_command,
            Some(OsString::from(command))
        );
    }

    #[test]
    fn help_and_version_accept_short_and_long_forms() {
        assert!(parse(&["-h"]).expect("short help").help);
        assert!(parse(&["--help"]).expect("long help").help);
        assert!(parse(&["-V"]).expect("short version").version);
        assert!(parse(&["--version"]).expect("long version").version);
    }

    #[test]
    fn help_can_describe_an_otherwise_valid_invocation() {
        let options = parse(&["--help", "--sandbox", "read-only"]).expect("valid help");
        assert!(options.help);
        assert_eq!(options.sandbox_mode, Some(SandboxMode::ReadOnly));
    }

    #[test]
    fn rejects_unknown_arguments_and_short_option_bundles() {
        assert_eq!(
            parse(&["--yolo"]),
            Err(ParseError::UnknownArgument(OsString::from("--yolo")))
        );
        assert_eq!(
            parse(&["-hV"]),
            Err(ParseError::UnknownArgument(OsString::from("-hV")))
        );
    }

    #[test]
    fn accepts_one_positional_prompt_including_after_end_of_options() {
        assert_eq!(
            parse(&["prompt"]).expect("prompt").prompt.as_deref(),
            Some("prompt")
        );
        assert_eq!(
            parse(&["--", "--help"])
                .expect("hyphenated prompt")
                .prompt
                .as_deref(),
            Some("--help")
        );
        assert_eq!(parse(&["--"]), Ok(LaunchOptions::default()));
    }

    #[test]
    fn rejects_more_than_one_positional_prompt() {
        assert_eq!(
            parse(&["first", "second"]),
            Err(ParseError::UnexpectedPositional(OsString::from("second")))
        );
        assert_eq!(
            parse(&["first", "--", "second"]),
            Err(ParseError::UnexpectedPositional(OsString::from("second")))
        );
    }

    #[test]
    fn rejects_missing_and_empty_values() {
        assert_eq!(
            parse(&["--cd"]),
            Err(ParseError::MissingValue { option: "--cd" })
        );
        assert_eq!(
            parse(&["--cd", "--help"]),
            Err(ParseError::MissingValue { option: "--cd" })
        );
        assert_eq!(
            parse(&["--add-dir="]),
            Err(ParseError::EmptyValue {
                option: "--add-dir"
            })
        );
    }

    #[test]
    fn rejects_invalid_typed_values() {
        assert_eq!(
            parse(&["--sandbox", "container"]),
            Err(ParseError::InvalidValue {
                option: "--sandbox",
                value: OsString::from("container"),
                expected: "read-only, workspace-write, danger-full-access",
            })
        );
        assert_eq!(
            parse(&["--ask-for-approval=untrusted"]),
            Err(ParseError::InvalidValue {
                option: "--ask-for-approval",
                value: OsString::from("untrusted"),
                expected: "on-request, never",
            })
        );
    }

    #[test]
    fn rejects_empty_image_segments() {
        for arguments in [
            vec!["--image=,one.png"],
            vec!["--image=one.png,"],
            vec!["-i", "one.png,,two.png"],
        ] {
            assert_eq!(
                parse(&arguments),
                Err(ParseError::EmptyValue { option: "--image" })
            );
        }
    }

    #[test]
    fn rejects_values_on_boolean_flags() {
        for (argument, option) in [
            ("--no-alt-screen=true", "--no-alt-screen"),
            ("--full-auto=false", "--full-auto"),
            ("--help=true", "--help"),
        ] {
            assert_eq!(
                parse(&[argument]),
                Err(ParseError::UnexpectedValue { option })
            );
        }
    }

    #[test]
    fn rejects_duplicate_singleton_options_across_aliases() {
        assert_eq!(
            parse(&["-C", "one", "--cd=two"]),
            Err(ParseError::DuplicateOption { option: "--cd" })
        );
        assert_eq!(
            parse(&["-m", "one", "--model=two"]),
            Err(ParseError::DuplicateOption { option: "--model" })
        );
        assert_eq!(
            parse(&["-s", "read-only", "--sandbox=workspace-write"]),
            Err(ParseError::DuplicateOption {
                option: "--sandbox"
            })
        );
        assert_eq!(
            parse(&["-h", "--help"]),
            Err(ParseError::DuplicateOption { option: "--help" })
        );
    }

    #[test]
    fn repeatable_directories_allow_repeated_values() {
        let options =
            parse(&["--add-dir", "shared", "--add-dir=shared"]).expect("add-dir is repeatable");
        assert_eq!(
            options.additional_writable_dirs,
            vec![PathBuf::from("shared"), PathBuf::from("shared")]
        );
    }

    #[test]
    fn full_auto_conflicts_with_explicit_or_more_permissive_policy() {
        for (arguments, second) in [
            (
                vec!["--full-auto", "--sandbox", "workspace-write"],
                "--sandbox",
            ),
            (
                vec!["--ask-for-approval", "on-request", "--full-auto"],
                "--ask-for-approval",
            ),
            (vec![DANGEROUS_BYPASS, "--full-auto"], DANGEROUS_BYPASS),
        ] {
            assert_eq!(
                parse(&arguments),
                Err(ParseError::ConflictingOptions {
                    first: "--full-auto",
                    second,
                })
            );
        }
    }

    #[test]
    fn dangerous_bypass_conflicts_with_explicit_policy() {
        assert_eq!(
            parse(&[DANGEROUS_BYPASS, "--sandbox", "danger-full-access"]),
            Err(ParseError::ConflictingOptions {
                first: DANGEROUS_BYPASS,
                second: "--sandbox",
            })
        );
        assert_eq!(
            parse(&["--ask-for-approval=never", DANGEROUS_BYPASS]),
            Err(ParseError::ConflictingOptions {
                first: DANGEROUS_BYPASS,
                second: "--ask-for-approval",
            })
        );
    }

    #[test]
    fn help_and_version_are_mutually_exclusive() {
        assert_eq!(
            parse(&["--version", "--help"]),
            Err(ParseError::ConflictingOptions {
                first: "--help",
                second: "--version",
            })
        );
    }

    #[test]
    fn parse_errors_are_actionable() {
        assert_eq!(
            parse(&["--sandbox=container"])
                .expect_err("invalid sandbox")
                .to_string(),
            "invalid value `container` for `--sandbox`; expected one of: read-only, workspace-write, danger-full-access"
        );
    }

    #[cfg(unix)]
    #[test]
    fn separated_paths_preserve_non_utf8_bytes() {
        use std::os::unix::ffi::OsStringExt;

        let path = OsString::from_vec(vec![b'w', b'o', b'r', b'k', 0xff]);
        let options =
            parse_args([OsString::from("--cd"), path.clone()]).expect("non-UTF-8 separated path");
        assert_eq!(options.cwd, Some(PathBuf::from(path)));
    }
}
