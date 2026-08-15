use image::imageops::{self, FilterType};
use image::RgbaImage;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::PathBuf;

const SOURCE: &str = "assets/mascot/shaltaiboltai-humpty-sprites.png";
const LOGICAL_WIDTH: u32 = 26;
const LOGICAL_HEIGHT: u32 = 34;
const NORMALIZED_WIDTH: u32 = 600;
const ALPHA_THRESHOLD: u8 = 80;
const MIN_POSE_PIXELS: usize = 10_000;
const EXPECTED_POSES: usize = 4;

#[derive(Debug)]
struct Component {
    pixels: Vec<(u32, u32)>,
    min_x: u32,
    max_x: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={SOURCE}");

    let sheet = image::open(SOURCE)?.to_rgba8();
    let frames = isolate_poses(&sheet)?;

    let mut generated = String::new();
    writeln!(
        generated,
        "pub const FRAME_WIDTH: usize = {LOGICAL_WIDTH};\npub const FRAME_HEIGHT: usize = {};",
        LOGICAL_HEIGHT / 2
    )?;
    writeln!(
        generated,
        "#[derive(Debug, PartialEq, Eq)]\npub struct MascotFrame {{\n    pub cells: [[MascotCell; FRAME_WIDTH]; FRAME_HEIGHT],\n}}"
    )?;
    writeln!(
        generated,
        "pub static GENERATED_FRAMES: [MascotFrame; {}] = [",
        frames.len()
    )?;
    for frame in frames {
        writeln!(generated, "    MascotFrame {{ cells: [")?;
        for row in 0..(LOGICAL_HEIGHT / 2) {
            write!(generated, "        [")?;
            for column in 0..LOGICAL_WIDTH {
                let top = packed_pixel(frame.get_pixel(column, row * 2).0);
                let bottom = packed_pixel(frame.get_pixel(column, row * 2 + 1).0);
                write!(generated, "MascotCell::new({top:#010x}, {bottom:#010x}),")?;
            }
            writeln!(generated, "],")?;
        }
        writeln!(generated, "    ] }},")?;
    }
    writeln!(generated, "];")?;

    let output = PathBuf::from(std::env::var("OUT_DIR")?).join("mascot_frames.rs");
    fs::write(output, generated)?;
    Ok(())
}

fn isolate_poses(sheet: &RgbaImage) -> Result<Vec<RgbaImage>, Box<dyn Error>> {
    let width = sheet.width();
    let height = sheet.height();
    let mut visited = vec![false; (width * height) as usize];
    let mut components = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if visited[index] {
                continue;
            }
            visited[index] = true;
            if sheet.get_pixel(x, y).0[3] < ALPHA_THRESHOLD {
                continue;
            }

            let mut queue = VecDeque::from([(x, y)]);
            let mut pixels = Vec::new();
            let mut min_x = x;
            let mut max_x = x;
            while let Some((current_x, current_y)) = queue.pop_front() {
                pixels.push((current_x, current_y));
                min_x = min_x.min(current_x);
                max_x = max_x.max(current_x);

                let start_x = current_x.saturating_sub(1);
                let end_x = (current_x + 1).min(width - 1);
                let start_y = current_y.saturating_sub(1);
                let end_y = (current_y + 1).min(height - 1);
                for neighbor_y in start_y..=end_y {
                    for neighbor_x in start_x..=end_x {
                        let neighbor = (neighbor_y * width + neighbor_x) as usize;
                        if visited[neighbor] {
                            continue;
                        }
                        visited[neighbor] = true;
                        if sheet.get_pixel(neighbor_x, neighbor_y).0[3] >= ALPHA_THRESHOLD {
                            queue.push_back((neighbor_x, neighbor_y));
                        }
                    }
                }
            }

            if pixels.len() >= MIN_POSE_PIXELS {
                components.push(Component {
                    pixels,
                    min_x,
                    max_x,
                });
            }
        }
    }

    components.sort_by_key(|component| component.min_x);
    if components.len() != EXPECTED_POSES {
        return Err(invalid_data(format!(
            "expected {EXPECTED_POSES} isolated mascot poses, found {}",
            components.len()
        ))
        .into());
    }

    components
        .into_iter()
        .map(|component| {
            let component_width = component.max_x - component.min_x + 1;
            if component_width > NORMALIZED_WIDTH {
                return Err(invalid_data(format!(
                    "mascot pose is {component_width}px wide; normalized canvas is {NORMALIZED_WIDTH}px"
                )));
            }

            // Center every connected pose on one fixed-width canvas while
            // keeping its authored absolute y coordinate. This preserves the
            // common standing baseline and the airborne hop without allowing
            // neighboring poses to bleed into a frame.
            let component_center = (component.min_x + component.max_x) as i64 / 2;
            let offset_x = NORMALIZED_WIDTH as i64 / 2 - component_center;
            let mut normalized = RgbaImage::new(NORMALIZED_WIDTH, height);
            for (source_x, source_y) in component.pixels {
                let target_x = source_x as i64 + offset_x;
                if !(0..NORMALIZED_WIDTH as i64).contains(&target_x) {
                    return Err(invalid_data("mascot pose escaped its normalized canvas"));
                }
                normalized.put_pixel(
                    target_x as u32,
                    source_y,
                    *sheet.get_pixel(source_x, source_y),
                );
            }
            Ok(imageops::resize(
                &normalized,
                LOGICAL_WIDTH,
                LOGICAL_HEIGHT,
                FilterType::Triangle,
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn packed_pixel(pixel: [u8; 4]) -> u32 {
    if pixel[3] < ALPHA_THRESHOLD {
        0
    } else {
        0x0100_0000 | (u32::from(pixel[0]) << 16) | (u32::from(pixel[1]) << 8) | u32::from(pixel[2])
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
