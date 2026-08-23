# Shaltaiboltai mascot reference

`shaltaiboltai-humpty-sprites.png` is the canonical four-pose source for the
lead character: the classic Humpty Dumpty egg gentleman with his face directly
on the shell, a cravat, navy coat, red waistcoat, teal breeches, and boots.

The image was generated specifically for this project as a four-pose pixel-art
sheet. At build time, `build.rs` identifies each connected pose and centers it
on one fixed canvas while retaining the shared baseline. The normalized poses
are embedded in the binary for high-resolution Kitty graphics; a second
source-color sample becomes the portable terminal-cell fallback. No sidecar
image is needed at runtime, and smaller terminals do not substitute a
separately drawn icon.
