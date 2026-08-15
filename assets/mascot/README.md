# Shaltaiboltai mascot reference

`shaltaiboltai-reference.png` is the canonical visual reference for the lead
agent: warm ivory egg shell, dark teal visor, cyan expression and chest core,
orange scarf, dark limbs, and purple-blue boots.

The image was generated specifically for this project as a four-pose pixel-art
dance sheet. The runtime TUI does not decode or ship the PNG beside the binary;
`src/mascot.rs` translates its identity into fixed-width terminal cells and
Ratatui theme colors so it remains portable over SSH and ordinary terminals.
