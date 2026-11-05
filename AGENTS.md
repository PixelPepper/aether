# Notes for chapter agents

- Project root: `/home/user/aether`. Read `CONTRACT.md` first; it is the interface.
- You own exactly the files named in your brief. Others are being written concurrently by other agents.
- Do not run `git` (the lead commits). Do not `npm install` anything.
- `src/scenes/genesis.js` currently contains a tiny placeholder that shows working r186 idioms
  (`instancedArray`, `Fn().compute(n)`, `renderer.computeAsync`, `SpriteNodeMaterial` with `positionNode = buf.toAttribute()`).
  Your own chapter file currently holds a copy of that placeholder: overwrite it.
- Capture: `NODE_PATH=/opt/node-tools/node_modules node tools/shot.mjs <id> ...` (see CONTRACT.md). Read the PNG to check it.
- When done, reply with: what you built (techniques, counts per tier), GUI controls, pointer interaction, known limitations,
  and the paths of your best screenshots.
