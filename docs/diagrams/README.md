# Diagram sources

Each `*.svg` in this folder is rendered from the matching `*.mmd`
([Mermaid](https://mermaid.js.org/)) source, and embedded in the design docs
under [`docs/`](..).

## Regenerate

With [`@mermaid-js/mermaid-cli`](https://github.com/mermaid-js/mermaid-cli)
installed (`npm install -g @mermaid-js/mermaid-cli`), from this folder:

```bash
for f in *.mmd; do mmdc -i "$f" -o "${f%.mmd}.svg"; done
```

Edit the `.mmd` source, re-render, and commit both files together.
