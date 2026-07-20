`trace.jsonl` is stored as three line-aligned parts because the GitHub upload
connector has a 1 MiB read ceiling. Concatenate the parts in numeric order to
reconstruct the original file.

Original SHA-256:

`eceb656381a95c5becbcc279dd8beff8f6b99a385d8ec7a296dadd9278f9a178`

The three parts contain 36, 36, and 1 JSONL records, respectively.
