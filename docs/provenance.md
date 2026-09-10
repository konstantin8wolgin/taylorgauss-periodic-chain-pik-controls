# Provenance

The controlling implementation was audited read-only in a separate worktree of [`konstantin8wolgin/taylorgauss-research`](https://github.com/konstantin8wolgin/taylorgauss-research). Its Git base was commit `4b2502f9cbd46f61d9b4b41cd1402ae2d072443f` (`docs: freeze completed sampler source`).

The six authority modules, four focused test modules, three reports, and three compact receipts were untracked additions in that worktree. They therefore have no source Git blob or commit attribution. [`source-lineage.json`](../evidence/source-lineage.json) pins every audited file by SHA-256 and explicitly records this status. The six implementation modules in this repository are byte-for-byte copies of those audited authorities; thin public facades and standalone tests are new here.

The original receipts referenced private absolute machine paths and external local-state archives. Those files and paths are not reproduced. [`validated-results.json`](../evidence/validated-results.json) contains the concise public scientific results and the SHA-256 hashes of the original receipts, which preserves byte-level lineage without publishing local data.

Audit replay used Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, PyTorch 2.11.0+cu130, and pytest 9.0.2. With bytecode and pytest caches disabled, all 57 scoped source tests passed. The standalone release receipt records the fresh package verification independently.
