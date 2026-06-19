# Python syntax bootcamp — running list

Nightly parallel track (see CLAUDE.md). Pure syntax gaps surfaced during daytime drills land here; don't re-derive concepts, just drill the syntax until it's automatic.

## 2026-06-19 — surfaced during `drills/backprop_from_scratch.py`

Recurring pattern: the logic/algorithm was correct, but written in JavaScript syntax instead of Python. Practice tonight:

- [ ] `for` loops: `for i in range(n):` — no `let`/`const`, no `i++`, colon required, no parens around condition.
- [ ] No braces — blocks are defined by indentation (4 spaces), not `{ }`.
- [ ] No semicolons at end of lines.
- [ ] Function calls: `forward(X, W1, b1, W2, b2)` directly — no `this.` prefix (no class involved).
- [ ] Tuple unpacking: `a, b, c = some_func_returning_tuple()`.
- [ ] Ternary expression: `value_if_true if condition else value_if_false` (not `condition ? a : b`).
- [ ] `if`/`for`/`while`/`def` all need a trailing `:`.
- [ ] Case sensitivity: `.T` (transpose) vs `.t` — Python/numpy attributes are case-sensitive.
- [ ] `np.random.randn(shape...)` for random init, `np.zeros((shape_tuple))` for zeros — note the double parens on `zeros`.

Try rewriting the `train()` loop and test script from memory, without looking at the working file, then diff against it.
