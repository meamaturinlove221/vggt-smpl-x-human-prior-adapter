# V13300 Anti-Billboard Metric v2

Metric v2 combines thickness, cross-section occupancy, front/back separation, depth-layer entropy, connected-component continuity, and optional body-part component balance. It explicitly rejects thickness-only success.

Required interpretation:

- high thickness alone is not pass;
- multi-layer occupancy and front/back separation must be present;
- largest connected component and part continuity must remain strong;
- same-topology/shuffled/thickness-only controls must not dominate true.
