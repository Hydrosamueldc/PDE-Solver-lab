# Interpretation

## Physical Meaning of the Heat Equation

The heat equation models diffusion. If one region is hotter than nearby regions,
heat flows away from the hot region and into the cooler regions.

```text
u_t = alpha u_xx
```

The term `u_t` measures how temperature changes over time. The term `u_xx`
measures spatial curvature. Large curvature means the temperature profile bends
strongly, so diffusion changes it quickly.

## Why Heat Smooths

Heat naturally moves from high temperature regions to low temperature regions.
Sharp peaks flatten, steep jumps soften, and the temperature profile becomes
smoother over time.

In the scripts, a hot center region begins with a sharp shape. As time advances,
the center cools and neighboring points warm up.

## Explicit Method Idea

FTCS is an explicit method because the next value is computed directly from
known values at the current time level:

```text
future = weighted combination of known current values
```

This makes FTCS easy to understand and easy to program.

## Why Instability Happens

The simplicity of FTCS comes with a restriction. If the time step is too large
relative to the space step, the method tries to move too much heat in one update.

For the 1D heat equation, the stability condition is:

```text
r = alpha*k/h^2 <= 0.5
```

When `r > 0.5`, artificial numerical oscillations can appear. These oscillations
may grow rapidly, producing a result that is not physically meaningful.
