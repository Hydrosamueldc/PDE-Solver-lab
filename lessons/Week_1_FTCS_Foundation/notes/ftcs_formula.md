# FTCS Formula

## Heat Equation

The one-dimensional heat equation is

```text
u_t = alpha u_xx
```

It says that the rate of temperature change in time depends on the curvature of
the temperature profile in space.

## Discretization

At a grid point `x_i` and time level `t_j`, write the numerical approximation as

```text
U_i^j approximates u(x_i, t_j)
```

Use a forward difference in time:

```text
u_t approx (U_i^(j+1) - U_i^j) / k
```

Use a centered difference in space:

```text
u_xx approx (U_(i+1)^j - 2U_i^j + U_(i-1)^j) / h^2
```

Substitute these into the heat equation:

```text
(U_i^(j+1) - U_i^j) / k
= alpha (U_(i+1)^j - 2U_i^j + U_(i-1)^j) / h^2
```

## Mesh Ratio

Define

```text
r = alpha*k/h^2
```

Here:

- `alpha` is the thermal diffusivity.
- `k` is the time step size.
- `h` is the space step size.
- `r` measures how strongly neighboring points affect the next value.

## Final FTCS Scheme

```text
U_i^(j+1) = r U_(i-1)^j + (1 - 2r) U_i^j + r U_(i+1)^j
```

The future value is a weighted combination of the left neighbor, center value,
and right neighbor from the previous time level.

## Stability Condition

For the 1D heat equation, the FTCS method is stable when

```text
r <= 0.5
```

If `r > 0.5`, numerical oscillations can grow and the computed solution may no
longer represent physical heat diffusion.
