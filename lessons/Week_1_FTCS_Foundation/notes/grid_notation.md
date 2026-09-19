# Grid Notation

## Space Grid

The spatial domain is divided into equally spaced points:

```text
x_0, x_1, x_2, ..., x_N
```

The distance between neighboring spatial points is

```text
h = x_(i+1) - x_i
```

## Time Grid

Time is divided into levels:

```text
t_0, t_1, t_2, ..., t_M
```

The distance between neighboring time levels is

```text
k = t_(j+1) - t_j
```

## Meaning of U_i^j

The symbol

```text
U_i^j
```

means the numerical approximation to the temperature at space index `i` and time
level `j`.

In words:

```text
U_i^j approximates u(x_i, t_j)
```

## Space vs Time Indexing

- The lower index `i` tells where we are in space.
- The upper index `j` tells when we are in time.
- Moving right changes the space index.
- Moving upward in a space-time mesh changes the time index.

## Computational Mesh

A computational mesh is the collection of all points `(x_i, t_j)` where the
solution is approximated. FTCS uses values along one time level to predict the
next time level.

For one interior point, FTCS uses:

```text
U_(i-1)^j, U_i^j, U_(i+1)^j
```

to compute:

```text
U_i^(j+1)
```
