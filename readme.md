## revolve.py
A simple script to create .stl models of solids of revolution.

----

## Examples

You can run without arguments and will get prompted interactively:

`> python3 revolve.py`
```
output file name (will append .stl): test
type (shells, smooth): shells
axis (x, y, z): y
radius start (example "0"): 0.5
radius end (example "sqrt(10)"): sqrt(10)
height as a function of radius (r) (example "10-r**2"): 10-r**2
shell count (example "8"): 8
circle detail (example "32"): 32
gap size (example "0.1"): 0.1
```

You can also pass arguments directly:

`> python3 revolve.py paraboloid smooth z 0 10 "sqrt(10-y)" 32 32`

`> python3 revolve.py paraboloid shells z 0 "sqrt(10)" "10-r**2" 8 32 0.1`

----

## Using math
You can use mathematical expressions but you may have to put it in quotes when passing as arguments directly. This script supports all the [python math library](https://docs.python.org/3/library/math.html) functions like `pow`, `exp`, `log`, `sqrt`, `sin`, `cos`, etc, and constants like `pi` and `e`. As well as some other built in functions like, `abs`, `min`, and `max`.

