# Julia 

Jullia is a general-prupose functional programming languade for Datascience, it balanced the performance of c++ and the ease-of-use of Python. 

## Remark

`pso.jl`: the main file of the folder, it contains the implementation of the PSO algorithm, in less than 20 lines!!!

## Install Packages 

To install the `Optim` package, run the following command in the Julia REPL:

```julia
Using Pkg 
Pkg.add("Optim)
```

## Run the code 

To run the code from bash execute the following command:

```bash
$ julia pso.jl
```

## Example Output 

```
 * Status: failure (reached maximum number of iterations)

 * Candidate solution
    Final objective value:     -1.913223e+00

 * Found with
    Algorithm:     Particle Swarm

 * Convergence measures
    |x - x'|               = NaN ≰ 0.0e+00
    |x - x'|/|x'|          = NaN ≰ 0.0e+00
    |f(x) - f(x')|         = NaN ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = NaN ≰ 0.0e+00
    |g(x)|                 = NaN ≰ 1.0e-08

 * Work counters
    Seconds run:   1  (vs limit Inf)
    Iterations:    1000
    f(x) calls:    101100
    ∇f(x) calls:   0

  0.057675 seconds (232.11 k allocations: 86.916 MiB, 26.94% gc time)
 * Status: failure (reached maximum number of iterations)

 * Candidate solution
    Final objective value:     -8.402283e+05

 * Found with
    Algorithm:     Particle Swarm

 * Convergence measures
    |x - x'|               = NaN ≰ 0.0e+00
    |x - x'|/|x'|          = NaN ≰ 0.0e+00
    |f(x) - f(x')|         = NaN ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = NaN ≰ 0.0e+00
    |g(x)|                 = NaN ≰ 1.0e-08

 * Work counters
    Seconds run:   3  (vs limit Inf)
    Iterations:    1000
    f(x) calls:    1002000
    ∇f(x) calls:   0

  2.197762 seconds (2.53 M allocations: 7.556 GiB, 4.39% gc time, 0.37% compilation time)
```