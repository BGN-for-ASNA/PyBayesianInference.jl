# Quick Start Guide - BayesianInference.jl

## Using in Jupyter Notebook

1. **Open the notebook**: Open `test/usage_example.ipynb`

2. **Run the first cell** to activate the environment:
   ```julia
   using Pkg
   Pkg.activate(joinpath(@__DIR__, ".."))
   Pkg.instantiate()
   
   using BayesianInference
   using PythonCall
   ```

3. **Follow the notebook cells** - each cell is documented and ready to run sequentially.

## Using in Julia REPL

1. **Navigate to the package directory**:
   ```bash
   cd c:/Users/sebastian_sosa/Nextcloud/BIJ2
   ```

2. **Start Julia with the project**:
   ```bash
   julia --project=.
   ```

3. **Load the package**:
   ```julia
   using BayesianInference
   using PythonCall
   ```

4. **Use the package**:
   ```julia
   # Initialize BI
   m = BayesianInference.importBI()
   
   # Generate some data
   x = m.dist.normal(0, 1, shape=(100,), sample=true)
   y = m.dist.normal(0.2 + 0.6 * x, 1.2, sample=true)
   
   # Define and fit a model
   function model(; x, y)
       alpha = m.dist.normal(loc=0, scale=1, name="alpha")
       beta  = m.dist.normal(loc=0, scale=1, name="beta")
       sigma = m.dist.exponential(1, name="sigma")
       mu = alpha + beta * x
       m.dist.normal(mu, sigma, obs=y)
   end
   
   m.fit(model, num_warmup=1000, num_samples=1000, num_chains=1)
   m.summary()
   ```

## Important Notes

- **Always activate the project**: Use `julia --project=.` or `Pkg.activate(".")` to ensure the package and its dependencies are available.

- **First time usage**: The first time you use the package, CondaPkg will automatically create a Python virtual environment and install `BayesInference` and `ipython`. This may take a few minutes.

- **Import both packages**: Always import both `BayesianInference` and `PythonCall` when using the package, as `PythonCall` provides the `pyhasattr` and other utilities you might need.

## Troubleshooting

If you see `Package BayesianInference not found in current path`:
- Make sure you're in the package directory
- Run `julia --project=.` instead of just `julia`
- Or activate the project with `Pkg.activate("c:/Users/sebastian_sosa/Nextcloud/BIJ2")`

If Python packages are not installed:
- Run `julia setup_env.jl` to manually trigger the environment setup
