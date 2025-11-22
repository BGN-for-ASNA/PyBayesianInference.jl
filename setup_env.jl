using Pkg
Pkg.activate(@__DIR__)

using CondaPkg

println("Adding BayesInference and ipython to CondaPkg...")
CondaPkg.add_pip("BayesInference")
CondaPkg.add_pip("ipython")

println("\nResolving environment...")
CondaPkg.resolve()

println("\nEnvironment status:")
CondaPkg.status()

println("\nDone! The Python environment is ready.")
