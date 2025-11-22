module BayesianInference

using PythonCall
using CondaPkg

include("utils.jl")
using .Utils
export create_env, delete_env, install_package, update_package

export importBI

"""
    importBI(; platform="cpu", cores=nothing, rand_seed=true, deallocate=false, print_devices_found=true, backend="numpyro")

Initializes and returns the Python `BI` object from the `BayesInference` package.
"""
function importBI(;
    platform="cpu",
    cores=nothing,
    rand_seed=true,
    deallocate=false,
    print_devices_found=true,
    backend="numpyro"
)
    # Import BI module
    bi_module = pyimport("BI")

    # Access the BI class from the module
    # This is equivalent to: from BI import bi
    BI_class = bi_module.bi

    # Instantiate and return the BI object
    return BI_class(
        platform=platform,
        cores=cores,
        rand_seed=rand_seed,
        deallocate=deallocate,
        print_devices_found=print_devices_found,
        backend=backend
    )
end

# JAX/Array Interoperability
# Teach julia to use jax array basic operations

Base.:*(a::Py, b::PyArray) = a * Py(b)
Base.:*(a::PyArray, b::Py) = Py(a) * b
Base.:+(a::Py, b::PyArray) = a + Py(b)
Base.:+(a::PyArray, b::Py) = Py(a) + b
Base.:-(a::Py, b::PyArray) = a - Py(b)
Base.:-(a::PyArray, b::Py) = Py(a) - b
Base.:/(a::Py, b::PyArray) = a / Py(b)
Base.:/(a::PyArray, b::Py) = Py(a) / b
Base.://(a::Py, b::PyArray) = a // Py(b)
Base.://(a::PyArray, b::Py) = Py(a) // b
Base.:%(a::Py, b::PyArray) = a % Py(b)
Base.:%(a::PyArray, b::Py) = Py(a) % b
Base.:^(a::Py, b::PyArray) = a^Py(b)
Base.:^(a::PyArray, b::Py) = Py(a)^b

end
