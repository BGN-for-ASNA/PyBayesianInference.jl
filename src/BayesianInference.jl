module BayesianInference

using PythonCall
using CondaPkg

# Check if utils.jl exists in your folder, otherwise comment this out
if isfile(joinpath(@__DIR__, "utils.jl"))
    include("utils.jl")
    using .Utils
    export create_env, delete_env, install_package, update_package
end

export importBI, @BI, @pyplot, jnp, jax, pybuiltins, pydict, pylist


# -------------------------------------------------------
# Teach PythonCall to convert Julia ':' to Python slice(None)
# This allows syntax like array[:, 0] to work on Py objects
# -------------------------------------------------------
PythonCall.Py(::Colon) = pybuiltins.slice(nothing)

# -------------------------------------------------------
# Import BI Module
# -------------------------------------------------------

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

# -------------------------------------------------------
# JAX/Array Interoperability
# Teach julia to use jax array basic operations
# -------------------------------------------------------

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


# -------------------------------------------------------
# Function Wrapper & Macro
# -------------------------------------------------------

# 1. The Wrapper Struct
struct InspectableFunction <: Function
    f::Function
end

# 2. Make it callable in Julia (so it behaves like a normal function)
(obj::InspectableFunction)(args...; kwargs...) = obj.f(args...; kwargs...)

# 3. Teach PythonCall how to convert it
#    This is called automatically when you pass the object to Python
function PythonCall.Py(obj::InspectableFunction)
    meth = first(methods(obj.f))
    
    # Introspection: Get arg names (skip #self#)
    # Note: This only reliably works for POSITIONAL arguments.
    arg_names = [String(a) for a in Base.method_argnames(meth)[2:end]]
    args_str = join(arg_names, ", ")

    # Create the lambda factory code
    lambda_code = "lambda fn: lambda $args_str: fn($args_str)"

    # FIX: Provide explicit globals/locals dictionaries to avoid SystemError
    # We use pybuiltins.dict() to create empty Python dicts
    globals_dict = pybuiltins.dict()
    locals_dict = pybuiltins.dict()
    
    # Eval the code in this isolated context
    factory = pybuiltins.eval(lambda_code, globals_dict, locals_dict)
    
    return factory(obj.f)
end

# 4. The Macro for Python Models
macro BI(ex)
    # Check validity
    if !isa(ex, Expr) || (ex.head != :function && ex.head != :(=))
        error("@BI must be used on a function definition")
    end
    
    call_expr = ex.args[1]
    if !isa(call_expr, Expr) || call_expr.head != :call
         error("Invalid function signature")
    end

    # 1. Capture the user's name (e.g., :model) from the ORIGINAL expression
    user_func_name = call_expr.args[1]

    # 2. Create a unique hidden name (e.g., ##model#291)
    internal_name = gensym(user_func_name)

    # 3. DEEPCOPY the expression before modifying it.
    #    Using `copy()` is shallow and corrupts the input AST, causing the crash.
    internal_def = deepcopy(ex)
    
    # 4. Rename the function in the copy
    internal_def.args[1].args[1] = internal_name

    quote
        # Define the actual logic via the hidden name (escaped to exist in user scope)
        $(esc(internal_def))

        # Assign the Wrapper to the User's name
        # InspectableFunction is NOT escaped, so it refers to BayesianInference.InspectableFunction
        $(esc(user_func_name)) = InspectableFunction($(esc(internal_name)))
    end
end

# -------------------------------------------------------
# Plotting Macro (@pyplot)
# -------------------------------------------------------
"""
    @pyplot expression

Executes a Python plotting command, intercepts plt.show()/plt.close(),
captures the figure, and displays it in Julia.
Uses a native Python lambda for the hook to avoid __signature__ errors.
"""
macro pyplot(ex)
    return quote
        local plt = PythonCall.pyimport("matplotlib.pyplot")
        local pybuiltins = PythonCall.pybuiltins
        
        # Save original functions
        local real_show = plt.show
        local real_close = plt.close
        
        # Define a Python No-Op lambda (Safe for introspection)
        # We pass empty dicts to avoid "SystemError: frame does not exist"
        local g = pybuiltins.dict()
        local l = pybuiltins.dict()
        local noop = pybuiltins.eval("lambda *args, **kwargs: None", g, l)

        # Inject No-Ops to prevent the figure from being closed/hidden
        plt.show = noop
        plt.close = noop

        local fig = nothing
        try
            # Run the user's plotting code
            $(esc(ex))
            
            # Grab the current figure (it should still be alive)
            fig = plt.gcf()
            
            # Check if it has content
            if length(fig.axes) > 0
                display(fig)
            else
                println("⚠️ No plot was generated (Figure is empty).")
            end
        catch e
            rethrow(e)
        finally
            # Restore original functions
            plt.show = real_show
            plt.close = real_close
            
            # Now we can safely close the figure to free memory
            if fig !== nothing
                real_close(fig)
            end
        end
    end
end

# Global Constants for jax.numpy
# 1. Define a global constant for jnp
const jnp = PythonCall.pynew()
const jax = PythonCall.pynew()

# 2. Initialize it when the module loads
function __init__()
    PythonCall.pycopy!(jnp, pyimport("jax.numpy"))
    PythonCall.pycopy!(jax, pyimport("jax"))
end


# -------------------------------------------------------
# JAX/Array/Iterable Interoperability
# -------------------------------------------------------
# Teach Julia to pass math operations on PyArray/PyIterable back to Python.
# This fixes MethodError when doing math on arguments passed as lists/iterables.

# Define the types we want to unwrap automatically
const PyWrappers = Union{PythonCall.PyArray, PythonCall.PyIterable}

for op in (:+, :-, :*, :/, ://, :%, :^)
    @eval begin
        # Case 1: Py <op> Wrapper (e.g., b * monastery)
        Base.$op(a::Py, b::PyWrappers) = $op(a, Py(b))
        
        # Case 2: Wrapper <op> Py (e.g., monastery * b)
        Base.$op(a::PyWrappers, b::Py) = $op(Py(a), b)
        
        # Case 3: Wrapper <op> Wrapper (e.g., array + array)
        Base.$op(a::PyWrappers, b::PyWrappers) = $op(Py(a), Py(b))
    end
end

# -------------------------------------------------------
# Allow access to Python attributes (like .shape) on wrapped arrays.
# If the property isn't a field of the Julia struct, fetch it from Python.
# -------------------------------------------------------
function Base.getproperty(x::PyWrappers, s::Symbol)
    if hasfield(typeof(x), s)
        return getfield(x, s)
    else
        return getproperty(Py(x), s)
    end
end



# 2. Fix Indexing (getindex/setindex!)
# This allows accessing income[0] even if income is wrapped as an Iterable
Base.getindex(a::PyWrappers, i::Any) = getindex(Py(a), i)
Base.setindex!(a::PyWrappers, v::Any, i::Any) = setindex!(Py(a), v, i)


end