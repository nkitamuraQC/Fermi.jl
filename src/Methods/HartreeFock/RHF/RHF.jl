using Fermi.DIIS

using TensorOperations
using LinearAlgebra
using Formatting
using Fermi.PhysicalConstants: bohr_to_angstrom
import Base: show

export RHF, dipole_moment, Polarizability, polarizability

"""
    Fermi.HartreeFock.RHFAlgorithm

Abstract type for RHF implementations.
"""
abstract type RHFAlgorithm end


"""
    Fermi.HartreeFock.RHF

Wave function object for Restricted Hartree-Fock methods

# High Level Interface 
Run a RHF computation and return the RHF object:
```
julia> @energy rhf
```
Equivalent to
```
julia> Fermi.HartreeFock.RHF()
```
Computes RHF using information from `Fermi.Options.Current`

# Fields

| Name   |   Description     |
|--------|---------------------|
| `molecule` |   Molecule object |
| `energy`   |   RHF Energy      |
| `ndocc`    | Number of doubly occupied spatial orbitals |
| `nvir`  | Number of virtual spatial orbitals |
| `orbitals` |    RHF Orbitals object      |
| `e_conv`   | ΔE from the last iteration  |
| `d_conv`   |  Orbitals RMS change from the last iteration|

# Relevant options 

These options can be set with `@set <option> <value>`

| Option         | What it does                      | Type      | choices [default]     |
|----------------|-----------------------------------|-----------|-----------------------|
| `rhf_alg`      | Picks RHF algorithm               | `Int`     | [1]                   |
| `scf_max_rms`  | RMS density convergence criterion | `Float64` | [10^-9]               |
| `scf_max_iter` | Max number of iterations          | `Int`     | [50]                  |
| `scf_e_conv`   | Energy convergence criterion      | `Float64` | [10^-10]              |
| `basis`        | What basis set to use             | `String`  | ["sto-3g"]            |
| `df`           | Whether to use density fitting    | `Bool`    | `true` [`false`]      |
| `jkfit`        | What aux. basis set to use for JK | `String`  | ["auto"]              |
| `diis`         | Whether to use DIIS               | `Bool`    | [`true`] `false`      |
| `oda`          | Whether to use ODA                | `Bool`    | [`true`] `false`      |
| `oda_cutoff`   | When to turn ODA off (RMS)        | `Float64` | [1E-1]                |
| `oda_shutoff`  | When to turn ODA off (iter)       | `Int`     | [20]                  |
| `scf_guess`    | Which guess density to use        | `String`  | "core" ["gwh"]        |

# Struct tree

**RHF** <: AbstractHFWavefunction <: AbstractWavefunction
"""
struct RHF <: AbstractHFWavefunction
    molecule::Molecule
    energy::Float64
    ndocc::Int
    nvir::Int
    orbitals::RHFOrbitals
    e_conv::Float64
    d_conv::Float64
end

function RHF(x...)
    if !any(i-> i isa RHFAlgorithm, x)
        RHF(x..., get_rhf_alg())
    else
        # Print the type of arguments given for a better feedback
        args = "("
        for a in x[1:end-1]
            args *= "$(typeof(a)), "
        end
        args = args[1:end-2]*")"
        throw(FermiException("invalid arguments for RHF method: $args"))
    end
end

function dipole_moment(wfn::RHF)
    ints = Fermi.Integrals.IntegralHelper{Float64}(molecule=wfn.molecule, basis=wfn.orbitals.basis)
    mu_ao = ints["dipole"]

    Co = wfn.orbitals.C[:, 1:wfn.ndocc]
    D = 2.0 * Co * transpose(Co)

    electronic = zeros(Float64, 3)
    for i in 1:3
        electronic[i] = sum(D .* mu_ao[:, :, i])
    end

    nuclear = zeros(Float64, 3)
    for atom in wfn.molecule.atoms
        nuclear .+= atom.Z .* (atom.xyz ./ bohr_to_angstrom)
    end

    return nuclear .- electronic
end

function _electric_field_component(evec, direction::Integer)
    1 <= direction <= 3 || throw(FermiException("Electric-field direction must be 1, 2, or 3."))
    length(evec) == 3 || throw(FermiException("Electric field must have three components."))
    return Float64(evec[direction])
end

function RHF(evec::AbstractVector, direction::Integer=1)
    RHF(IntegralHelper{Float64}(), evec, direction, get_rhf_alg())
end

function RHF(mol::Molecule, evec::AbstractVector, direction::Integer=1)
    RHF(IntegralHelper{Float64}(molecule=mol), evec, direction, get_rhf_alg())
end

function Polarizability(mol::Molecule=Molecule(); step::Real=1.0e-3, Alg::RHFAlgorithm=get_rhf_alg())
    step > 0 || throw(FermiException("Finite-difference step must be positive."))
    alpha = zeros(Float64, 3, 3)
    for direction in 1:3
        evec = zeros(Float64, 3)
        evec[direction] = Float64(step)
        plus = RHF(mol, evec, direction, Alg)
        evec[direction] = -Float64(step)
        minus = RHF(mol, evec, direction, Alg)
        alpha[:, direction] .= (dipole_moment(plus) - dipole_moment(minus)) / (2 * step)
    end
    return alpha
end

polarizability(args...; kwargs...) = Polarizability(args...; kwargs...)

"""
    Fermi.HartreeFock.get_rhf_alg()

Returns a singleton type corresponding to a RHF implementation.
"""
function get_rhf_alg(N::Int = Options.get("rhf_alg"))
    try 
        return get_rhf_alg(Val(N))
    catch MethodError
        throw(FermiException("implementation number $N not available for RHF."))
    end
end

# Actual RHF routine is in here
# For each implementation a singleton type must be create
struct RHFa <: RHFAlgorithm end
include("RHFa.jl")
include("RHFHelper.jl")
# And a number is assigned to the implementation
get_rhf_alg(x::Val{1}) = RHFa()

struct Direct <: RHFAlgorithm end
include("Direct.jl")
include("DirectHelper.jl")
get_rhf_alg(x::Val{2}) = Direct()


# Gradient methods
include("Gradients/RHFgrad.jl")

### MISCELLANEOUS
# Pretty printing
function string_repr(X::RHF)
    out = ""
    out = out*" ⇒ Fermi Restricted Hartree--Fock Wave function\n"
    out = out*" ⋅ Basis:                  $(X.orbitals.basis)\n"
    out = out*" ⋅ Energy:                 $(X.energy)\n"
    out = out*" ⋅ Occ. Spatial Orbitals:  $(X.ndocc)\n"
    out = out*" ⋅ Vir. Spatial Orbitals:  $(X.nvir)\n"
    out = out*"Convergence: " 
    out = out*"ΔE => $(format("{:1.2e}",abs(X.e_conv)))"
    out = out*" Dᵣₘₛ => $(format("{:1.2e}",abs(X.d_conv)))"
    return out
end

function show(io::IO, ::MIME"text/plain", X::RHF)
    print(io, string_repr(X))
end

