function compute_S!(I::IntegralHelper{T, <:AbstractERI, O}, ints::IntegralHelper{T, <:AbstractERI, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    # Compute AO overlap
    Sao = ints["S"]

    # Convert AO overlap to MO overlap
    C = I.orbitals.C
    if eltype(I.orbitals.C) !== T
        C = T.(C)
    end
    @tensoropt S[i,j] := Sao[μ, ν]*C[μ,i]*C[ν,j]
    I.cache["S"] = S
end

function compute_T!(I::IntegralHelper{T, <:AbstractERI, O}, ints::IntegralHelper{T, <:AbstractERI, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    # Compute AO overlap
    Tao = ints["T"]

    # Convert AO overlap to MO overlap
    C = I.orbitals.C
    if eltype(I.orbitals.C) !== T
        C = T.(C)
    end
    @tensoropt Tmo[i,j] := Tao[μ, ν]*C[μ,i]*C[ν,j]
    I.cache["T"] = Tmo
end

function compute_V!(I::IntegralHelper{T, <:AbstractERI, O}, ints::IntegralHelper{T, <:AbstractERI, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    # Compute AO overlap
    Vao = ints["V"]

    # Convert AO overlap to MO overlap
    C = I.orbitals.C
    if eltype(I.orbitals.C) !== T
        C = T.(C)
    end
    @tensoropt V[i,j] := Vao[μ, ν]*C[μ,i]*C[ν,j]
    I.cache["V"] = V
end

function compute_dipole!(I::IntegralHelper{T, <:AbstractERI, O}, ints::IntegralHelper{T, <:AbstractERI, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    # Compute AO dipole integrals
    dA = ints["dipole"]

    # Convert AO dipole integrals to MO dipole integrals
    C = I.orbitals.C
    if eltype(I.orbitals.C) !== T
        C = T.(C)
    end
    @tensoropt d[i,j,x] := dA[μ, ν, x]*C[μ,i]*C[ν,j]
    I.cache["dipole"] = d
end