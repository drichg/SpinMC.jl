using Random
using LinearAlgebra



function uniformOnSphere(rng=Random.GLOBAL_RNG)
    phi = 2.0 * pi * rand(rng)
    z = 2.0 * rand(rng) - 1.0
    r = sqrt(1.0 - z * z)
    return (r * cos(phi), r * sin(phi), z)
end
function Ising_spin(rng=Random.GLOBAL_RNG)
    z = sign(rand(rng) - 0.5)

    return (0.0, 0.0, z)
end
R(θ, ϕ) = [cos(θ)*cos(ϕ) -sin(ϕ) sin(θ)*cos(ϕ);
    cos(θ)*sin(ϕ) cos(ϕ) sin(θ)*sin(ϕ);
    -sin(θ) 0 cos(θ)]
function rotate_update(s, beta, rng=Random.GLOBAL_RNG)
    θ = acos(s[3])
    ϕ = atan(s[2], s[1])
    if ϕ < 0
        ϕ += 2π
    end
    Δθ = acos(1 - 1.0 / beta)
    ϕ_new = rand(rng) * 2π
    θ_new = Δθ * rand(rng)
    si_new = R(θ, ϕ) * [sin(θ_new) * cos(ϕ_new),
        sin(θ_new) * sin(ϕ_new),
        cos(θ_new)]
    return (si_new[1], si_new[2], si_new[3])
end
function rotate_update_fast(s::NTuple{3,Float64}, beta, rng=Random.GLOBAL_RNG)
    z = s[3]
    θ = acos(z)
    ϕ = atan(s[2], s[1])
    ϕ += (ϕ < 0) * 2π

    Δθ = acos(1 - 1.0 / beta)
    ϕ_new = 2π * rand(rng)
    θ_new = Δθ * rand(rng)

    # Original vector in local frame
    vx = sin(θ_new) * cos(ϕ_new)
    vy = sin(θ_new) * sin(ϕ_new)
    vz = cos(θ_new)

    # Directly apply rotated form
    cosθ = cos(θ)
    sinθ = sin(θ)
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)

    x = cosθ * cosϕ * vx - sinϕ * vy + sinθ * cosϕ * vz
    y = cosθ * sinϕ * vx + cosϕ * vy + sinθ * sinϕ * vz
    z = -sinθ * vx + cosθ * vz

    return (x, y, z)
end
function over_relaxation(lattice, site::Int)
    h = (0, 0, 0)
    for (interaction, s) in zip(lattice.interactionMatrices[site], lattice.interactionSites[site])
        interactionSpin = getSpin(lattice, s)
        # Accumulate the interaction contributions
        h = h .+ (interaction.m11 * interactionSpin[1], interaction.m22 * interactionSpin[2], interaction.m33 * interactionSpin[3])
    end
    local_field = getInteractionField(lattice, site)
    h = h .+ local_field
    if norm(h) < 1e-10
        spin_new = getSpin(lattice, site)  # No change if local field is zero
    else
        spin_old = getSpin(lattice, site)
        spin_new = 2 .* h .* dot(spin_old, h) ./ norm(h)^2 .- spin_old
    end
    return spin_new
end

function exchangeEnergy(s1, M::InteractionMatrix, s2)::Float64
    return s1[1] * (M.m11 * s2[1] + M.m12 * s2[2] + M.m13 * s2[3]) + s1[2] * (M.m21 * s2[1] + M.m22 * s2[2] + M.m23 * s2[3]) + s1[3] * (M.m31 * s2[1] + M.m32 * s2[2] + M.m33 * s2[3])
end

function getEnergy(lattice::Lattice{D,N})::Float64 where {D,N}
    energy = 0.0

    for site in 1:length(lattice)
        s0 = getSpin(lattice, site)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            if site > interactionSites[i]
                energy += exchangeEnergy(s0, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
            end
        end

        #onsite interaction
        energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end

function getEnergyDifference(lattice::Lattice{D,N}, site::Int, newState::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    dE = 0.0
    oldState = getSpin(lattice, site)
    ds = newState .- oldState

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    for i in 1:length(interactionSites)
        dE += exchangeEnergy(ds, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
    end

    #onsite interaction
    interactionOnsite = getInteractionOnsite(lattice, site)
    dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getMagnetization(lattice::Lattice{D,N}) where {D,N}
    mx, my, mz = 0.0, 0.0, 0.0
    for i in 1:length(lattice)
        spin = getSpin(lattice, i)
        mx += spin[1]
        my += spin[2]
        mz += spin[3]
    end
    return [mx, my, mz] / length(lattice)
end

function getCorrelation(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice.unitcell.basis))
    for i in 1:length(lattice.unitcell.basis)
        s0 = getSpin(lattice, i)
        for j in 1:length(lattice)
            corr[j, i] = dot(s0, getSpin(lattice, j))
        end
    end
    return corr
end

