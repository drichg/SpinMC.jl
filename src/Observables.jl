using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    # energy::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    magnetization::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    magnetizationVector::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    correlation::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    monopoleOrderh::FullBinner{Float64}
    monopoleOrderv::FullBinner{Float64}
end

function Observables(lattice::T) where T<:Lattice
    return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64, 3)), LogBinner(zeros(Float64, lattice.length, length(lattice.unitcell.basis))), FullBinner(Float64), FullBinner(Float64))
    # return Observables(LogBinner(Float64), LogBinner(Float64), LogBinner(zeros(Float64, 3)), LogBinner(zeros(Float64, lattice.length, length(lattice.unitcell.basis))))
end

function performMeasurements!(observables::Observables, lattice::T, energy::Float64) where T<:Lattice
    #measure energy and energy^2
    push!(observables.energy, energy / length(lattice), energy * energy / (length(lattice) * length(lattice)))
    # push!(observables.energy, energy / length(lattice))
    #measure magnetization
    m = getMagnetization(lattice)
    push!(observables.magnetization, norm(m))
    push!(observables.magnetizationVector, m)

    #measure spin correlations
    push!(observables.correlation, getCorrelation(lattice))
    #measure monopole order
    push!(observables.monopoleOrderh, monopole_order(lattice)[1])
    push!(observables.monopoleOrderv, monopole_order(lattice)[2])
end
function monopole_order(lattice::T) where T<:Lattice
    # Calculate the monopole order parameter for Shastry-Sutherland model
    bh = 0.0
    bv = 0.0

    for i in 0:2(size(lattice)[1])-1
        for j in 0:2(size(lattice)[2])-1
            x1 = getSiteIndex(lattice, (i, j))
            x2 = getSiteIndex(lattice, (mod(i + 1, 2 * lattice.size[1]), j))
            x3 = getSiteIndex(lattice, (i, mod(j + 1, 2 * lattice.size[2])))

            s0 = getSpin(lattice, x1)
            sh = getSpin(lattice, x2)
            sv = getSpin(lattice, x3)

            bh += (-1)^i * dot(s0, sh)
            bv += (-1)^j * dot(s0, sv)
        end
    end

    return bh / length(lattice), bv / length(lattice)
end