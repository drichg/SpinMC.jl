using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    # energy::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    magnetization::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    magnetizationVector::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    correlation::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    monopoleOrderTotal::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    monopoleOrderTotal2::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    monopoleOrderh::FullBinner{Float64}
    monopoleOrderv::FullBinner{Float64}
end

function Observables(lattice::T) where T<:Lattice
    return Observables(ErrorPropagator(Float64), LogBinner(Float64), LogBinner(zeros(Float64, 3)), LogBinner(zeros(Float64, lattice.length, length(lattice.unitcell.basis))),
        LogBinner(Float64), LogBinner(Float64), FullBinner(Float64), FullBinner(Float64))
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
    push!(observables.monopoleOrderTotal, monopole_order(lattice)[3])
    push!(observables.monopoleOrderTotal2, monopole_order(lattice)[4])
end
function monopole_order(lattice::T) where T<:Lattice
    # Calculate the monopole order parameter for Shastry-Sutherland model
    bh = 0.0
    bv = 0.0


    L = length(lattice)
    Lx2, Ly2 = 2 * size(lattice)[1], 2 * size(lattice)[2]
    signs_i = ntuple(i -> (-1)^i, Lx2)
    signs_j = ntuple(j -> (-1)^j, Ly2)

    for i in 0:Lx2-1
        for j in 0:Ly2-1


            ip1 = (i + 1 < Lx2) ? i + 1 : 0
            jp1 = (j + 1 < Ly2) ? j + 1 : 0

            pos1 = (i, j)
            pos2 = (ip1, j)
            pos3 = (i, jp1)

            @inbounds x1 = lattice.positionMap[pos1]
            @inbounds x2 = lattice.positionMap[pos2]
            @inbounds x3 = lattice.positionMap[pos3]
            s0 = getSpin(lattice, x1)
            sh = getSpin(lattice, x2)
            sv = getSpin(lattice, x3)

            # dsh = s0[1] * sh[1] + s0[2] * sh[2] + s0[3] * sh[3]
            # dsv = s0[1] * sv[1] + s0[2] * sv[2] + s0[3] * sv[3]
            dsh = dot(s0, sh)
            dsv = dot(s0, sv)

            @fastmath bh += signs_i[i+1] * dsh
            @fastmath bv += signs_j[j+1] * dsv

            # for k in 0:2(size(lattice)[2])-1
            #     for l in 0:2(size(lattice)[2])-1
            #         y1 = getSiteIndex(lattice, (k, l))
            #         y2 = getSiteIndex(lattice, (mod(k + 1, 2 * lattice.size[1]), l))
            #         y3 = getSiteIndex(lattice, (k, mod(l + 1, 2 * lattice.size[2])))

            #         t0 = getSpin(lattice, y1)
            #         th = getSpin(lattice, y2)
            #         tv = getSpin(lattice, y3)
            #         dth = dot(t0, th)
            #         dtv = dot(t0, tv)
            #         Mt1 += (-1)^(i + k) * dsh * dth + (-1)^(j + l) * dsv * dtv -
            #                (-1)^(j + k) * dsv * dth - (-1)^(i + l) * dsh * dtv

            #     end
            # end
        end
    end
    Δ = bh - bv
    return bh / L, bv / L, (Δ^2) / (L^2), (Δ^4) / (L^4)
end