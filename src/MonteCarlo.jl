using Random
using Dates
using Printf
using MPI

mutable struct MonteCarloStatistics
    sweeps::Int
    attemptedLocalUpdates::Int
    acceptedLocalUpdates::Int
    attemptedReplicaExchanges::Int
    acceptedReplicaExchanges::Int
    initializationTime::Float64

    MonteCarloStatistics() = new(0, 0, 0, 0, 0, time())
end

mutable struct MonteCarlo{T<:Lattice,U<:AbstractRNG}
    lattice::T

    beta::Float64
    thermalizationSweeps::Int
    measurementSweeps::Int
    measurementRate::Int
    overRelaxationRate::Int
    randomizeInitialConfiguration::Bool

    replicaExchangeRate::Int
    reportInterval::Int
    checkpointInterval::Int

    rng::U
    seed::UInt
    sweep::Int

    observables::Observables
end

function MonteCarlo(
    lattice::T,
    beta::Float64,
    thermalizationSweeps::Int,
    measurementSweeps::Int;
    measurementRate::Int=1,
    overRelaxationRate::Int=1,
    randomizeInitialConfiguration::Bool=true,
    replicaExchangeRate::Int=10,
    reportInterval::Int=round(Int, 0.05 * (thermalizationSweeps + measurementSweeps)),
    checkpointInterval::Int=3600,
    rng::U=copy(Random.GLOBAL_RNG),
    seed::UInt=rand(Random.RandomDevice(), UInt)
) where T<:Lattice where U<:AbstractRNG

    mc = MonteCarlo(deepcopy(lattice), beta, thermalizationSweeps, measurementSweeps, measurementRate, overRelaxationRate, randomizeInitialConfiguration, replicaExchangeRate, reportInterval, checkpointInterval, rng, seed, 0, Observables(lattice))
    Random.seed!(mc.rng, mc.seed)

    return mc
end

function run!(mc::MonteCarlo{T}; outfile::Union{String,Nothing}=nothing) where T<:Lattice
    #init MPI
    rank = 0
    commSize = 1
    allBetas = zeros(0)
    enableMPI = false
    if MPI.Initialized()
        commSize = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if commSize > 1
            allBetas = zeros(commSize)
            allBetas[rank+1] = mc.beta
            MPI.Allgather!(UBuffer(allBetas, 1), MPI.COMM_WORLD)
            enableMPI = true
            rank == 0 && @printf("MPI detected. Enabling replica exchanges across %d simulations.\n", commSize)
        end
    end

    #init IO
    enableOutput = typeof(outfile) != Nothing
    if enableOutput
        enableMPI && (outfile *= "." * string(rank))
        isfile(outfile) && error("File ", outfile, " already exists. Terminating.")
    end

    #init spin configuration
    if mc.sweep == 0 && mc.randomizeInitialConfiguration
        for i in 1:length(mc.lattice)

            setSpin!(mc.lattice, i, uniformOnSphere(mc.rng))

        end
    end

    #init Monte Carlo run
    totalSweeps = mc.thermalizationSweeps + mc.measurementSweeps
    partnerSpinConfiguration = deepcopy(mc.lattice.spins)
    energy = getEnergy(mc.lattice)

    #launch Monte Carlo run
    lastCheckpointTime = time()
    statistics = MonteCarloStatistics()
    rank == 0 && @printf("Simulation started on %s.\n\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    while mc.sweep < totalSweeps
        #perform local sweep
        for i in 1:length(mc.lattice)
            #select random spin
            site = rand(mc.rng, 1:length(mc.lattice))

            if mc.sweep < mc.thermalizationSweeps || mc.beta <= 0.6
                newSpinState = uniformOnSphere(mc.rng)
            else
                newSpinState = rotate_update_fast(getSpin(mc.lattice, site), mc.beta, mc.rng)
            end

            energyDifference = getEnergyDifference(mc.lattice, site, newSpinState)

            #check acceptance of new configuration
            statistics.attemptedLocalUpdates += 1
            p = exp(-mc.beta * energyDifference)
            if (rand(mc.rng) < min(1.0, p))
                setSpin!(mc.lattice, site, newSpinState)
                energy += energyDifference
                statistics.acceptedLocalUpdates += 1
            end
        end
        #perform over-relaxation
        if mod(mc.sweep, mc.overRelaxationRate) == 0
            # if mod(mc.sweep, mc.reportInterval) == 0
            #     @info("Over-relaxation at sweep $(mc.sweep)")
            # end

            for site in 1:length(mc.lattice)

                newSpinState = over_relaxation(mc.lattice, site)
                spinold = getSpin(mc.lattice, site)
                if norm(newSpinState .- spinold) < 1e-10
                    @warn("Over-relaxation at site $site did not change the spin state. Spin state: $spinold")

                end
                setSpin!(mc.lattice, site, newSpinState)

                energyDifference = getEnergyDifference(mc.lattice, site, newSpinState)
                if abs(energyDifference) > 1e-10

                    @error("Over-relaxation at site $site did not yield zero energy difference. Energy difference: $energyDifference")
                end
            end

        end



        statistics.sweeps += 1

        #perform replica exchange
        if enableMPI && mc.sweep % mc.replicaExchangeRate == 0
            #determine MPI rank to exchagne configuration with
            if iseven(mc.sweep ÷ mc.replicaExchangeRate)
                partnerRank = iseven(rank) ? rank + 1 : rank - 1
            else
                partnerRank = iseven(rank) ? rank - 1 : rank + 1
            end

            if partnerRank >= 0 && partnerRank < commSize
                #obtain energy of new configuration
                partnerEnergy = MPISendrecvFloat(energy, partnerRank, MPI.COMM_WORLD)

                #check acceptance of new configuration
                statistics.attemptedReplicaExchanges += 1
                exchangeAccepted = false
                if iseven(rank)
                    p = exp(-(allBetas[rank+1] - allBetas[partnerRank+1]) * (partnerEnergy - energy))
                    exchangeAccepted = (rand(mc.rng) < min(1.0, p)) ? true : false
                    MPISendBool(exchangeAccepted, partnerRank, MPI.COMM_WORLD)
                else
                    exchangeAccepted = MPIRecvBool(partnerRank, MPI.COMM_WORLD)
                end
                if (exchangeAccepted)
                    energy = partnerEnergy
                    MPI.Sendrecv!(mc.lattice.spins, partnerRank, 0, partnerSpinConfiguration, partnerRank, 0, MPI.COMM_WORLD)
                    (mc.lattice.spins, partnerSpinConfiguration) = (partnerSpinConfiguration, mc.lattice.spins)
                    statistics.acceptedReplicaExchanges += 1
                end
            end
        end

        #perform measurement
        if mc.sweep >= mc.thermalizationSweeps
            if mc.sweep % mc.measurementRate == 0
                performMeasurements!(mc.observables, mc.lattice, energy)
            end
        end

        #increment sweep
        statistics.sweeps += 1
        mc.sweep += 1

        #runtime statistics
        t = time()
        if mc.sweep % mc.reportInterval == 0
            #collect statistics
            progress = 100.0 * mc.sweep / totalSweeps
            thermalized = (mc.sweep >= mc.thermalizationSweeps) ? "YES" : "NO"
            sweeprate = statistics.sweeps / (t - statistics.initializationTime)
            sweeptime = 1.0 / sweeprate
            eta = (totalSweeps - mc.sweep) / sweeprate

            localUpdateAcceptanceRate = 100.0 * statistics.acceptedLocalUpdates / statistics.attemptedLocalUpdates
            if enableMPI
                replicaExchangeAcceptanceRate = 100.0 * statistics.acceptedReplicaExchanges / statistics.attemptedReplicaExchanges
                allLocalAppectanceRate = zeros(commSize)
                allLocalAppectanceRate[rank+1] = localUpdateAcceptanceRate
                MPI.Allgather!(UBuffer(allLocalAppectanceRate, 1), MPI.COMM_WORLD)
                allReplicaExchangeAcceptanceRate = zeros(commSize)
                allReplicaExchangeAcceptanceRate[rank+1] = replicaExchangeAcceptanceRate
                MPI.Allgather!(UBuffer(allReplicaExchangeAcceptanceRate, 1), MPI.COMM_WORLD)
            end

            #print statistics
            if rank == 0
                str = ""
                str *= @sprintf("Sweep %d / %d (%.1f%%)", mc.sweep, totalSweeps, progress)
                str *= @sprintf("\t\tETA : %s\n", Dates.format(Dates.now() + Dates.Second(round(Int64, eta)), "dd u yyyy HH:MM:SS"))
                str *= @sprintf("\t\tthermalized : %s\n", thermalized)
                str *= @sprintf("\t\tsweep rate : %.1f sweeps/s\n", sweeprate)
                str *= @sprintf("\t\tsweep duration : %.3f ms\n", sweeptime * 1000)

                if enableMPI
                    for n in 1:commSize
                        str *= @sprintf("\t\tsimulation %d update acceptance rate: %.2f%%\n", n - 1, allLocalAppectanceRate[n])
                        str *= @sprintf("\t\tsimulation %d replica exchange acceptance rate : %.2f%%\n", n - 1, allReplicaExchangeAcceptanceRate[n])
                    end
                else
                    str *= @sprintf("\t\tupdate acceptance rate: %.2f%%\n", localUpdateAcceptanceRate)
                end
                str *= @sprintf("\n")
                print(str)
            end

            #reset statistics
            statistics = MonteCarloStatistics()
        end

        #write checkpoint
        if enableOutput
            checkpointPending = time() - lastCheckpointTime >= mc.checkpointInterval
            enableMPI && (checkpointPending = MPIBcastBool(checkpointPending, 0, MPI.COMM_WORLD))
            if checkpointPending
                writeMonteCarlo(outfile, mc)
                lastCheckpointTime = time()
                rank == 0 && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
            end
        end
    end

    #write final checkpoint
    if enableOutput
        writeMonteCarlo(outfile, mc)
        rank == 0 && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    end

    #return
    rank == 0 && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    return nothing
end

function anneal!(mc::MonteCarlo{T}, betas::Vector{Float64};
    outfile::Union{String,Nothing}=nothing, procid=nothing,
    randomizeInitialConfiguration::Bool=true) where {T<:Lattice}

    mc.randomizeInitialConfiguration = randomizeInitialConfiguration

    for (i, beta) in enumerate(betas)
        @info "Running simulation $(i)/$(length(betas)) at β = $beta"

        mc.beta = beta
        mc.sweep = 0
        mc.observables = deepcopy(MonteCarlo(mc.lattice, mc.beta, 0, 0).observables)
        out = isnothing(outfile) ? nothing :
              replace(outfile, r"\.h5$" => "_beta$(beta).h5") * (isnothing(procid) ? "" : ".$(procid)")

        run!(mc, outfile=out)


        mc.randomizeInitialConfiguration = false
    end
end

