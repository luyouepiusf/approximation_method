function generate_data_multistate(aseed)
    Random.seed!(aseed)
    nind = 2000
    ntimepoints = 200
    alltimepoints = collect(1:ntimepoints)
    omega = 1 / ntimepoints
    timeomega = alltimepoints * omega
    nstates = 4
    hazard0_true = fill(0.0, nstates, nstates, ntimepoints)
    hazard0_12 = (20.0.*timeomega.*exp.(-5.0.*timeomega)) .* omega
    hazard0_13 = (20.0.*timeomega.*exp.(-5.0.*timeomega)) .* omega
    hazard0_24 = (1.0./(1.0.+exp.(5.0.-10.0.*timeomega))) .* omega
    hazard0_34 = (1.0./(1.0.+exp.(5.0.-10.0.*timeomega))) .* omega

    hazard0_true[1,2,:] = hazard0_12
    hazard0_true[1,3,:] = hazard0_13
    hazard0_true[2,4,:] = hazard0_24
    hazard0_true[3,4,:] = hazard0_34

    ndimz = 4
    betaz_true = fill(0.0, nstates, nstates, ndimz)
    betaz_true[1,2,:] = [0.5, 0.1, 0.1, 0.1]
    betaz_true[1,3,:] = [0.1, 0.5, 0.1, 0.1]
    betaz_true[2,4,:] = [0.1, 0.1, 0.5, 0.1]
    betaz_true[3,4,:] = [0.1, 0.1, 0.1, 0.5]
    zzi = rand(Distributions.Normal(0.0, 1.0), nind, ndimz)

    nmaxs = 20
    ssij = fill(-1, nind, 2 * nstates)
    TTij = fill(-1, nind, 2 * nstates)
    nobss = fill(-1, nind)
    censori = fill(-1, nind)

    II = LinearAlgebra.Diagonal(ones(nstates))
    statei_true = fill(-1, nind, ntimepoints)

    for ii in 1:nind
        current = 1
        for tt in 1:ntimepoints
            expterm = exp.([
            sum(zzi[ii,:] .* betaz_true[current,1,:]),
            sum(zzi[ii,:] .* betaz_true[current,2,:]),
            sum(zzi[ii,:] .* betaz_true[current,3,:]),
            sum(zzi[ii,:] .* betaz_true[current,4,:])])
            hexpterm = hazard0_true[current,:,tt] .* expterm
            probterm = 1.0 .- exp.(-hexpterm)
            probterm[current] = 1.0 - sum(deleteat!(deepcopy(probterm), current))
            current = first(findall(rand(Distributions.Multinomial(1, probterm)) .== 1))
            statei_true[ii,tt] = current
        end
        TTij_temp = convert(Array{Int64}, cumsum(ceil.(rand(Distributions.Weibull(2.0,20.0), nmaxs))))
        if TTij_temp[length(TTij_temp)] >= ntimepoints
            TTij_temp = TTij_temp[findall(TTij_temp .< ntimepoints)]
            push!(TTij_temp, ntimepoints)
        end
        ssij_temp = statei_true[ii,TTij_temp]
        pushfirst!(TTij_temp, 0)
        pushfirst!(ssij_temp, 1)
        idx = fill(false, length(ssij_temp))
        for jj in 1:length(ssij_temp)
            if jj == 1
                idx[jj] = true
            elseif ssij_temp[jj] != ssij_temp[jj - 1]
                idx[jj] = true
            end
            if ssij_temp[jj] != 4
                if jj == length(ssij_temp)
                    idx[jj] = true
                elseif ssij_temp[jj] != ssij_temp[jj + 1]
                    idx[jj] = true
                end
            end
        end
        TTij_temp2 = TTij_temp[idx]
        ssij_temp2 = ssij_temp[idx]
        nobss_temp2 = length(TTij_temp2)
    
        TTij[ii,1:nobss_temp2] = TTij_temp2
        ssij[ii,1:nobss_temp2] = ssij_temp2
        nobss[ii] = nobss_temp2
        censori[ii] = TTij_temp2[nobss_temp2]
    end

    return(
        nind,ndimz,nstates,ntimepoints,
        TTij,ssij,nobss,censori,zzi)
end