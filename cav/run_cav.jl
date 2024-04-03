import Pkg
import CSV
import Random
import CPUTime
import Tables
import DataFrames
import Distributions
import LinearAlgebra

function main()
    dir="clean_data/"
    df_TTij=CSV.read(dir*"TTij.csv",DataFrames.DataFrame)
    df_ssij=CSV.read(dir*"ssij.csv",DataFrames.DataFrame)
    df_zzi=CSV.read(dir*"zzi.csv",DataFrames.DataFrame)
    df_censori=CSV.read(dir*"censori.csv",DataFrames.DataFrame)
    df_nobss=CSV.read(dir*"nobss.csv",DataFrames.DataFrame)
    
    TTij=Matrix{Int64}(df_TTij)
    ssij=Matrix{Int64}(df_ssij)
    zzi=Matrix{Float64}(df_zzi)
    censori=Vector{Int64}(df_censori[:,1])
    nobss=Vector{Int64}(df_nobss[:,1])
    
    nind=size(TTij,1)
    ndimz=size(zzi,2)
    ntimepoints=maximum(TTij)
    nstates=maximum(ssij)

    possible_transition = [
        false true false true;
        true false true true;
        false true false true;
        false false false false]
    next_state_1 = [
        true true false true;
        true true true true;
        false true true true;
        false false false true]
    next_state_inf = [
        true true true true;
        true true true true;
        true true true true;
        false false false true]
    F_betaz_est = Matrix{Vector}(undef, nstates, nstates)
    F_betaz_old = Matrix{Vector}(undef, nstates, nstates)
    F_II_betaz = Matrix{Matrix}(undef, nstates, nstates)
    F_eps = fill(0.0, nstates, nstates)
    F_hazard0_est = Matrix{Vector}(undef, nstates, nstates)
    for begin_idx = 1:nstates
        for end_idx = 1:nstates
            if possible_transition[begin_idx,end_idx]
                F_betaz_est[begin_idx,end_idx] = fill(0.0, ndimz)
                F_betaz_old[begin_idx,end_idx] = fill(0.0, ndimz)
                F_II_betaz[begin_idx,end_idx] = fill(0.0, ndimz, ndimz)
                F_hazard0_est[begin_idx,end_idx] = fill(0.0001, ntimepoints)
            end
        end
    end
    F_sub_states = Matrix{Vector}(undef, nstates, nstates)
    F_sub_states[1,1] = [1,2,3]
    F_sub_states[1,2] = [1,2,3]
    F_sub_states[1,3] = [1,2,3]
    F_sub_states[1,4] = [1,2,3,4]
    F_sub_states[2,1] = [1,2,3]
    F_sub_states[2,2] = [1,2,3]
    F_sub_states[2,3] = [1,2,3]
    F_sub_states[2,4] = [1,2,3,4]
    F_sub_states[3,1] = [1,2,3]
    F_sub_states[3,2] = [1,2,3]
    F_sub_states[3,3] = [1,2,3]
    F_sub_states[3,4] = [1,2,3,4]
    F_sub_states[4,4] = [4]
    F_sub_transitions = Matrix{Array{CartesianIndex{2},1}}(undef, nstates, nstates)
    F_sub_transitions[1,1] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[1,2] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[1,3] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[1,4] = [
        CartesianIndex(1,2),
        CartesianIndex(1,4),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(2,4),
        CartesianIndex(3,2),
        CartesianIndex(3,4)]
    F_sub_transitions[2,1] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[2,2] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[2,3] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[2,4] = [
        CartesianIndex(1,2),
        CartesianIndex(1,4),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(2,4),
        CartesianIndex(3,2),
        CartesianIndex(3,4)]
    F_sub_transitions[3,1] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[3,2] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[3,3] = [
        CartesianIndex(1,2),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(3,2)]
    F_sub_transitions[3,4] = [
        CartesianIndex(1,2),
        CartesianIndex(1,4),
        CartesianIndex(2,1),
        CartesianIndex(2,3),
        CartesianIndex(2,4),
        CartesianIndex(3,2),
        CartesianIndex(3,4)]
    F_Egd = fill(NaN, nind, nstates, nstates, ntimepoints)
    F_Egg = fill(NaN, nind, nstates, ntimepoints)
    F_risk = fill(NaN, nind, nstates, nstates)
    F_expr = fill(NaN, nind, nstates, nstates)
    F_zexpr = fill(NaN, nind, nstates, nstates, ndimz)
    F_zzexpr = fill(NaN, nind, nstates, nstates, ndimz, ndimz)
    F_hexpr = fill(NaN, nind, nstates, nstates, ntimepoints)
    F_prob = fill(NaN, nind, nstates, nstates, ntimepoints)
    
    F_SS_betaz = Matrix{Vector}(undef, nstates, nstates)
    F_II_betaz = Matrix{Matrix}(undef, nstates, nstates)
    SS_betaz = fill(0.0, ndimz)
    II_betaz = fill(0.0, ndimz, ndimz)
    SS_penalty = fill(0.0, ndimz)
    II_penalty = fill(0.0, ndimz, ndimz)
    sum_risk = 0.0
    identity = Matrix{Float64}(LinearAlgebra.I,ndimz,ndimz)
    
    niter=2000
    factor_min=0.1
    factor_max=0.5
    factor=factor_min
    all_factor=LinRange(factor_min,factor_max,niter)

    for iter = 1:niter
        factor=all_factor[iter]
        println("iter: ",iter,", factor: ",factor)
        
        F_prob .= 0.0
        for begin_idx = 1:nstates
            for end_idx = 1:nstates
                if !possible_transition[begin_idx,end_idx] continue end
                for ii = 1:nind
                    F_risk[ii,begin_idx,end_idx] = LinearAlgebra.dot(F_betaz_est[begin_idx,end_idx], zzi[ii,:])
                    F_expr[ii,begin_idx,end_idx] = exp(F_risk[ii,begin_idx,end_idx])
                    F_zexpr[ii,begin_idx,end_idx,:] = F_expr[ii,begin_idx,end_idx] * zzi[ii,:]
                    F_zzexpr[ii,begin_idx,end_idx,:,:] = F_expr[ii,begin_idx,end_idx] * zzi[ii,:] * transpose(zzi[ii,:])        
                    for tt = 1:ntimepoints
                        F_hexpr[ii,begin_idx,end_idx,tt] = F_hazard0_est[begin_idx,end_idx][tt] * F_expr[ii,begin_idx,end_idx]
                        F_prob[ii,begin_idx,end_idx,tt] = 1.0 - exp(-F_hexpr[ii,begin_idx,end_idx,tt])
                    end
                end
            end
        end
        for ii = 1:nind
            for begin_idx = 1:nstates
                for tt = 1:ntimepoints
                    F_prob[ii,begin_idx,begin_idx,tt] = 1.0 - sum(F_prob[ii,begin_idx,:,tt])
                end
            end
        end
        F_Egg .= 0.0
        F_Egd .= 0.0
        prob_left_ttm1 = fill(0.0, nstates, nstates, ntimepoints)
        prob_right_ttp1 = fill(0.0, nstates, nstates, ntimepoints)
        prob_right_tt = fill(0.0, nstates, nstates, ntimepoints)
        prob_all = fill(0.0, nstates, nstates)
        for ii = 1:nind
            for jj = 1:(nobss[ii] - 1)
                state_begin = ssij[ii,jj]
                state_end = ssij[ii,jj + 1]
                # if state_begin == state_end
                #     F_Egg[ii,state_begin,(TTij[ii,jj] + 1):TTij[ii,jj + 1]] .= 1.0
                #     continue
                # end
                prob_left_ttm1 = fill(NaN, nstates, nstates, ntimepoints) ###
                prob_right_ttp1 = fill(NaN, nstates, nstates, ntimepoints) ###
                prob_right_tt = fill(NaN, nstates, nstates, ntimepoints) ###
                prob_all = fill(NaN, nstates, nstates) ###
            
                prob_temp = LinearAlgebra.Diagonal(ones(nstates))
                for tt = (TTij[ii,jj] + 1):TTij[ii,jj + 1]
                    prob_left_ttm1[:,:,tt] = prob_temp
                    prob_temp = prob_temp * F_prob[ii,:,:,tt]
                end
                prob_all = prob_temp
                prob_temp = LinearAlgebra.Diagonal(ones(nstates))
                for tt = TTij[ii,jj + 1]:(-1):(TTij[ii,jj] + 1)
                    prob_right_ttp1[:,:,tt] = prob_temp
                    prob_temp = F_prob[ii,:,:,tt] * prob_temp
                    prob_right_tt[:,:,tt] = prob_temp
                end
                for a_state in F_sub_states[state_begin,state_end]
                    for tt = (TTij[ii,jj] + 1):TTij[ii,jj + 1]
                        if prob_all[state_begin,state_end]>0.0
                            F_Egg[ii,a_state,tt] =
                                prob_left_ttm1[state_begin,a_state,tt] * prob_right_tt[a_state,state_end,tt] /
                                prob_all[state_begin,state_end]
                        end
                    end
                end
                for a_sub_transition in F_sub_transitions[state_begin,state_end]
                    a_sub_state_begin = a_sub_transition[1]
                    a_sub_state_end = a_sub_transition[2]
                    for tt = (TTij[ii,jj] + 1):TTij[ii,jj + 1]
                        if prob_all[state_begin,state_end]>0.0
                            F_Egd[ii,a_sub_state_begin,a_sub_state_end,tt] =
                                prob_left_ttm1[state_begin,a_sub_state_begin,tt] *
                                F_prob[ii,a_sub_state_begin,a_sub_state_end,tt] *
                                prob_right_ttp1[a_sub_state_end,state_end,tt] /
                                prob_all[state_begin,state_end]
                        end
                    end
                end
            end
        end
        for begin_idx = 1:nstates
            for end_idx = 1:nstates
                if !possible_transition[begin_idx,end_idx] continue end
                denominator = fill(0.0, ntimepoints)
                numerator1 = fill(0.0, ndimz, ntimepoints)
                numerator2 = fill(0.0, ndimz, ndimz, ntimepoints)
                for tt = 1:ntimepoints
                    for ii = 1:nind
                        denominator[tt] = denominator[tt] +
                    (F_Egg[ii,begin_idx,tt] - F_Egd[ii,begin_idx,end_idx,tt] / 2.0) * F_expr[ii,begin_idx,end_idx]
                        numerator1[:,tt] = numerator1[:,tt] +
                    (F_Egg[ii,begin_idx,tt] - F_Egd[ii,begin_idx,end_idx,tt] / 2.0) * F_zexpr[ii,begin_idx,end_idx,:]
                        numerator2[:,:,tt] = numerator2[:,:,tt] +
                    (F_Egg[ii,begin_idx,tt] - F_Egd[ii,begin_idx,end_idx,tt] / 2.0) * F_zzexpr[ii,begin_idx,end_idx,:,:]
                    end
                end
                SS_betaz = fill(0.0, ndimz)
                II_betaz = fill(0.0, ndimz, ndimz)
                for ii = 1:nind
                    for tt = 1:ntimepoints
                        if F_Egg[ii,begin_idx,tt] == 0.0 continue end
                        if denominator[tt] <= 0.0 continue end
                        SS_betaz = SS_betaz +
                    F_Egd[ii,begin_idx,end_idx,tt] * (zzi[ii,:] - numerator1[:,tt] / denominator[tt])
                        II_betaz = II_betaz +
                    F_Egd[ii,begin_idx,end_idx,tt] * (-numerator2[:,:,tt] / denominator[tt] +
                    (numerator1[:,tt] / denominator[tt]) * transpose(numerator1[:,tt] / denominator[tt]))
                    end
                end
                SS_penalty = -0.0.*F_betaz_est[begin_idx,end_idx]
                II_penalty = -0.0.*identity
                F_II_betaz[begin_idx,end_idx] = II_betaz
                F_betaz_old[begin_idx,end_idx] = deepcopy(F_betaz_est[begin_idx,end_idx])
                F_betaz_est[begin_idx,end_idx] = F_betaz_est[begin_idx,end_idx] - factor.*((II_betaz + II_penalty) \ (SS_betaz + SS_penalty))
                F_betaz_est[begin_idx,end_idx] = min.(max.(F_betaz_est[begin_idx,end_idx], -5.0), 5.0)
                F_eps[begin_idx,end_idx] = sum(abs.(F_betaz_est[begin_idx,end_idx] - F_betaz_old[begin_idx,end_idx]))
                print(begin_idx, "->", end_idx, ": ", round.(F_betaz_est[begin_idx,end_idx];digits=5), "(", round(F_eps[begin_idx,end_idx], digits=5), ")", "\n")
                for tt = 1:ntimepoints
                    sum_risk = sum(F_Egd[:,begin_idx,end_idx,tt])
                    if sum_risk == 0.0 
                        F_hazard0_est[begin_idx,end_idx][tt] = 0.0
                    else 
                        F_hazard0_est[begin_idx,end_idx][tt] = min(max(sum_risk / denominator[tt], 0.0), 5.0)
                    end
                end
            end
        end
        file_beta = "beta_multistate_1st_order.csv"
        names="beta".*string.(collect(1:ndimz))
        write_beta = DataFrames.DataFrame()
        for begin_idx = 1:nstates
            for end_idx = 1:nstates
                if !possible_transition[begin_idx,end_idx] continue end
                a_row_beta = DataFrames.DataFrame(Dict(names .=> F_betaz_est[begin_idx,end_idx]))
                DataFrames.insertcols!(a_row_beta, 1, :iter => iter)
                DataFrames.insertcols!(a_row_beta, 2, :from => begin_idx)
                DataFrames.insertcols!(a_row_beta, 3, :to => end_idx)
                write_beta = vcat(write_beta, a_row_beta)
            end
        end
        CSV.write(file_beta,write_beta,append=true,writeheader=false)
        file_II = "II_multistate_1st_order.csv"
        names="II".*string.(collect(1:ndimz))
        write_II = DataFrames.DataFrame()
        for begin_idx = 1:nstates
            for end_idx = 1:nstates
                if !possible_transition[begin_idx,end_idx] continue end
                a_row_II = DataFrames.DataFrame(F_II_betaz[begin_idx,end_idx],:auto)
                DataFrames.insertcols!(a_row_II, 1, :iter => iter)
                DataFrames.insertcols!(a_row_II, 2, :from => begin_idx)
                DataFrames.insertcols!(a_row_II, 3, :to => end_idx)
                DataFrames.insertcols!(a_row_II, 4, :row => collect(1:ndimz))
                write_II = vcat(write_II, a_row_II)
            end
        end
        CSV.write(file_II,write_II,append=true,writeheader=false)
        file_hazard0 = "hazard0_multistate_1st_order.csv"
        names="hazard0".*string.(collect(1:ndimz))
        write_hazard0 = DataFrames.DataFrame()
        for begin_idx = 1:nstates
            for end_idx = 1:nstates
                if !possible_transition[begin_idx,end_idx] continue end
                a_row_hazard0 = DataFrames.DataFrame(transpose(F_hazard0_est[begin_idx,end_idx]),:auto)
                DataFrames.insertcols!(a_row_hazard0, 1, :iter => iter)
                DataFrames.insertcols!(a_row_hazard0, 2, :from => begin_idx)
                DataFrames.insertcols!(a_row_hazard0, 3, :to => end_idx)
                write_hazard0 = vcat(write_hazard0, a_row_hazard0)
            end
        end
        CSV.write(file_hazard0,write_hazard0,append=true,writeheader=false)
        if isnan(maximum(F_eps))
            println("Singular Systems!")
            break
        end
        println("Max eps: ", round(maximum(F_eps), digits=5))
        if maximum(F_eps)<1e-4 break end
    end
end

main()
