using ArgParse
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--seedstart"
            arg_type = Int
            default = 1
        "--seedend"
            arg_type = Int
            default = 20
    end
    return parse_args(s)
end
args=parse_commandline()
seedstart=args["seedstart"]
seedend=args["seedend"]

include("import.jl")
include("generate_data_multistate.jl")

function run_multistate_1st_order(aseed)
    (nind, ndimz, nstates, ntimepoints, TTij, ssij, nobss, censori, zzi) = generate_data_multistate(aseed)
    ## initialize
    possible_transition = [
    false true true false;
    false false false true;
    false false false true;
    false false false false]
    next_state_1 = [
    true true true false;
    false true false true;
    false false true true;
    false false false true]
    next_state_inf = [
    true true true true;
    false true false true;
    false false true true;
    false false false true]
    F_betaz_est = Matrix{Vector}(undef, nstates, nstates)
    F_betaz_old = Matrix{Vector}(undef, nstates, nstates)
    F_eps = fill(0.0, nstates, nstates)
    F_hazard0_est = Matrix{Vector}(undef, nstates, nstates)
    for begin_idx = 1:nstates
        for end_idx = 1:nstates
            if possible_transition[begin_idx,end_idx]
                F_betaz_est[begin_idx,end_idx] = fill(0.0, ndimz)
                F_betaz_old[begin_idx,end_idx] = fill(0.0, ndimz)
                F_hazard0_est[begin_idx,end_idx] = fill(0.002, ntimepoints)
            end
        end
    end
    F_sub_states = Matrix{Vector}(undef, nstates, nstates)
    F_sub_states[1,1] = [1]
    F_sub_states[1,2] = [1,2]
    F_sub_states[1,3] = [1,3]
    F_sub_states[1,4] = [1,2,3,4]
    F_sub_states[2,2] = [2]
    F_sub_states[2,4] = [2,4]
    F_sub_states[3,3] = [3]
    F_sub_states[3,4] = [3,4]
    F_sub_states[4,4] = [4]
    F_sub_transitions = Matrix{Array{CartesianIndex{2},1}}(undef, nstates, nstates)
    F_sub_transitions[1,2] = [CartesianIndex(1, 2)]
    F_sub_transitions[1,3] = [CartesianIndex(1, 3)]
    F_sub_transitions[1,4] = [
    CartesianIndex(1, 2),
    CartesianIndex(1, 3),
    CartesianIndex(2, 4),
    CartesianIndex(3, 4)]
    F_sub_transitions[2,4] = [CartesianIndex(2, 4)]
    F_sub_transitions[3,4] = [CartesianIndex(3, 4)]
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
    sum_risk = 0.0
    for iter = 1:1000
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
                if state_begin == state_end
                    F_Egg[ii,state_begin,(TTij[ii,jj] + 1):TTij[ii,jj + 1]] .= 1.0
                    continue
                end
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
                    prob_temp = prob_temp * F_prob[ii,:,:,tt]
                    prob_right_tt[:,:,tt] = prob_temp
                end
                for a_state in F_sub_states[state_begin,state_end]
                    for tt = (TTij[ii,jj] + 1):TTij[ii,jj + 1]
                        F_Egg[ii,a_state,tt] =
                    prob_left_ttm1[state_begin,a_state,tt] * prob_right_tt[a_state,state_end,tt] /
                    prob_all[state_begin,state_end]
                    end
                end
                for a_sub_transition in F_sub_transitions[state_begin,state_end]
                    a_sub_state_begin = a_sub_transition[1]
                    a_sub_state_end = a_sub_transition[2]
                    for tt = (TTij[ii,jj] + 1):TTij[ii,jj + 1]
                        F_Egd[ii,a_sub_state_begin,a_sub_state_end,tt] =
                    prob_left_ttm1[state_begin,a_sub_state_begin,tt] *
                    F_prob[ii,a_sub_state_begin,a_sub_state_end,tt] *
                    prob_right_ttp1[a_sub_state_end,state_end,tt] /
                    prob_all[state_begin,state_end]
                    end
                end
            end
        end
        # for begin_idx = 1:nstates
        #     for end_idx = 1:nstates
        #         if !possible_transition[begin_idx,end_idx] continue end
        #         for ii = 1:nind
        #             for tt = 1:ntimepoints
        #                 F_Eexp[begin_idx,end_idx][ii,tt] = exp(LinearAlgebra.dot(F_betaz_est[begin_idx,end_idx], zzi[ii,:]))
        #                 F_Ezexp[begin_idx,end_idx][ii,:,tt] = F_Eexp[begin_idx,end_idx][ii,tt] * zzi[ii,:]
        #                 F_Ezzexp[begin_idx,end_idx][ii,:,:,tt] = F_Eexp[begin_idx,end_idx][ii,tt] * zzi[ii,:] * transpose(zzi[ii,:])
        #             end
        #         end
        #     end
        # end

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
                        SS_betaz = SS_betaz +
                    F_Egd[ii,begin_idx,end_idx,tt] * (zzi[ii,:] - numerator1[:,tt] / denominator[tt])
                        II_betaz = II_betaz +
                    F_Egd[ii,begin_idx,end_idx,tt] * (-numerator2[:,:,tt] / denominator[tt] +
                    (numerator1[:,tt] / denominator[tt]) * transpose(numerator1[:,tt] / denominator[tt]))
                    end
                end
                F_betaz_old[begin_idx,end_idx] = deepcopy(F_betaz_est[begin_idx,end_idx])
                F_betaz_est[begin_idx,end_idx] = F_betaz_est[begin_idx,end_idx] - II_betaz \ SS_betaz
                F_eps[begin_idx,end_idx] = sum(abs.(F_betaz_est[begin_idx,end_idx] - F_betaz_old[begin_idx,end_idx]))
                # print(begin_idx, "->", end_idx, ": ", round.(F_betaz_est[begin_idx,end_idx];digits=5), "(", round(F_eps[begin_idx,end_idx], digits=5), ")", "\n")
                for tt = 1:ntimepoints
                    sum_risk = sum(F_Egd[:,begin_idx,end_idx,tt])
                    if sum_risk == 0.0 
                        F_hazard0_est[begin_idx,end_idx][tt] = 0.0
                    else 
                        F_hazard0_est[begin_idx,end_idx][tt] = sum_risk / denominator[tt]
                    end
                end
            end
        end
        println("Max eps: ", round(maximum(F_eps), digits=5))
        if maximum(F_eps)<1e-4 break end
    end
    file_beta = "beta_multistate_1st_order.csv"
    names="beta".*string.(collect(1:ndimz))
    write_beta = DataFrames.DataFrame()
    for begin_idx = 1:nstates
        for end_idx = 1:nstates
            if !possible_transition[begin_idx,end_idx] continue end
            a_row_beta = DataFrames.DataFrame(Dict(names .=> F_betaz_est[begin_idx,end_idx]))
            DataFrames.insertcols!(a_row_beta, 1, :seed => aseed)
            DataFrames.insertcols!(a_row_beta, 2, :from => begin_idx)
            DataFrames.insertcols!(a_row_beta, 3, :to => end_idx)
            write_beta = vcat(write_beta, a_row_beta)
        end
    end
    CSV.write(file_beta,write_beta,append=true,writeheader=false)
end

println("start:",seedstart,"  end:",seedend)
for aseed=seedstart:seedend
    println("Sim ",aseed)
    run_multistate_1st_order(aseed)
end
