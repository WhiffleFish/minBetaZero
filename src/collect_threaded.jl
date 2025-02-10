struct GPUquery{C<:Channel, QC<:Array{Float32}, QG<:CuArray, V<:Vector, P<:Matrix}
    channels::Vector{C}
    cpu_batched_query::QC
    gpu_batched_query::QG
    cpu_value::V
    cpu_policy::P
end

function GPUquery(; sz::Tuple, na::Int, batchsize::Int)
    return GPUquery(
        Channel[],
        zeros(Float32, sz..., batchsize),
        CUDA.zeros(Float32, sz..., batchsize),
        zeros(Float32, batchsize),
        zeros(Float32, na, batchsize)
    )
end

struct LockedNetwork{N}
    net::N
    lock::ReentrantLock
end
LockedNetwork(net) = LockedNetwork(net, ReentrantLock())
(net::LockedNetwork)(args...) = lock(net.lock) do
    net.net(args...)
end

function gen_data_threaded(pomdp::POMDP, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes, GumbelSolver_args, na, input_dims, inference_batchsize) = params

    @sync begin
        progress = Progress(n_episodes)
        results_channel = Channel(n_episodes)

        progresstaskref = Ref{Task}()
        storage_channel = Channel(n_episodes; spawn=true, taskref=progresstaskref, threadpool=:interactive) do ch
            for _ in 1:n_episodes
                put!(results_channel, take!(ch))
                next!(progress)
            end
        end

        query_ch = Channel(n_episodes)

        worker_tasks = [threaded_worker(query_ch, storage_channel, pomdp, params) for _ in 1:n_episodes]

        net = LockedNetwork(gpu(net))
        sz = (input_dims..., GumbelSolver_args.n_particles)
        batchsize = inference_batchsize

        gpu_tasks = Threads.@spawn gpu_worker(query_ch, net; sz, na, batchsize)
        errormonitor(gpu_tasks)

        # gpu_tasks = Vector{Task}(undef,2)
        # for i in 1:2
        #     gpu_tasks[i] = Threads.@spawn gpu_worker(query_ch, net; sz, na, batchsize)
        #     errormonitor(gpu_tasks[i])
        # end

        wait.(worker_tasks)
        close(query_ch)

        wait(progresstaskref[])
        ch_to_buff(results_channel, buffer, n_episodes)
    end
end

function ch_to_buff(results_channel, buffer, n_episodes)
    ret_vec = zeros(Float32, n_episodes)
    steps = 0
    for i in 1:n_episodes
        data = take!(results_channel)
        steps += length(data.value_target)
        set_buffer!(
            buffer;
            network_input = data.network_input,
            value_target  = data.value_target,
            policy_target = data.policy_target
        )
        ret_vec[i] = data.returns
    end
    close(results_channel)
    return ret_vec, steps
end

function threaded_worker(query_ch::Channel, storage_channel::Channel, pomdp::POMDP, params::minBetaZeroParameters)
    (; GumbelSolver_args) = params

    worker_ch = Channel()

    worker_task = Threads.@spawn begin
        function getpolicyvalue(b)
            b_rep = input_representation(b)
            put!(query_ch, (worker_ch, b_rep))
            out = take!(worker_ch)
            return (; value=out.value[], policy=vec(out.policy))
        end


        solver = GumbelSolver(; getpolicyvalue, GumbelSolver_args...)
        planner = solve(solver, pomdp)

        data = work_fun(pomdp, planner, params)
        put!(storage_channel, data)

        return nothing
    end

    errormonitor(worker_task)
    bind(worker_ch, worker_task)

    return worker_task
end

function gpu_worker(query_ch::Channel, net; sz::Tuple, na::Int, batchsize::Int)
    Q = GPUquery(; sz, na, batchsize)

    while isopen(query_ch)
        empty!(Q.channels)

        lock(query_ch)
        try
            isready(query_ch) || wait(query_ch)
            _gpu_worker(query_ch, net, Q)
            # GC.gc(false)
        catch e
            expected_exit = isa(e, InvalidStateException) && e.state === :closed
            expected_exit ? break : rethrow()
        finally
            unlock(query_ch)
        end
    end

    return nothing
end

function _gpu_worker(query_ch::Channel, net, Q::GPUquery)
    for cpu_query in eachslice(Q.cpu_batched_query; dims=ndims(Q.cpu_batched_query))
        isready(query_ch) || break
        (ch, query) = take!(query_ch)
        push!(Q.channels, ch)
        copyto!(cpu_query, query)
    end

    copyto!(Q.gpu_batched_query, Q.cpu_batched_query)

    net_out = net(Q.gpu_batched_query; logits=true)

    copyto!(Q.cpu_value, net_out.value)
    copyto!(Q.cpu_policy, net_out.policy)

    CUDA.synchronize()

    for (i, ch) in enumerate(Q.channels)
        # pass by reference would be breaking, copying only
        response = (value = Q.cpu_value[i], policy = Q.cpu_policy[:,i])
        put!(ch, response)
    end

    return nothing
end
