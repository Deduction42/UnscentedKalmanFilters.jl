module UnscentedKalmanFilters
    include(joinpath(@__DIR__, "_StateSpaceModel.jl"))
    export 
        StateSpaceModel, 
        SigmaParams,
        GaussianState,
        kalman_filter!,
        predict_state!,
        update_state!
end
