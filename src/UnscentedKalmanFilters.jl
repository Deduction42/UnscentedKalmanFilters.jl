module UnscentedKalmanFilters
    include(joinpath(@__DIR__, "_StateSpaceModel.jl"))
    export 
        StateSpaceModel, 
        SigmaParams, 
        kalman_filter!
end
