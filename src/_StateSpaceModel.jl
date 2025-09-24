#=
To Do:
    1: Make this square root form work for linear systems too, now that we split up predict and update
    2: Add the "observation reduction" function to modify the observer to allign with finite values of y 
       This model reduction should be automatic. 
       Since it operates on the same states, updating the modified problem should update the original
=#

using LinearAlgebra
import Statistics.mean
import Statistics.cov

"""
Sigma point parameters for the unscented kalman filter (α~0 => EKF α~1=> Nonlinear Gaussian, κ=0, β=2 for Gaussian)
"""
Base.@kwdef struct SigmaParams
    α :: Float64 = 0.001
    κ :: Float64 = 0.0
    β :: Float64 = 2.0
end

LinearPredictor = Tuple{<:AbstractArray,<:AbstractArray}
StatePredictor  = Union{Function, LinearPredictor}

"""
State-Space model
Fields:   
    fxu: state transition function f of inputs x and u (or a tuple of two matrices for the linear version)
    hxu: state observation function h of inputs x and u (or a tuple of two matrices for the linear version)
    x: "a priori" state estimate
    PU: "a priori" estimated state covariance (Upper-Triangular form)
    QU: process noise covariance (Upper-Triangular form)
    RU: measurement noise covariance (Upper-Triangular form)
"""
Base.@kwdef struct StateSpaceModel{T<:Real, F1<:StatePredictor, F2<:StatePredictor}
    fxu ::   F1
    hxu ::   F2
    x   ::   Vector{T}
    QU  ::   UpperTriangular{T, Matrix{T}}
    RU  ::   UpperTriangular{T, Matrix{T}}
    PU  ::   UpperTriangular{T, Matrix{T}}
    θ   ::   SigmaParams = SigmaParams()
end

function StateSpaceModel(fx::F1, hx::F2, x, QU, RU, PU, θ) where {F1,F2}
    T = promote_type(eltype(x), eltype(QU), eltype(RU), eltype(PU))
    return StateSpaceModel{T, F1, F2}(fx, hx, x, QU, RU, PU, θ)
end

"""
GaussianState: contains the canonical representation of a model's state (x = state, P = state covariance)
"""
@kwdef struct GaussianState
    x :: Vector{Float64}
    P :: Matrix{Float64}
end

function GaussianState(model::StateSpaceModel)
    return GaussianState(
        x = deepcopy(model.x),
        P = model.PU'model.PU
    )
end

"""
kalman_filter!(SS::StateSpaceModel{T}, y::AbstractVector, u; multithreaded_predict=false, multithreaded_observe=false, outlier=3.0) where T

Applies state prediction and update functionality in-place to the state space model (does not update if there are NaN values)
Output: Primary output mutates state-space model fields (x, PU),
Additionaly, a NamedTuple is provided with the following fields:
    xh: State estimate before the update (for troubleshooting)
    Ph: State covariance before the update (for troubleshooting)
    yh: Predicted observation (for model validation)
    K:  Kalman gain (for troubleshooting)
"""
function kalman_filter!(SS::StateSpaceModel{T}, y::AbstractVector, u; multithreaded_predict=false, multithreaded_observe=false, outlier=3.0) where T
    #Propagate sigma points through transition
    TR = promote_type(T, Float64)
    (xh, Ph) = predict_state!(SS, u, multithreaded=multithreaded_predict)
    (yh, K)  = update_state!(SS, y, u,  multithreaded=multithreaded_observe, outlier=outlier)

    return (xh=xh, Ph=Ph, yh=yh, K=K)
end



"""
Weights for sigma points, calculated from SigmaParams and the state dimension L
"""
Base.@kwdef struct SigmaWeights
    c :: Float64
    μ :: Tuple{Float64, Float64}
    Σ :: Tuple{Float64, Float64}
end

function SigmaWeights(L::Int64, θ::SigmaParams=SigmaParams())
    α = θ.α
    κ = θ.κ
    β = θ.β

    λ  = α^2*(L+κ)-L                          #scaling factor
    c  = L + λ                                #scaling factor
    Wn = 0.5/c
    Wμ = ((λ/c), Wn)                         #weights for means
    WΣ = (Wμ[1] + (1-α^2+β), Wn)             #weights for covariance
    return SigmaWeights(c=c, μ=Wμ, Σ=WΣ)
end

function SigmaWeights(SS::StateSpaceModel)
    return SigmaWeights(length(SS.x), SS.θ)
end


"""
Sigma point generation (including weights)
"""
Base.@kwdef struct SigmaPoints{T}
    points   :: Matrix{T}
    weights  :: SigmaWeights
end

function SigmaPoints(x::AbstractVector{T}, L::LowerTriangular, w::SigmaWeights) where T<:Real
    A = sqrt(w.c)*L
    points = [(x) (x .+ A) (x .- A)]
    return SigmaPoints(points=points, weights=w)
end
SigmaPoints(M::AbstractMatrix{T}, w::SigmaWeights) where T = SigmaPoints{T}(M, w)
SigmaPoints(x::AbstractVector{T}, R::UpperTriangular, w::SigmaWeights) where T<:Real = SigmaPoints(x, R', w)
SigmaPoints(x::AbstractVector{T}, C::Cholesky, w::SigmaWeights) where T<:Real = SigmaPoints(x, C.L, w)



"""
In-place state prediction with result-checking; returns intermediate results for troubleshooting
"""
function predict_state!(SS::StateSpaceModel{<:Real, <:Any, <:Any}, u; multithreaded=false)
    #Propagate sigma points through transition
    (xh, Ph) = predict_state(SS, u, multithreaded=multithreaded)

    #Overwrite the current state space model
    if any(isnan, xh) | any(isnan, Ph.U)
        @warn "predict_state! Warning: NaN detected in state, skipping"
    else
        SS.x  .= xh 
        SS.PU .= Ph.U
    end
    return (xh=xh, Ph=Ph)
end

"""
Predict the observation given a state
"""
predict_observation(SS::StateSpaceModel{<:Real, <:Any, <:LinearPredictor}, u) = SS.hxu[1]*SS.x + SS.hxu[2]*u
predict_observation(SS::StateSpaceModel{<:Real, <:Any, <:Function}, u) = SS.hxu(SS.x, u)

"""
In-place state update with automatic handling of missing observations; returns intermediate results for troubleshooting
"""
function update_state!(SS::StateSpaceModel{T, <:Any, <:Any}, y, u; multithreaded=false, outlier=3.0) where T <: Real
    
    if !all(isfinite, y) #Remove NaNs/Infs from the observations and model and call again
        ind = isfinite.(y)
        SR  = reduce_observer(SS, ind)
        return update_state!(SR, y[ind], u, multithreaded=multithreaded, outlier=outlier)
   
    elseif isempty(y) #No valid observations
        @warn "update_state! Warning: no observations, skipping update"
        TR = promote_type(T,Float64)
        return (yh=TR[], K=zeros(TR, length(SS.x), 0))
    end

    #Update state space model objects
    OBS = update_state(SS, y, u, multithreaded=multithreaded, outlier=outlier)
    if any(isnan, OBS.xh) | any(isnan, OBS.Ph.U)
        @warn "update_state! Warning: NaN detected in state, skipping"
    else
        SS.x  .= OBS.xh
        SS.PU .= OBS.Ph.U
    end
    return (yh=OBS.yh, K=OBS.K)
end

"""
Nonlinear state prediction
"""
function predict_state(SS::StateSpaceModel{<:Real, <:Function, <:Any}, u; multithreaded=false)
    #Propagate sigma points through transition
    w = SigmaWeights(SS)
    𝒳t = SigmaPoints(SS.x, SS.PU, w)
    𝒳t = predict!(SS.fxu, 𝒳t, u, multithreaded=multithreaded)
    
    #Obtain prediction covariance
    xh = mean(𝒳t)
    Ph = chol_update(SS.QU, subtract(𝒳t, xh))
    return (xh=xh, Ph=Ph)
end

"""
Linear state prediction
"""
function predict_state(SS::StateSpaceModel{<:Real, <:LinearPredictor, <:Any}, u; multithreaded=false)
    (A, B) = (SS.fxu[1], SS.fxu[2])
    xh = A*SS.x + B*u
    Ph = Cholesky(root_sum_squared(SS.PU*A', SS.QU), :U, 0)
    return (xh=xh, Ph=Ph)
end


"""
Nonlinear state updating
"""
function update_state(SS::StateSpaceModel{<:Real, <:Any, <:Function}, y, u; multithreaded=false, outlier=3.0)
    w = SigmaWeights(SS)

    #Propagate new predicted sigma points though observation
    𝒳 = SigmaPoints(SS.x, SS.PU, w)
    𝒴 = predict(SS.hxu, 𝒳, u, multithreaded=multithreaded)
    yh = mean(𝒴)
    
    S = chol_update(SS.RU, subtract(𝒴, yh)) #Obtain cholesky of the innovation covariance
    Pxy = cov(𝒳, 𝒴) #Obtain cross-covariance of state and measurement innovations
    K = (Pxy/(S.U))/S.L #Kalman gain

    σz = chol_std(S)
    xh = SS.x .+ K*scale_innovation.(y.-yh, σz, outlier=outlier)
    Ph = chol_update!(Cholesky(copy(SS.PU),:U,0), K*S.L, -1)

    return (xh=xh, Ph=Ph, yh=yh, K=K)
end

"""
Linear state updating
"""
function update_state(SS::StateSpaceModel{<:Real, <:Any, <:LinearPredictor}, y, u; multithreaded=false, outlier=3.0)
    (C, D) = (SS.hxu[1], SS.hxu[2])
    yh = C*SS.x .+ D*u

    S = Cholesky(root_sum_squared(SS.PU*C', SS.RU)) #Obtain cholesky of the innovation covariance
    Pxy = (SS.PU'*SS.PU)*C' #Obtain cross-covariance of state and measurement innovations
    K = (Pxy/(S.U))/S.L #Kalman gain

    σz = chol_std(S)
    xh = SS.x .+ K*scale_innovation.(y.-yh, σz, outlier=outlier)
    Ph = root_sum_squared(SS.PU*(I-K*C)', SS.RU*K')

    return (xh=xh, Ph=Cholesky(Ph,:U,0), yh=yh, K=K)
end


"""
Retruns a state space model with a reduced observer space determined by the second argument "not_missing", 
This is useful for handling missing data (where the non-missing elements are defined by "not_missing" )
"""
function reduce_observer(SS::StateSpaceModel{T, <:Any, <:Function}, not_missing) where T
    reduced_obsfunc(x,u) = SS.hxu(x,u)[not_missing]
    return StateSpaceModel{T, typeof(SS.fxu), typeof(reduced_obsfunc)}(
        fxu = SS.fxu,
        hxu = reduced_obsfunc,
        x  = SS.x,
        QU = SS.QU,
        RU = UpperTriangular(SS.RU[not_missing, not_missing]),
        PU = SS.PU,
        θ  = SS.θ
    )
end

function reduce_observer(SS::StateSpaceModel{T, <:Any, <:LinearPredictor}, not_missing) where T
    return StateSpaceModel{T, typeof(SS.fxu), typeof(SS.hxu)}(
        fxu = SS.fxu,
        hxu = (SS.hxu[1][not_missing,:], SS.hxu[2][not_missing,:]),
        x  = SS.x,
        QU = SS.QU,
        RU = UpperTriangular(SS.RU[not_missing, not_missing]),
        PU = SS.PU,
        θ  = SS.θ
    )
end



"""
In-place modification of sigma points 𝒳 by applying f to each element. Can be multithreaded if f is computationally intense
"""
function predict!(f::Function, 𝒳::SigmaPoints{T}, u; multithreaded=false) where T
    X = 𝒳.points
    if multithreaded
        Threads.@threads for ii in axes(X,2)
            X[:,ii] = f(view(X,:,ii), u)
        end
    else
        for ii in axes(X,2)
            X[:,ii] = f(view(X,:,ii), u)
        end
    end
    return 𝒳
end

"""
Produces a new set of sigma points by applying f to all sigma points in 𝒳 Can be multithreaded if f is computationally intense
"""
function predict(f::Function, 𝒳::SigmaPoints{T}, u; multithreaded=false) where T
    if multithreaded
        predictions = [Threads.@spawn f(x, u) for x in eachcol(𝒳.points)]
        return SigmaPoints(points = mapreduce(fetch, hcat, predictions), weights = 𝒳.weights)
    else
        predictions = (f(x, u) for x in eachcol(𝒳.points))
        return SigmaPoints(points = reduce(hcat, predictions), weights = 𝒳.weights)
    end
end

"""
Returns √(A² + B²) for matrices A and B where √M is the upper triangular square-root of matrix M
"""
function root_sum_squared(A::AbstractMatrix, B::AbstractMatrix)
    R = UpperTriangular(qr!([A;B]).R)
    if any(c->R[c]<0, diagind(R))
        R .= .-R
    end
    return R
end


"""
Produces a new set of sigma points by subtracting x from each point
"""
function subtract(𝒳::SigmaPoints, x::AbstractVector)
    return SigmaPoints(
        points  = 𝒳.points .- x,
        weights = 𝒳.weights
    )
end


"""
Returns a weighted mean vector of a set of sigma points
"""
function mean(𝒳::SigmaPoints{T}) where T
    weight(ii::Integer) = ifelse(ii==1, 𝒳.weights.μ[1], 𝒳.weights.μ[2])

    μ  = zeros(promote_type(T, Float64), size(𝒳.points,1))
    ii = 0
    for x in eachcol(𝒳.points)
        ii += 1
        μ .+= weight(ii) .* x
    end
    return μ
end


"""
Returns a weighted covariance matrix of two sets of sigma points, based on weights from the first set
"""
function cov(𝒳::SigmaPoints{T1}, 𝒴::SigmaPoints{T2}; centered=false) where {T1, T2}
    weight(ii::Integer) = ifelse(ii==1, 𝒳.weights.Σ[1], 𝒳.weights.Σ[2])

    if size(𝒳.points, 2) != size(𝒴.points, 2)
        error("Two sets of sigma points must have the same number of points")
    end

    #Caclulate the mean
    (μx, μy) = centered ? (0.0, 0.0) : (mean(𝒳), mean(𝒴))
    
    #Fill out the covariance matrix
    T = promote_type(Float64, T1, T2)
    S = zeros(T, size(𝒳.points,1), size(𝒴.points,1))
    ii = 0
    for (x, y) in zip(eachcol(𝒳.points), eachcol(𝒴.points))
        ii += 1
        S .+= weight(ii) .* (x.-μx) .* (y.-μy)'
    end
    return S
end

cov(𝒳::SigmaPoints) = cov(𝒳,𝒳)


#Updating of cholesky objects
"""
Returns the cholesky decopmposition of the sum of two covariance matrices S1, S2 where
S1 is given as a Cholesky Decomposition
S2 is given as a set of sigma points
"""
function chol_update(S1::Cholesky, 𝒮2::SigmaPoints)
    return chol_update(S1.L, 𝒮2)
end

#The QR decomposition method for updating cholesky decompositions with sigma points stems from some very clever Linear Algebra
#Our target is to calculate the cholesky decomposition of S3 = S1 + S2 where S2 is given as sigma points (𝒮2)
#S2 = (sqrt(w2) 𝒮2)*(sqrt(w2) 𝒮2)' where 𝒮2 are the sigma points of S2 and w2 are the weights
#S2 = A*A' = [(sqrt(w2)*𝒮2)  √S1]*[(sqrt(w2)*𝒮2)  √S1]' where √S1 is the left-square root (lower triangular of the cholesky)
#By doing QR decomposition on A' so that A' = QₐRₐ, (where Qₐ is an orthogonal matrix, Qₐ*Qₐ'=I) we get
#S2 = (QₐRₐ)'(QₐRₐ) = (Rₐ'Qₐ')(QₐRₐ) = Rₐ'Rₐ
#Thus, Rₐ is the upper-right cholesky decomposition of S3
#The reason why the central sigma point left out of the QR decompodition is that w0 can be negative, which throws everything off 
#The first sigma point is therefore accounted for though the chol_update! (low-rank downdate) which can handle negative w0

function chol_update(SL1::LowerTriangular, 𝒮2::SigmaPoints)
    (w0, w1) = (𝒮2.weights.Σ[1], 𝒮2.weights.Σ[2])
    A   = @views [(sqrt(w1).*𝒮2.points[:, 2:end]) SL1]
    SC⁺ = Cholesky(qr(A').R, :U, 0)
    return chol_update!(SC⁺, view(𝒮2.points,:,1), w0)
end

chol_update(SR1::UpperTriangular, 𝒮2::SigmaPoints) = chol_update(SR1', 𝒮2)


function chol_update!(ch::Cholesky, x::AbstractVector, w::Real)
    if w >= 0
        return lowrankupdate!(ch, sqrt(w)*x)
    else
        return lowrankdowndate!(ch, sqrt(abs(w))*x)
    end
end

function chol_update!(ch::Cholesky, X::AbstractArray{<:Real,2}, w::Real)
    for x in eachcol(X)
        chol_update!(ch, x, w)
    end
    return ch
end

function chol_std(ch::Cholesky)
    selfdot(x) = dot(x,x)
    return [sqrt(selfdot(col)) for col in eachcol(ch.U)]
end

#Option to limit transition differences on the state (to avoid chasing outlisers)
function limit_diff!(SS::StateSpaceModel, xh; Sigmas=10.0)
    Δx = Sigmas.*diag(SS.sQ) 
    SS.x .= clamp.(SS.x, xh.-Δx, xh.+Δx)
    return SS
end

#Scale the innoviation to avoid chasing outliers
function scale_innovation(Δy::Real, σy::Real; outlier)
    σε = (outlier/3)*σy
    return asinh(Δy/σε)*σε
end





# =========================================================================================================
# Legacy code
# =========================================================================================================
#=
#Sigma points around reference point
function sigmas(x0::AbstractVector, L::LowerTriangular, c)
    x = copy(x0)
    N = length(x0)
    A = sqrt(c)*L

    x_plus_A  = [x + A[:,k] for k in 1:N]
    x_minus_A = [x - A[:,k] for k in 1:N]
    return [[x]; x_plus_A; x_minus_A]
end

#Weights of sigma points
function sigma_weights(SS::StateSpaceModel)
    α = SS.α
    κ = SS.κ
    β = SS.β
    L = length(SS.x)

    λ = α^2*(L+κ)-L                           #scaling factor
    c = L + λ                                 #scaling factor
    W  = fill( 0.5/c , 2*L )
    Wm = [(λ/c) ; W]                          #weights for means
    Wc = [Wm[1] + (1-α^2+β) ; W]              #weights for covariance

    return c, Wm, Wc
end




#Multithreaded function sampling
function sample_func(f::Function, X::Vector)
    FuncTasks = [Threads.@spawn f(x) for x in X]
    return fetch.( FuncTasks)
end

#Weighted statistics
function sigma_mean(X::Vector, Wm::Vector{<:Real})
    return sum(Wm .* X)
end

#Uses unscented method transform to produce a covariance
function sigma_cov(ΔX::Vector, ΔY::Vector, Wc::Vector{<:Real})
    CovTasks = [Threads.@spawn Wc[ii]*ΔX[ii]*ΔY[ii]' for ii in 1:length(Wc) ]
    return sum( fetch.(CovTasks) )
end

#Uses outputs a cholseky decomposition of sigma points plus a diagonal covariance square root
function chol_sigmas_plus_cov(ΔX::Vector, Wc::Vector{<:Real}, sQ::LowerTriangular)
    N = length(ΔX[1])

    #Return first square of the QR factorization of this wide matrix
    sigmaTall = Array( [hcat( sqrt(Wc[2]).*ΔX[2:end]... ) sQ]' )
    ch = Cholesky(qr( sigmaTall ).R, :U, 0)
    return chol_update!(ch, ΔX[1], Wc[1])
end
=#