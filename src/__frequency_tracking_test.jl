

#=
#Attempt to use inverse notch filter approach
using ControlSystemsBase
using Symbolics
@variables ρ² Ω₀

A = [2cos(Ω₀) -1; 1 0]
B = [0;0]
C = [-(1+ρ²) ρ²]
D = [0]
Gd = ss(A,B,C,D,1.0)

tfd = tf([1, -(1+ρ²)cos(Ω₀), ρ²], [1, -2cos(Ω₀), 1], 1.0)
=#

using StaticArrays
using LinearAlgebra

#include(joinpath(@__DIR__, "_StateSpaceModel.jl"))



#Quick test using my own kalman filter
@kwdef struct GaussianState
    x :: Vector{Float64}
    P :: Hermitian{Float64, Matrix{Float64}}
end

@kwdef struct LinearStateSpaceModel
    A :: Matrix{Float64}
    B :: Matrix{Float64}
    C :: Matrix{Float64}
    Q :: Hermitian{Float64, Matrix{Float64}}
    R :: Hermitian{Float64, Matrix{Float64}}
end

using FiniteDifferences
function kalman_filter(s::GaussianState, y, u, Δt, m::LinearStateSpaceModel)
    x0 = oscillator_prediction(s.x, u, Δt)
    f(x) = oscillator_prediction(x, u, Δt)
    Ae = jacobian(central_fdm(5, 1), f, s.x)[1]
    P0 = Ae*s.P*Ae' + m.Q

    z  = y .- m.C*x0
    S  = Hermitian(m.C*P0*m.C' + m.R)
    K  = (P0*m.C')/S
    x1 = x0 + K*z
    P1 = (I-K*m.C)*P0
    return GaussianState(x1, Hermitian(0.5*(P1 + P1')))
end




function oscillator_prediction(X, u, Δt)
    k = exp(X[3])
    A = [
        0  -k   0;
        1   0   0;
        0   0   0 
    ]
    return exp(A*Δt)*X .- [0, 0, 0.1*Δt]
end

function oscillator_observation(X)
    return [X[2]]
end

ω  = 2π/50
N  = 1000
Δt = 0.1
σ  = 0.3
Y  = sin.((1:N).*ω) .+ σ*randn(N)
k0 = 1/(ω)^2

σ₊  = (σ+0.1)
vsQ = [0.1*ω, 0.1*σ₊, 0.1]
vsR = [σ]
vsP = [100*σ₊, 100*σ₊, 10]

model = LinearStateSpaceModel(
    A = zeros(3,3),
    B = zeros(3,1),
    C = [0 1 0],
    Q = Hermitian(Diagonal(vsQ.^2)),
    R = Hermitian(Diagonal(vsR.^2))
)

state = GaussianState(
    x = [0, 0, log(k0)],
    P = Hermitian(Diagonal(vsP.^2))
)


vs = [state]

for ii in 1:N
    newstate = kalman_filter(vs[ii], Y[ii], [0.0], Δt, model)
    push!(vs, newstate)
end

using PythonPlot; pygui(true)
figure()
plot(Y, ".k")
plot([s.x[1] for s in vs[1:(end-1)]])
plot([s.x[2] for s in vs[1:(end-1)]])
plot([sqrt( min(5*σ, s.x[1]^2/exp(s.x[3])) + s.x[2]^2) for s in vs[1:(end-1)]]) #amplitude-equivalent energy
legend(["measured", "velocity", "position", "energy amplitude"])
title("Frequency Tracking Summary: EKF")

figure()
title("Frequency Tracking Raw State: EKF")
labels = ["velocity", "position", "log spring"]
for ii in 1:3
    subplot(3,1,ii)
    plot([s.x[ii] for s in vs[1:(end-1)]])
    ylabel(labels[ii])
end

figure()
title("Frequency Tracking Uncertainty: EKF")
labels = ["velocity", "position", "spring"]
for ii in 1:3
    subplot(3,1,ii)
    plot([sqrt(s.P[ii,ii]) for s in vs[1:(end-1)]])
    ylabel(labels[ii])
end

