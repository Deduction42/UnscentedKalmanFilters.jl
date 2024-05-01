

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

include(joinpath(@__DIR__, "_StateSpaceModel.jl"))

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

const Δt = 0.1
ω  = 2π/50
N  = 1000
σ  = 0.3
y  = sin.((1:N).*ω) .+ σ*randn(N)
k0 = 100*(ω)^2

#Test spike
y[150] = 15

#Test missing data after stabilization
dy = [0; diff(y)]./Δt
Y  = [dy'; y']
Y[:,200] .= NaN
Y[1,500] = NaN
Y[2,501] = NaN

function oscillator_prediction(X, u)
    k = exp(X[3])
    A = [
        0  -k   0;
        1   0   0;
        0   0   0 
    ]
    return exp(A*Δt)*X #Discretize A and then predict result
end

#These methods should yield the same result
#oscillator_observation(X, u) = X[1:2]
oscillator_observation = ([1 0 0 ; 0 1 0], zeros(2,0))


σ₊  = (σ+0.1)
vsQ = [0.01*σ₊/Δt, 0.0001*σ₊, 0.1]
vsR = [2*σ₊/Δt, σ₊]
vsP = [10*σ₊/Δt, 10*σ₊, 10]

model = StateSpaceModel(
    fxu = oscillator_prediction,
    hxu = oscillator_observation,
    x  = [0, 0, log(k0)],
    QU = Diagonal(vsQ),
    RU = Diagonal(vsR),
    PU = Diagonal(vsP),
)
#=
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
=#

vs = [GaussianState(model)]

for ii in 1:N
    kalman_filter!(model, Y[:,ii], Float64[])
    push!(vs, GaussianState(model))
end

using PythonPlot; pygui(true)
figure()
plot(y, ".k")
plot([s.x[1] for s in vs[1:(end-1)]])
plot([s.x[2] for s in vs[1:(end-1)]])
plot([sqrt( min(5*σ, s.x[1]^2/exp(s.x[3])) + s.x[2]^2) for s in vs[1:(end-1)]]) #amplitude-equivalent energy
legend(["measured", "velocity", "position", "energy amplitude"])
title("Frequency Tracking Summary: UKF")

figure()
title("Frequency Tracking Raw State: UKF")
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

