# UnscentedKalmanFilters
A lightweight implementation of Kalman unscented Kalman filtering with the following features
1. Fallback to linear methods for transition if the state predictor is linear
2. Fallback to linear methods for observations if the state observer is linear
3. Consistent use of the square-root form for improved numerical stability
4. Automatic observation space reduction to remove non-finite observations
5. Optinal multithreading if state transition or observation is computationally intensive