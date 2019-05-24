# Rubiks Solver

## Background

I scrambled a friend's Rubik's cube, and when I went to solve it, I
found out that I am pretty bad at following instructions for solving a
Rubik's cuber. So, in order to get his cube back in order, I decided
that I would just implement a Rubik's cube solver and let it tell me
each and every step that I would need to perform in order to solve the
cube.

## Status

My aspiration for this project is to implement a solver that provides
step-by-step graphical solution instructions along with some camera
tools to recognize the initial state of the cube (so that I won't have
to manually enter the cube's state). However, as of May 24th 2019, I
haven't implemented an entry method, and I haven't implemented a way
to output the intermediate steps / plot the steps. However, I have
successfully implemented the solver (using [this
method](https://www.rubiks.com/how-to-solve-rubiks-cube)), and some
plotting functions to display the cube's state. I just need to add a
bit on the output / completely add the input and things should be good
(:

## Example: Solving a random initial state.

As an example, I put the cube into a randomized initial state and let
the solver run, displaying the cube's state after each major solution
step. The solution is pretty similar to [this
method](https://www.rubiks.com/how-to-solve-rubiks-cube).

<p float="center">
<img src="./img/step_0_initial_state.png" width="300">
<img src="./img/step_1_white_cross.png" width="300">
<img src="./img/step_2_white_corners.png" width="300">
<img src="./img/step_3_middle_edges.png" width="300">
<img src="./img/step_4_yellow_cross.png" width="300">
<img src="./img/step_5_yellow_corners.png" width="300">
<img src="./img/step_6_permute_corners.png" width="300">
<img src="./img/step_7_permute_edges.png" width="300">
</p>
