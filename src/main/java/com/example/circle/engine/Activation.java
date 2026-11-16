package com.example.circle.engine;
public enum Activation {
    LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU;
    public double forward(double z) {
        return switch(this) {
            case LINEAR -> z;
            case SIGMOID -> z>=0 ? 1/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z));
            case TANH -> Math.tanh(z);
            case RELU -> Math.max(0,z);
            case LEAKY_RELU -> z>=0 ? z : .01*z;
        };
    }
    public double derivative(double z,double output) {
        return switch(this) {
            case LINEAR -> 1;
            case SIGMOID -> output*(1-output);
            case TANH -> 1-output*output;
            case RELU -> z>0?1:0;
            case LEAKY_RELU -> z>0?1:.01;
        };
    }
}
