package com.example.circle.engine;
import com.example.circle.training.TrainingConfig.*;
import java.util.Random;

/** Adapted from module_1: one neuron owns its weights, activation and optional trainable bias. */
public final class Neuron {
    public final double[] weights;
    public double bias;
    final Activation activation;
    final boolean biasEnabled;
    public Neuron(int inputs, Layer config, Weights initialization, Random random) {
        activation=config.activation(); biasEnabled=config.bias().enabled(); weights=new double[inputs];
        for(int i=0;i<inputs;i++) weights[i]=switch(initialization) {
            case XAVIER -> (random.nextDouble()*2-1)*Math.sqrt(6.0/(inputs+config.neurons()));
            case HE -> random.nextGaussian()*Math.sqrt(2.0/inputs);
            case UNIFORM -> random.nextDouble()-.5;
        };
        bias=!biasEnabled?0:switch(config.bias().initialization()) {
            case ZERO -> 0;
            case CONSTANT -> config.bias().value();
            case RANDOM -> config.bias().value()+(2*random.nextDouble()-1)*config.bias().scale();
        };
    }
    double sum(double[] input) {
        double z=biasEnabled?bias:0;
        for(int i=0;i<weights.length;i++) z+=weights[i]*input[i];
        return z;
    }
    public double activate(double[] input) {
        if(input.length!=weights.length) throw new IllegalArgumentException("Input size");
        return activation.forward(sum(input));
    }
    public void update(double[] input,double delta,double rate) {
        for(int i=0;i<weights.length;i++) weights[i]-=rate*delta*input[i];
        if(biasEnabled) bias-=rate*delta; // Exactly once, outside the weight loop.
    }
}
