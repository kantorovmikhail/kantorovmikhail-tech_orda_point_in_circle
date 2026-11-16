package com.example.circle.engine;
import com.example.circle.training.TrainingConfig.*;
import java.util.Random;

/** Retains the reference layer's ownership of an array of neurons. */
public final class DenseLayer {
    public final Neuron[] neurons;
    DenseLayer(int inputs,Layer config,Weights initialization,Random random) {
        neurons=new Neuron[config.neurons()];
        for(int i=0;i<neurons.length;i++) neurons[i]=new Neuron(inputs,config,initialization,random);
    }
    void activate(double[] input,double[] z,double[] output) {
        for(int i=0;i<neurons.length;i++) { z[i]=neurons[i].sum(input); output[i]=neurons[i].activation.forward(z[i]); }
    }
}
