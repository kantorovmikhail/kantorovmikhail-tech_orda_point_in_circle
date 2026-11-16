package com.example.circle.engine;
import com.example.circle.training.TrainingConfig;
import com.example.circle.training.TrainingConfig.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
class NeuralNetworkTest {
    static final Bias BIAS=new Bias(true,Initialization.ZERO,0,.1);
    static Network config(){return new Network(2,42,Weights.XAVIER,List.of(new Layer(3,Activation.TANH,BIAS),new Layer(2,Activation.TANH,BIAS),new Layer(1,Activation.SIGMOID,BIAS)));}
    @Test void activationsAndFiniteDifferenceDerivatives(){
        for(Activation activation:Activation.values())for(double z:new double[]{-3,-.3,.4,3}){
            double epsilon=1e-6, numerical=(activation.forward(z+epsilon)-activation.forward(z-epsilon))/(2*epsilon);
            assertEquals(numerical,activation.derivative(z,activation.forward(z)),1e-8,activation.toString());
        }
        assertEquals(1,Activation.SIGMOID.forward(1000));assertEquals(0,Activation.SIGMOID.forward(-1000));
        assertEquals(.01,Activation.LEAKY_RELU.derivative(-1,-.01));assertEquals(0,Activation.RELU.forward(-1));
    }
    @Test void neuronForwardAndBiasExactlyOnce(){
        Neuron n=new Neuron(2,new Layer(1,Activation.LINEAR,BIAS),Weights.UNIFORM,new Random(1));
        n.weights[0]=.5;n.weights[1]=-.25;n.bias=.1;
        assertEquals(.35,n.activate(new double[]{1,1}),1e-15);
        n.update(new double[]{1,2},.4,.1);
        assertEquals(.06,n.bias,1e-15);assertArrayEquals(new double[]{.46,-.33},n.weights,1e-15);
        Neuron disabled=new Neuron(2,new Layer(1,Activation.LINEAR,new Bias(false,Initialization.CONSTANT,99,0)),Weights.UNIFORM,new Random(1));
        Arrays.fill(disabled.weights,0);disabled.bias=22;assertEquals(0,disabled.activate(new double[]{1,1}));disabled.update(new double[]{1,1},1,.1);assertEquals(22,disabled.bias);
    }
    @Test void initializationOptions(){
        for(Initialization init:Initialization.values()){
            Bias b=new Bias(true,init,2,.2);Neuron n=new Neuron(2,new Layer(1,Activation.LINEAR,b),Weights.HE,new Random(1));
            if(init==Initialization.ZERO)assertEquals(0,n.bias);else if(init==Initialization.CONSTANT)assertEquals(2,n.bias);else assertTrue(n.bias>=1.8&&n.bias<=2.2);
        }
    }
    @Test void everyWeightAndBiasAcrossMultipleLayersPassesGradientCheck(){
        NeuralNetwork n=new NeuralNetwork(config());double x=.7,y=-.3,epsilon=1e-5,rate=.001;
        var before=n.parameters();double[][][] gradients=new double[n.layers.length][][];double[][] biasGradients=new double[n.layers.length][];
        for(int l=0;l<n.layers.length;l++){
            gradients[l]=new double[n.layers[l].neurons.length][];biasGradients[l]=new double[n.layers[l].neurons.length];
            for(int j=0;j<n.layers[l].neurons.length;j++){
                Neuron neuron=n.layers[l].neurons[j];gradients[l][j]=new double[neuron.weights.length];
                for(int k=0;k<neuron.weights.length;k++){
                    double value=neuron.weights[k];neuron.weights[k]=value+epsilon;double plus=n.loss(x,y,1);neuron.weights[k]=value-epsilon;double minus=n.loss(x,y,1);neuron.weights[k]=value;gradients[l][j][k]=(plus-minus)/(2*epsilon);
                }
                double b=neuron.bias;neuron.bias=b+epsilon;double plus=n.loss(x,y,1);neuron.bias=b-epsilon;double minus=n.loss(x,y,1);neuron.bias=b;biasGradients[l][j]=(plus-minus)/(2*epsilon);
            }
        }
        n.trainSample(x,y,1,rate);
        for(int l=0;l<n.layers.length;l++)for(int j=0;j<n.layers[l].neurons.length;j++){
            Neuron neuron=n.layers[l].neurons[j];
            assertEquals(biasGradients[l][j],(before.biases()[l][j]-neuron.bias)/rate,1e-8);
            for(int k=0;k<neuron.weights.length;k++)assertEquals(gradients[l][j][k],(before.weights()[l][j][k]-neuron.weights[k])/rate,1e-8);
        }
    }
    @Test void arbitraryDepthAndStableLoss(){
        for(int hidden:new int[]{0,1,5}){
            List<Layer> layers=new ArrayList<>();for(int i=0;i<hidden;i++)layers.add(new Layer(4,Activation.LEAKY_RELU,BIAS));layers.add(new Layer(1,Activation.SIGMOID,BIAS));
            NeuralNetwork n=new NeuralNetwork(new Network(2,1,Weights.XAVIER,layers));
            double loss=n.trainSample(1,2,1,.01);assertTrue(Double.isFinite(loss));assertTrue(n.probability(1,2)>=0&&n.probability(1,2)<=1);assertTrue(n.predict(1,2,.5)==0||n.predict(1,2,.5)==1);
        }
        NeuralNetwork n=new NeuralNetwork(new Network(2,1,Weights.XAVIER,List.of(new Layer(1,Activation.SIGMOID,BIAS))));
        n.layers[0].neurons[0].weights[0]=1000;n.layers[0].neurons[0].weights[1]=0;
        assertEquals(1000,n.loss(1,0,0));assertEquals(1000,n.loss(-1,0,1));
        assertThrows(IllegalArgumentException.class,()->n.probability(Double.NaN,0));
    }
}
