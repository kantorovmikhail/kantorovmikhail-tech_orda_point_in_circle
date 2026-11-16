package com.example.circle.engine;
import com.example.circle.training.TrainingConfig.Network;
import java.util.*;

/** Single-thread owned educational SGD network; buffers are reused for every sample. */
public final class NeuralNetwork {
    public final DenseLayer[] layers;
    private final Network config;
    private final double[][] a,z,delta;
    public NeuralNetwork(Network config) {
        this.config=config; layers=new DenseLayer[config.layers().size()];
        a=new double[layers.length+1][]; z=new double[layers.length][]; delta=new double[layers.length][];
        a[0]=new double[2]; Random random=new Random(config.seed()); int inputs=2;
        for(int l=0;l<layers.length;l++) {
            layers[l]=new DenseLayer(inputs,config.layers().get(l),config.initialization(),random);
            inputs=layers[l].neurons.length; a[l+1]=new double[inputs]; z[l]=new double[inputs]; delta[l]=new double[inputs];
        }
    }
    public double probability(double x,double y) {
        if(!Double.isFinite(x)||!Double.isFinite(y)) throw new IllegalArgumentException("Coordinates must be finite");
        a[0][0]=x; a[0][1]=y;
        for(int l=0;l<layers.length;l++) layers[l].activate(a[l],z[l],a[l+1]);
        double p=a[layers.length][0];
        if(!Double.isFinite(p)) throw new ArithmeticException("Non-finite model output");
        return p;
    }
    public int predict(double x,double y,double threshold) { return probability(x,y)>=threshold?1:0; }
    public double loss(double x,double y,int target) {
        probability(x,y); double logit=z[layers.length-1][0];
        return Math.max(logit,0)-logit*target+Math.log1p(Math.exp(-Math.abs(logit)));
    }
    public double trainSample(double x,double y,int target,double rate) {
        if((target!=0&&target!=1)||!Double.isFinite(rate)||rate<=0) throw new IllegalArgumentException("Invalid SGD parameters");
        double loss=loss(x,y,target); int last=layers.length-1;
        delta[last][0]=a[last+1][0]-target;
        // Compute ALL deltas before mutating ANY weights.
        for(int l=last-1;l>=0;l--) for(int j=0;j<delta[l].length;j++) {
            double sum=0;
            for(int k=0;k<delta[l+1].length;k++) sum+=layers[l+1].neurons[k].weights[j]*delta[l+1][k];
            delta[l][j]=sum*layers[l].neurons[j].activation.derivative(z[l][j],a[l+1][j]);
        }
        for(int l=0;l<layers.length;l++) for(int j=0;j<delta[l].length;j++) layers[l].neurons[j].update(a[l],delta[l][j],rate);
        return loss;
    }
    public record Parameters(double[][][] weights,double[][] biases) {}
    public Parameters parameters() {
        double[][][] w=new double[layers.length][][]; double[][] b=new double[layers.length][];
        for(int l=0;l<layers.length;l++) {
            w[l]=new double[layers[l].neurons.length][]; b[l]=new double[w[l].length];
            for(int j=0;j<w[l].length;j++) { w[l][j]=layers[l].neurons[j].weights.clone(); b[l][j]=layers[l].neurons[j].bias; }
        }
        return new Parameters(w,b);
    }
    public void restore(Parameters p) {
        if(p==null || p.weights().length!=layers.length || p.biases().length!=layers.length) throw new IllegalArgumentException("Invalid model dimensions");
        for(int l=0;l<layers.length;l++) {
            if(p.weights()[l].length!=layers[l].neurons.length || p.biases()[l].length!=layers[l].neurons.length) throw new IllegalArgumentException("Invalid layer dimensions");
            for(int j=0;j<layers[l].neurons.length;j++) {
                Neuron n=layers[l].neurons[j]; double[] w=p.weights()[l][j];
                if(w.length!=n.weights.length || !Double.isFinite(p.biases()[l][j])) throw new IllegalArgumentException("Invalid neuron parameters");
                for(double v:w) if(!Double.isFinite(v)) throw new IllegalArgumentException("Non-finite weights");
                System.arraycopy(w,0,n.weights,0,w.length); n.bias=n.biasEnabled?p.biases()[l][j]:0;
            }
        }
    }
    public NeuralNetwork copy() { NeuralNetwork n=new NeuralNetwork(config); n.restore(parameters()); return n; }
    public record LayerMetric(int layer,double meanActivation,double activationStd,double weightNorm,
                              Double meanBias,Double biasStd,Double zeroRatio,Double saturationRatio) {}
    public List<LayerMetric> layerMetrics(CircleDataset data) {
        int n=layers.length; double[] sum=new double[n],sq=new double[n],zero=new double[n],sat=new double[n];
        int count=Math.min(data.size(),2048);
        for(int i=0;i<count;i++) {
            int index=(int)((long)i*data.size()/count); probability(data.x(index),data.y(index));
            for(int l=0;l<n;l++) for(double v:a[l+1]) { sum[l]+=v; sq[l]+=v*v; if(v==0)zero[l]++; if(Math.abs(v)>.98)sat[l]++; }
        }
        List<LayerMetric> result=new ArrayList<>();
        for(int l=0;l<n;l++) {
            double w=0,b=0,bs=0,den=(double)count*layers[l].neurons.length;
            for(Neuron neuron:layers[l].neurons) { for(double v:neuron.weights)w+=v*v; b+=neuron.bias; bs+=neuron.bias*neuron.bias; }
            int size=layers[l].neurons.length; double mean=sum[l]/den, bm=b/size; Neuron first=layers[l].neurons[0];
            double saturation=sat[l]/den;
            if(first.activation==Activation.SIGMOID) { // Include both tails for sigmoid.
                int tails=0;
                for(int i=0;i<count;i++) {int index=(int)((long)i*data.size()/count);probability(data.x(index),data.y(index)); for(double v:a[l+1])if(v<.02||v>.98)tails++;}
                saturation=tails/den;
            }
            result.add(new LayerMetric(l,mean,Math.sqrt(Math.max(0,sq[l]/den-mean*mean)),Math.sqrt(w),first.biasEnabled?bm:null,first.biasEnabled?Math.sqrt(Math.max(0,bs/size-bm*bm)):null,first.activation==Activation.RELU?zero[l]/den:null,(first.activation==Activation.TANH||first.activation==Activation.SIGMOID)?saturation:null));
        }
        return result;
    }
    /** Last training sample's full parameter-gradient norm, before its SGD update. */
    public List<Double> gradientNorms() {
        List<Double> result=new ArrayList<>();
        for(int l=0;l<layers.length;l++) { double norm=0; for(int j=0;j<delta[l].length;j++) {double d=delta[l][j]; if(layers[l].neurons[j].biasEnabled)norm+=d*d; for(double v:a[l])norm+=d*d*v*v;} result.add(Math.sqrt(norm)); }
        return result;
    }
}
