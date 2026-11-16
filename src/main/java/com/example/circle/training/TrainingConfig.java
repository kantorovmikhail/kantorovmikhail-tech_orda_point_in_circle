package com.example.circle.training;

import com.example.circle.engine.Activation;
import com.fasterxml.jackson.annotation.JsonFormat;
import java.util.List;

/** Immutable effective experiment configuration, bound from YAML and REST JSON. */
public record TrainingConfig(Circle circle, Network network, Datasets datasets, Training training,
                             double threshold, Visualization visualization, String resultsDirectory) {
    public record Circle(double x, double y, double radius, double tolerance) {}
    public enum Initialization { ZERO, RANDOM, CONSTANT }
    public enum Weights { XAVIER, HE, UNIFORM }
    public record Bias(boolean enabled, Initialization initialization, double value, double scale) {}
    public record Layer(int neurons, Activation activation, Bias bias) {}
    public record Network(int inputSize, @JsonFormat(shape=JsonFormat.Shape.STRING) long seed, Weights initialization, List<Layer> layers) {
        public Network { layers = layers == null ? null : List.copyOf(layers); }
    }
    public enum Strategy { UNIFORM, BOUNDARY_FOCUSED, HYBRID, GRID }
    public enum Distribution { GAUSSIAN, UNIFORM }
    public record Dataset(int sampleCount, @JsonFormat(shape=JsonFormat.Shape.STRING) long seed, double minX, double maxX, double minY, double maxY,
                          Strategy strategy, double step, double boundaryFraction, double boundaryWidth,
                          Distribution distribution, int exactBoundarySamples) {}
    public record Datasets(Dataset training, Dataset validation, Dataset test) {}
    public record Training(int epochs, double learningRate, boolean shuffle, @JsonFormat(shape=JsonFormat.Shape.STRING) long seed, int metricsEvery) {}
    public record Visualization(int resolution, int maxPoints) {}
    public static TrainingConfig defaults() {
        Bias b = new Bias(true, Initialization.ZERO, 0, .1);
        return new TrainingConfig(new Circle(0,0,5,.01),
            new Network(2,42,Weights.XAVIER,List.of(new Layer(16,Activation.TANH,b),new Layer(16,Activation.TANH,b),new Layer(1,Activation.SIGMOID,b))),
            new Datasets(dataset(3000,101),dataset(1000,202),dataset(2000,303)),
            new Training(100,.01,true,42,1), .5, new Visualization(80,600), "results");
    }
    public static Dataset dataset(int n, long seed) { return new Dataset(n,seed,-7,7,-7,7,Strategy.HYBRID,0,.5,.7,Distribution.GAUSSIAN,0); }
    public void validate() {
        require(circle != null && network != null && datasets != null && training != null && visualization != null,"Missing configuration section");
        require(finite(circle.x,circle.y,circle.radius,circle.tolerance,circle.radius*circle.radius) && Math.abs(circle.x)<1e100 && Math.abs(circle.y)<1e100 && circle.radius>0 && circle.tolerance>=0,"Invalid circle");
        require(network.inputSize==2 && network.initialization!=null && network.layers!=null && !network.layers.isEmpty() && network.layers.size()<=32,"Invalid network");
        long parameters=0; int inputs=2;
        for (Layer l:network.layers) {
            require(l!=null && l.neurons>0 && l.neurons<=2048 && l.activation!=null && l.bias!=null,"Invalid layer");
            require(l.bias.initialization!=null && finite(l.bias.value,l.bias.scale) && l.bias.scale>=0,"Invalid bias");
            parameters+=(long)inputs*l.neurons; inputs=l.neurons;
        }
        require(parameters<=2_000_000,"Network too large");
        Layer out=network.layers.getLast(); require(out.neurons==1 && out.activation==Activation.SIGMOID,"Output must be one sigmoid neuron for BCE");
        require(training.epochs>0 && training.epochs<=100000 && finite(training.learningRate) && training.learningRate>0 && training.learningRate<=1 && training.metricsEvery>0,"Invalid training parameters");
        require(finite(threshold) && threshold>0 && threshold<1,"Invalid threshold");
        require(visualization.resolution>=10 && visualization.resolution<=250 && visualization.maxPoints>=0 && visualization.maxPoints<=5000,"Invalid visualization limits");
        require(resultsDirectory!=null && !resultsDirectory.isBlank(),"Missing results directory");
        validateDataset(datasets.training); validateDataset(datasets.validation); validateDataset(datasets.test);
        require(datasets.training.seed!=datasets.validation.seed && datasets.training.seed!=datasets.test.seed && datasets.validation.seed!=datasets.test.seed,"Dataset seeds must differ");
    }
    private void validateDataset(Dataset d) {
        require(d!=null && d.strategy!=null && d.distribution!=null,"Missing dataset configuration");
        require(finite(d.minX,d.maxX,d.minY,d.maxY,d.step,d.boundaryFraction,d.boundaryWidth) && Math.abs(d.minX)<1e100 && Math.abs(d.maxX)<1e100 && Math.abs(d.minY)<1e100 && Math.abs(d.maxY)<1e100 && d.minX<d.maxX && d.minY<d.maxY && d.step>=0 && d.boundaryWidth>0 && d.boundaryFraction>=0 && d.boundaryFraction<=1,"Invalid dataset bounds or sampling parameters");
        require(d.sampleCount>0 && d.sampleCount<=10_000_000 && d.exactBoundarySamples>=0 && d.exactBoundarySamples<=d.sampleCount,"Invalid sample count");
        require(d.strategy!=Strategy.GRID || (d.step>0 && d.exactBoundarySamples==0),"Grid requires positive step and no extra boundary samples");
        if(d.exactBoundarySamples>0) require(d.minX<=circle.x-circle.radius && d.maxX>=circle.x+circle.radius && d.minY<=circle.y-circle.radius && d.maxY>=circle.y+circle.radius,"Bounds must contain explicit boundary points");
    }
    public static boolean finite(double... values) { for(double v:values) if(!Double.isFinite(v)) return false; return true; }
    public static void require(boolean condition,String message) { if(!condition) throw new IllegalArgumentException(message); }
}
