package com.example.circle.training;
import com.example.circle.engine.NeuralNetwork.LayerMetric;
import java.time.Instant;
import java.util.*;
public record TrainingSnapshot(UUID id,State state,int currentEpoch,int totalEpochs,Instant startedAt,
                               Instant finishedAt,double durationSeconds,List<Metric> metrics,
                               Scores testResults,String error,String resultFile) {
    public enum State { IDLE,RUNNING,STOPPING,COMPLETED,CANCELLED,FAILED }
    public record Scores(double loss,double accuracy,double precision,double recall,double f1,int samples) {}
    public record Metric(int epoch,double learningRate,double durationSeconds,Scores training,Scores validation,
                         List<LayerMetric> layers,List<Double> gradientNorms) {
        public Metric {layers=List.copyOf(layers);gradientNorms=List.copyOf(gradientNorms);}
    }
    public TrainingSnapshot { metrics=List.copyOf(metrics); }
    public boolean active(){return state==State.RUNNING||state==State.STOPPING;}
}
