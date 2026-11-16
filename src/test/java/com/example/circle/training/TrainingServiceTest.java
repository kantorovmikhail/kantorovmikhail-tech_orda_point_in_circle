package com.example.circle.training;
import com.example.circle.engine.*;
import com.example.circle.training.TrainingConfig.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;
class TrainingServiceTest {
    @TempDir Path directory;
    TrainingConfig config(int epochs){var d=TrainingConfig.defaults();return new TrainingConfig(d.circle(),d.network(),new Datasets(TrainingConfig.dataset(1500,101),uniform(700,202),uniform(1500,303)),new Training(epochs,.01,true,42,5),d.threshold(),d.visualization(),directory.toString());}
    Dataset uniform(int n,long seed){return new Dataset(n,seed,-7,7,-7,7,Strategy.UNIFORM,0,.5,.7,Distribution.GAUSSIAN,0);}
    @Test void integrationLearnsIndependentCircleDataAndReloadsExactly() throws Exception {
        var config=config(100);
        try(var service=new TrainingService(config)){
            var result=service.start().get(30,TimeUnit.SECONDS);
            assertEquals(TrainingSnapshot.State.COMPLETED,result.state(),result.error());assertEquals(100,result.currentEpoch());
            assertTrue(result.testResults().accuracy()>=.94,"Uniform independent test acceptance >=94%: "+result.testResults());
            assertTrue(result.metrics().getLast().training().loss()<result.metrics().getFirst().training().loss());
            assertEquals(1500,result.testResults().samples());assertEquals(700,result.metrics().getLast().validation().samples());
            var split=CircleDataset.generate(config.circle(),config.datasets(),()->false);
            var n=new NeuralNetwork(config.network());var random=new Random(config.training().seed());int[] order=new int[split.training().size()];for(int i=0;i<order.length;i++)order[i]=i;
            for(int epoch=0;epoch<100;epoch++){
                for(int i=order.length-1;i>0;i--){int j=random.nextInt(i+1),v=order[i];order[i]=order[j];order[j]=v;}
                for(int i:order)n.trainSample(split.training().x(i),split.training().y(i),split.training().target(i),.01);
            }
            assertEquals(TrainingService.evaluate(n,split.validation(),.5),result.metrics().getLast().validation());
            var prediction=service.predict(1,2);assertEquals(n.probability(1,2),prediction.probability());
            try(var loaded=new TrainingService(config)){loaded.load(Path.of(result.resultFile()));assertEquals(prediction,loaded.predict(1,2));assertEquals(100,loaded.visualization(10).probabilities().length);}
            assertTrue(Files.readString(Path.of(result.resultFile())).contains("weights"));
        }
    }
    @Test void deterministicRunsAndUnavailableMetrics() throws Exception {
        var c=config(5);try(var a=new TrainingService(c);var b=new TrainingService(c)){
            var ra=a.start().get(10,TimeUnit.SECONDS);var rb=b.start().get(10,TimeUnit.SECONDS);
            assertEquals(ra.testResults(),rb.testResults());assertEquals(a.predict(1,1),b.predict(1,1));
            var m=ra.metrics().getLast();assertNull(m.layers().getFirst().zeroRatio());assertNotNull(m.layers().getFirst().saturationRatio());assertTrue(m.gradientNorms().getFirst()>0);
        }
    }
    @Test void cancellationAndConcurrentStarts() throws Exception {
        try(var service=new TrainingService(config(100000))){var future=service.start();assertThrows(IllegalStateException.class,service::start);assertThrows(IllegalStateException.class,()->service.configure(config(1)));service.stop();var result=future.get(10,TimeUnit.SECONDS);assertEquals(TrainingSnapshot.State.CANCELLED,result.state());assertNotNull(result.finishedAt());assertNotNull(result.resultFile());}
    }
    @Test void historyIsBoundedAndFinalEpochCollected() throws Exception {
        var c=config(2101);
        var network=new Network(2,42,Weights.XAVIER,List.of(new Layer(1,Activation.SIGMOID,new Bias(false,Initialization.ZERO,0,0))));
        c=new TrainingConfig(c.circle(),network,new Datasets(TrainingConfig.dataset(3,101),uniform(3,202),uniform(3,303)),new Training(2101,.01,false,42,1),.5,c.visualization(),directory.toString());
        try(var service=new TrainingService(c)){
            var result=service.start().get(10,TimeUnit.SECONDS);assertEquals(TrainingSnapshot.State.COMPLETED,result.state(),result.error());
            assertTrue(result.metrics().size()<=1000);assertEquals(2101,result.metrics().getLast().epoch());assertNull(result.metrics().getLast().layers().getFirst().meanBias());
        }
    }
    @Test void persistenceFailureIsReported() throws Exception {
        var c=config(1);Path file=directory.resolve("file");Files.writeString(file,"existing file");
        c=new TrainingConfig(c.circle(),c.network(),c.datasets(),c.training(),c.threshold(),c.visualization(),file.toString());
        try(var service=new TrainingService(c)){var result=service.start().get(10,TimeUnit.SECONDS);assertEquals(TrainingSnapshot.State.FAILED,result.state());assertTrue(result.error().contains("save experiment"));}
    }
    @Test void invalidConfiguration(){
        var c=config(1);assertThrows(IllegalArgumentException.class,()->new TrainingConfig(c.circle(),c.network(),c.datasets(),new Training(0,.01,true,42,1),.5,c.visualization(),directory.toString()).validate());
        assertThrows(IllegalArgumentException.class,()->new TrainingConfig(c.circle(),new Network(3,1,Weights.HE,c.network().layers()),c.datasets(),c.training(),.5,c.visualization(),directory.toString()).validate());
        assertThrows(IllegalArgumentException.class,()->new TrainingConfig(c.circle(),new Network(2,1,Weights.HE,List.of(new Layer(2,Activation.TANH,new Bias(true,Initialization.ZERO,0,0)))) ,c.datasets(),c.training(),.5,c.visualization(),directory.toString()).validate());
    }
}
