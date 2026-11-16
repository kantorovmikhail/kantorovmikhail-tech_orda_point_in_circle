package com.example.circle.training;
import com.example.circle.engine.*;
import com.example.circle.training.TrainingSnapshot.*;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;
import tools.jackson.databind.json.JsonMapper;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Adapted single-worker lifecycle from module_1; shared by HTTP and CLI. */
@Service
public final class TrainingService implements AutoCloseable {
    private final ExecutorService executor=Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancellation=new AtomicBoolean();
    private final JsonMapper json=JsonMapper.builder().build();
    private TrainingConfig config;
    private TrainingSnapshot snapshot=new TrainingSnapshot(null,State.IDLE,0,0,null,null,0,List.of(),null,null,null);
    private NeuralNetwork published;
    private TrainingConfig modelConfig;
    private CircleDataset displayed;
    private boolean closed;
    public TrainingService(TrainingConfig config){config.validate();this.config=config;}
    public synchronized TrainingConfig config(){return config;}
    public synchronized TrainingConfig configure(TrainingConfig requested){
        if(snapshot.active())throw new IllegalStateException("Training is active");
        if(requested==null)throw new IllegalArgumentException("Configuration required");
        requested.validate();config=requested;return config;
    }
    public synchronized TrainingSnapshot status(){
        if(!snapshot.active())return snapshot;
        return new TrainingSnapshot(snapshot.id(),snapshot.state(),snapshot.currentEpoch(),snapshot.totalEpochs(),snapshot.startedAt(),null,Duration.between(snapshot.startedAt(),Instant.now()).toMillis()/1000.0,snapshot.metrics(),null,snapshot.error(),null);
    }
    public synchronized CompletableFuture<TrainingSnapshot> start(){
        if(closed||snapshot.active())throw new IllegalStateException("Service closed or training already active");
        config.validate(); TrainingConfig effective=config;cancellation.set(false);
        published=null;modelConfig=null;displayed=null;
        snapshot=new TrainingSnapshot(UUID.randomUUID(),State.RUNNING,0,effective.training().epochs(),Instant.now(),null,0,List.of(),null,null,null);
        return CompletableFuture.supplyAsync(()->train(effective),executor);
    }
    public synchronized TrainingSnapshot stop(){if(snapshot.active()){cancellation.set(true);snapshot=new TrainingSnapshot(snapshot.id(),State.STOPPING,snapshot.currentEpoch(),snapshot.totalEpochs(),snapshot.startedAt(),null,snapshot.durationSeconds(),snapshot.metrics(),null,null,null);}return status();}
    private boolean cancelled(){return cancellation.get()||Thread.currentThread().isInterrupted();}
    private TrainingSnapshot train(TrainingConfig c){
        List<Metric> metrics=new ArrayList<>();int completed=0;NeuralNetwork network=null;Scores test=null;String error=null;State state=State.COMPLETED;
        try {
            CircleDataset.Split split=CircleDataset.generate(c.circle(),c.datasets(),this::cancelled);
            network=new NeuralNetwork(c.network());Random shuffle=new Random(c.training().seed());
            int[] order=new int[split.training().size()];for(int i=0;i<order.length;i++)order[i]=i;
            int stride=Math.max(c.training().metricsEvery(),(c.training().epochs()+999)/1000);
            for(int epoch=1;epoch<=c.training().epochs();epoch++) {
                if(cancelled())break;
                if(c.training().shuffle())for(int i=order.length-1;i>0;i--){int j=shuffle.nextInt(i+1),tmp=order[i];order[i]=order[j];order[j]=tmp;}
                for(int i:order){if(cancelled())break;network.trainSample(split.training().x(i),split.training().y(i),split.training().target(i),c.training().learningRate());}
                if(cancelled())break;
                completed=epoch;
                if(epoch==1||epoch%stride==0||epoch==c.training().epochs()) {
                    List<Double> gradients=network.gradientNorms();
                    metrics.add(new Metric(epoch,c.training().learningRate(),elapsed(),evaluate(network,split.training(),c.threshold()),evaluate(network,split.validation(),c.threshold()),network.layerMetrics(split.training()),gradients));
                    if(metrics.size()>1000)metrics.remove(0);
                    synchronized(this){published=network.copy();modelConfig=c;displayed=split.test();}
                    publish(State.RUNNING,completed,metrics,null,null,null);
                } else publish(State.RUNNING,completed,metrics,null,null,null);
            }
            if(!cancelled())test=evaluate(network,split.test(),c.threshold());
            synchronized(this){published=network.copy();modelConfig=c;displayed=split.test();}
        } catch(CancellationException ex){state=State.CANCELLED;}
          catch(Exception ex){state=State.FAILED;error=ex.getClass().getSimpleName()+": "+ex.getMessage();}
        if(cancelled() && state!=State.FAILED)state=State.CANCELLED;
        // Remain active while persisting, so a new run cannot replace this run's state.
        synchronized(this) {
        TrainingSnapshot terminal;
        terminal=new TrainingSnapshot(snapshot.id(),state,completed,c.training().epochs(),snapshot.startedAt(),Instant.now(),elapsed(),metrics,test,error,null);
        String file=null;
        try {
            Path directory=Path.of(c.resultsDirectory());Files.createDirectories(directory);
            Path target=directory.resolve(terminal.id()+".json");
            json.writerWithDefaultPrettyPrinter().writeValue(target.toFile(),new Experiment(1,c,terminal,network==null?null:network.parameters()));file=target.toString();
        }catch(Exception ex){state=State.FAILED;error="Unable to save experiment: "+ex.getMessage();}
        return publish(state,completed,metrics,test,error,file);
        }
    }
    public static Scores evaluate(NeuralNetwork n,CircleDataset data,double threshold){
        double loss=0;int correct=0,tp=0,fp=0,fn=0;
        for(int i=0;i<data.size();i++) {
            int expected=data.target(i);loss+=n.loss(data.x(i),data.y(i),expected);
            int predicted=n.predict(data.x(i),data.y(i),threshold);
            if(expected==predicted)correct++;if(predicted==1){if(expected==1)tp++;else fp++;}else if(expected==1)fn++;
        }
        double precision=tp+fp==0?0:tp/(double)(tp+fp),recall=tp+fn==0?0:tp/(double)(tp+fn);
        if(!Double.isFinite(loss))throw new ArithmeticException("Non-finite loss");
        return new Scores(loss/data.size(),correct/(double)data.size(),precision,recall,precision+recall==0?0:2*precision*recall/(precision+recall),data.size());
    }
    private synchronized double elapsed(){return Duration.between(snapshot.startedAt(),Instant.now()).toMillis()/1000.0;}
    private synchronized TrainingSnapshot publish(State state,int epoch,List<Metric> metrics,Scores test,String error,String file){
        if(state==State.RUNNING && cancellation.get())state=State.STOPPING;
        boolean terminal=state!=State.RUNNING&&state!=State.STOPPING;
        snapshot=new TrainingSnapshot(snapshot.id(),state,epoch,snapshot.totalEpochs(),snapshot.startedAt(),terminal?Instant.now():null,elapsed(),metrics,test,error,file);return snapshot;
    }
    public record Experiment(int version,TrainingConfig config,TrainingSnapshot snapshot,NeuralNetwork.Parameters model){}
    public synchronized void load(Path path) throws java.io.IOException {
        if(snapshot.active())throw new IllegalStateException("Training is active");
        Experiment saved=json.readValue(path.toFile(),Experiment.class);saved.config().validate();
        if(saved.version()!=1||saved.model()==null)throw new IllegalArgumentException("Unsupported or missing model");
        NeuralNetwork n=new NeuralNetwork(saved.config().network());n.restore(saved.model());
        config=saved.config();modelConfig=config;published=n;displayed=null;snapshot=saved.snapshot();
    }
    public record Prediction(double x,double y,CircleDataset.Geometry geometry,int expected,double probability,int predicted,boolean correct){}
    public synchronized Prediction predict(double x,double y){
        if(published==null)throw new IllegalStateException("No model available yet");
        double p=published.probability(x,y);int expected=CircleDataset.label(modelConfig.circle(),x,y),predicted=p>=modelConfig.threshold()?1:0;
        return new Prediction(x,y,CircleDataset.geometry(modelConfig.circle(),x,y),expected,p,predicted,expected==predicted);
    }
    public record Visualization(TrainingConfig.Circle circle,double threshold,double minX,double maxX,double minY,double maxY,int resolution,double[] probabilities,List<Prediction> points){}
    public synchronized Visualization visualization(Integer requested){
        if(published==null)throw new IllegalStateException("No model available yet");
        int resolution=requested==null?modelConfig.visualization().resolution():requested;
        if(resolution<10||resolution>250)throw new IllegalArgumentException("Resolution must be 10..250");
        var d=modelConfig.datasets().test();double[] probabilities=new double[resolution*resolution];
        for(int row=0;row<resolution;row++)for(int col=0;col<resolution;col++)probabilities[row*resolution+col]=published.probability(d.minX()+(d.maxX()-d.minX())*col/(resolution-1),d.minY()+(d.maxY()-d.minY())*row/(resolution-1));
        List<Prediction> points=new ArrayList<>();
        if(displayed!=null){int count=Math.min(displayed.size(),modelConfig.visualization().maxPoints());for(int i=0;i<count;i++){int index=(int)((long)i*displayed.size()/count);points.add(predict(displayed.x(index),displayed.y(index)));}}
        return new Visualization(modelConfig.circle(),modelConfig.threshold(),d.minX(),d.maxX(),d.minY(),d.maxY(),resolution,probabilities,points);
    }
    @PreDestroy public void close(){synchronized(this){closed=true;stop();executor.shutdown();}try{if(!executor.awaitTermination(10,TimeUnit.SECONDS))executor.shutdownNow();}catch(InterruptedException e){executor.shutdownNow();Thread.currentThread().interrupt();}}
}
