package com.example.circle;
import com.example.circle.training.TrainingService;
import com.example.circle.training.TrainingSnapshot;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.WebApplicationType;
import java.nio.file.Path;
public final class CliApplication {
    public static void main(String[] args){System.exit(run(args));}
    public static int run(String... args){
        SpringApplication app=new SpringApplication(WebApplication.class);app.setWebApplicationType(WebApplicationType.NONE);
        try(var context=app.run(args)) {
            TrainingService service=context.getBean(TrainingService.class);
            String model=context.getEnvironment().getProperty("model");
            if(model!=null){service.load(Path.of(model));double x=context.getEnvironment().getProperty("x",Double.class,0.0),y=context.getEnvironment().getProperty("y",Double.class,0.0);System.out.println(service.predict(x,y));return 0;}
            var future=service.start();
            while(!future.isDone()){var status=service.status();System.out.printf("%s epoch %d/%d (%.1fs)%n",status.state(),status.currentEpoch(),status.totalEpochs(),status.durationSeconds());
                if(!status.metrics().isEmpty()){var m=status.metrics().getLast();System.out.printf("loss %.5f / val %.5f; accuracy %.4f / val %.4f%n",m.training().loss(),m.validation().loss(),m.training().accuracy(),m.validation().accuracy());}
                Thread.sleep(500);}
            TrainingSnapshot result=future.join();System.out.printf("%s: epoch %d, duration %.2fs%nTest: %s%nSaved: %s%n",result.state(),result.currentEpoch(),result.durationSeconds(),result.testResults(),result.resultFile());
            if(result.error()!=null)System.err.println(result.error());
            return result.state()==TrainingSnapshot.State.COMPLETED?0:1;
        }catch(Exception ex){System.err.println("Experiment failed: "+ex.getMessage());return 1;}
    }
}
