package com.example.circle;
import com.example.circle.training.TrainingConfig;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
@SpringBootApplication
public class WebApplication {
    public static void main(String[] args){SpringApplication.run(WebApplication.class,args);}
    @Bean TrainingConfig experiment(Environment environment){
        TrainingConfig c=Binder.get(environment).bind("experiment",TrainingConfig.class).orElse(TrainingConfig.defaults());c.validate();return c;
    }
}
