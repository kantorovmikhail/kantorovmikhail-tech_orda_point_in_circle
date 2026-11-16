package com.example.circle.training;
import com.example.circle.WebApplication;
import com.example.circle.CliApplication;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import java.net.*;
import java.net.http.*;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;
@SpringBootTest(classes=WebApplication.class,webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT)
class ApplicationTest {
    @LocalServerPort int port;
    @TempDir Path directory;
    final HttpClient client=HttpClient.newHttpClient();
    HttpResponse<String> call(String method,String path,String body) throws Exception{
        return client.send(HttpRequest.newBuilder(URI.create("http://localhost:"+port+path)).header("Content-Type","application/json").method(method,body==null?HttpRequest.BodyPublishers.noBody():HttpRequest.BodyPublishers.ofString(body)).build(),HttpResponse.BodyHandlers.ofString());
    }
    @Test void webStartupAndRestValidation() throws Exception {
        assertEquals(200,call("GET","/",null).statusCode());var config=call("GET","/api/config",null);assertEquals(200,config.statusCode());assertTrue(config.body().contains("SIGMOID"));
        assertEquals(200,call("PUT","/api/config",config.body()).statusCode());
        for(String body:new String[]{"null","{}","{","{\"threshold\":2}"})assertEquals(400,call("PUT","/api/config",body).statusCode());
        assertEquals(409,call("GET","/api/prediction?x=0&y=0",null).statusCode());assertEquals(400,call("GET","/api/prediction?x=no&y=0",null).statusCode());assertEquals(400,call("GET","/api/prediction?x=0",null).statusCode());
        assertEquals(200,call("GET","/api/metrics",null).statusCode());assertEquals(409,call("GET","/api/test-results",null).statusCode());
        assertEquals(202,call("POST","/api/training",null).statusCode());assertEquals(409,call("POST","/api/training",null).statusCode());assertEquals(202,call("POST","/api/training/stop",null).statusCode());
    }
    @Test void cliRunsWithoutServerAndTerminates(){
        assertEquals(0,CliApplication.run("--experiment.training.epochs=1","--experiment.datasets.training.sample-count=100","--experiment.datasets.validation.sample-count=50","--experiment.datasets.test.sample-count=50","--experiment.results-directory="+directory,"--spring.main.banner-mode=off"));
        try(var files=java.nio.file.Files.list(directory)) {
            Path artifact=files.filter(path->path.toString().endsWith(".json")).findFirst().orElseThrow();
            assertEquals(0,CliApplication.run("--model="+artifact,"--x=3","--y=4","--spring.main.banner-mode=off"));
        } catch(java.io.IOException ex){fail(ex);}
        assertEquals(1,CliApplication.run("--experiment.training.epochs=0","--spring.main.banner-mode=off"));
    }
}
