package com.example.circle.api;
import com.example.circle.training.*;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import java.util.Map;
@RestController
@RequestMapping("/api")
public class TrainingController {
    private final TrainingService service;
    public TrainingController(TrainingService service){this.service=service;}
    @GetMapping("/config") public TrainingConfig config(){return service.config();}
    @PutMapping("/config") public TrainingConfig configure(@RequestBody TrainingConfig config){return service.configure(config);}
    @PostMapping("/training") public ResponseEntity<TrainingSnapshot> start(){service.start();return ResponseEntity.accepted().body(service.status());}
    @PostMapping("/training/stop") public ResponseEntity<TrainingSnapshot> stop(){return ResponseEntity.accepted().body(service.stop());}
    @GetMapping("/training") public TrainingSnapshot status(){return service.status();}
    @GetMapping("/metrics") public Object metrics(){return service.status().metrics();}
    @GetMapping("/metrics/layers") public Object layers(){var m=service.status().metrics();return m.isEmpty()?java.util.List.of():m.getLast();}
    @GetMapping("/test-results") public Object test(){var scores=service.status().testResults();if(scores==null)throw new IllegalStateException("Final test results unavailable");return scores;}
    @GetMapping("/prediction") public Object predict(@RequestParam double x,@RequestParam double y){return service.predict(x,y);}
    @GetMapping("/visualization") public Object visualization(@RequestParam(required=false) Integer resolution){return service.visualization(resolution);}
    @ExceptionHandler(IllegalArgumentException.class) @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String,String> invalid(IllegalArgumentException ex){return Map.of("message",ex.getMessage());}
    @ExceptionHandler({HttpMessageNotReadableException.class,MethodArgumentTypeMismatchException.class,org.springframework.web.bind.MissingServletRequestParameterException.class}) @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String,String> malformed(Exception ex){return Map.of("message","Malformed request or unsupported configuration value");}
    @ExceptionHandler(IllegalStateException.class) @ResponseStatus(HttpStatus.CONFLICT)
    public Map<String,String> conflict(IllegalStateException ex){return Map.of("message",ex.getMessage());}
}
