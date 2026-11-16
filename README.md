# Circle Neural Network Lab

Java 21 / Spring Boot 4.1.2 educational experiment application. A configurable dense network learns whether a point is inside a circle using online SGD. There is no ML framework or database. The web and CLI entry points share the same network, generator, evaluation and training service.

## Assignment: Point Inside Circle

### Task

Create a neural network that determines whether a point `(x, y)` lies inside a circle centered at `(0, 0)` with radius `5`.

- Input: coordinates `x`, `y`.
- Output: `1` for a point strictly inside the circle; `0` for a point outside or on its boundary.
- Generate training and test datasets.
- Implement a method to train the neural network on the generated data.

The reference label is `1` when `x*x + y*y < 25`, otherwise `0`. The formula is used to generate labels; predictions are produced by the trained neural network.

### Assignment implementation

The optional, assignment-focused entry point is `src/main/java/com/example/circle/assignment/CircleAssignment.java`. **This class must be added to the project separately; it is not included in the original archive.** It reuses the existing dataset generator and neural-network implementation rather than implementing a second model.

| Method | Responsibility |
| --- | --- |
| `generateTrainingData()` | Obtain generated training samples and their labels |
| `generateTestData()` | Obtain an independent test split |
| `trainModel()` | Train the neural network using SGD |
| `testModel()` | Evaluate the trained network on the test split |
| `predict(double x, double y)` | Return the predicted class `0` or `1` |
| `main(String[] args)` | Demonstrate dataset generation, training, testing, and predictions |

The shared generator also creates a validation split. The simplified assignment runner does not use that split for training or report validation metrics; the full training service does.

### Running the assignment in IntelliJ IDEA

1. Install JDK 21 and open the project as a Gradle project.
2. Add `CircleAssignment.java` under `src/main/java/com/example/circle/assignment/` if it has not already been added.
3. Let IntelliJ import Gradle dependencies.
4. Open `CircleAssignment.java`, then run its `main()` method using the green Run icon.
5. Review the training epoch logs, test metrics, and predictions in the Run console.

### Running the assignment from a terminal

The following commands assume that `CircleAssignment.java` has been added and compiles successfully. From the project root on Linux/macOS:

```bash
chmod +x gradlew
./gradlew classes
./gradlew --stop  # optional: stop existing Gradle daemons
```

To run its `main()` without changing the Gradle build, add this task to `build.gradle`:

```groovy
tasks.register('assignment', JavaExec) {
    classpath = sourceSets.main.runtimeClasspath
    mainClass = 'com.example.circle.assignment.CircleAssignment'
}
```

Then execute:

```bash
./gradlew assignment
```

This prints dataset sizes, training progress, test metrics, and classifications for example points. The predictions are learned approximations: a point exactly on the circle boundary, such as `(3, 4)`, may be misclassified even though its reference label is `0`.

### Running the full application

The original project supports two entry points without the assignment class:

**Web application**

```bash
./gradlew bootRun
```

Open `http://localhost:8080` to configure, start, and inspect training. Stop with `Ctrl+C`.

**Command-line application**

```bash
./gradlew cli
./gradlew cli --args='--experiment.training.epochs=20 --experiment.results-directory=results/demo'
```

**Automated tests**

```bash
./gradlew test
./gradlew build
```

The commands above use the bundled Gradle wrapper. Java 21 is required. The first run may download Gradle and project dependencies.

### What to demonstrate during review

Show that the project generates separate training and test samples, trains a neural network using coordinates as inputs, reports accuracy on unseen test points, and predicts the class of individual coordinates. Explain that the reference circle equation supplies labels while the neural network performs inference.

---

## Run

Requires Java 21. The included Gradle 9.8.2 wrapper downloads Gradle and Maven dependencies on the first run.

```bash
./gradlew bootRun
# Open http://localhost:8080

./gradlew cli
./gradlew cli --args='--experiment.training.epochs=20 --experiment.results-directory=results/demo'
./gradlew cli --args='--spring.config.additional-location=file:./experiment.yml'

./gradlew test
./gradlew build
java -jar build/libs/point-in-circle-0.0.1-SNAPSHOT.jar
```

`WebApplication` starts the embedded MVC server, bound to loopback by default. `CliApplication` explicitly selects `WebApplicationType.NONE`, prints progress and final scores, waits for training and persistence, closes the context/executor, and exits 0 on success or 1 on failure. `cli` is a Gradle JavaExec task with the CLI main class; `bootRun` and the executable jar select the web main class.

Load a saved experiment for inference without retraining:

```bash
./gradlew cli --args='--model=results/REPLACE-WITH-RUN-ID.json --x=3 --y=4'
```

The saved artifact includes all configuration, seeds, bounded epoch metrics, final test metrics, duration, status, and weights/biases. JSON schema version 1 is validated when loaded. Failed/cancelled experiments also produce artifacts when the result directory is writable; cancellation during generation may have no model. The artifact is written once at termination; the live API returns its path. Paths are local operator configuration, not remotely uploaded model paths.

## Structure

- `engine/Activation`: forward functions and analytic derivatives.
- `engine/Neuron`, `DenseLayer`, `NeuralNetwork`: trainable object hierarchy, buffers, generalized backpropagation, parameter snapshots and diagnostics.
- `engine/CircleDataset`: primitive storage, labeling, sampling, split separation.
- `training/TrainingConfig`: effective immutable configuration.
- `training/TrainingService`, `TrainingSnapshot`: shared training lifecycle, evaluation, inference, experiment artifacts.
- `api/TrainingController`: HTTP adapter.
- `WebApplication`, `CliApplication`: independent entry points.
- `resources/static`: HTML, CSS, JavaScript; Chart.js loaded from pinned CDN URL.

## Mathematics and learning

For center `(cx,cy)` and radius `r`, target `1` means `(x-cx)²+(y-cy)² < r²`. Equality is outside (target `0`). Defaults are center `(0,0)`, radius `5`, threshold `0.5`. The model receives exactly `[x,y]`; it does not receive radius, squared distance, labels, or engineered geometric features. Geometry is only used for dataset labels and reference evaluation/visualization. Prediction always runs the neural network.

A neuron computes `z = Σ(w_i * input_i) + b` and `a = activation(z)`. Weights determine how inputs contribute. Bias shifts the activation threshold independently of inputs. Disabled bias contributes zero and never updates. Each dense layer feeds its activations to the next. Hidden layers may use LINEAR, SIGMOID, TANH, RELU or LEAKY_RELU (slope 0.01); the final layer must be one SIGMOID neuron so output is a probability. ReLU's derivative at zero is defined as zero; leaky ReLU uses 0.01 there.

Sigmoid uses separate positive/negative branches to avoid exponential overflow. BCE is calculated from the final logit using `max(z,0) - z*target + log1p(exp(-abs(z)))`. Its combined sigmoid-output delta is `probability-target`, including at saturated outputs. Hidden deltas are `(nextWeightsᵀ * nextDeltas) * activationDerivative`. All deltas use pre-update weights. Each weight then updates by `w -= rate * delta * input`; enabled bias by `b -= rate * delta` exactly once. There is no batching, momentum or Adam. Shuffling is seeded Fisher–Yates. Validation/test samples never enter this update loop.

## Configuration

The complete runnable YAML is [application.yml](src/main/resources/application.yml). Spring relaxed binding accepts lowercase YAML enum names and kebab-case properties. REST uses enum names such as `TANH` and camelCase JSON properties. The web editor exposes the full JSON configuration, including adding/removing layers, every bias initialization parameter, all dataset parameters, seeds, metric frequency and visualization limits. Save validates before accepting; Start validates the edited configuration too.

For a separate YAML override file (merged with defaults):

```yaml
experiment:
  circle: { x: 0, y: 0, radius: 5, tolerance: 0.01 }
  threshold: 0.5
  network:
    input-size: 2
    seed: 42
    initialization: XAVIER # HE or UNIFORM also supported
    layers:
      - neurons: 16
        activation: tanh
        bias: { enabled: true, initialization: zero, value: 0, scale: 0.1 }
      - neurons: 16
        activation: relu
        bias: { enabled: true, initialization: random, value: 0, scale: 0.1 }
      - neurons: 1
        activation: sigmoid
        bias: { enabled: true, initialization: constant, value: 0, scale: 0 }
  training:
    epochs: 100
    learning-rate: 0.01
    shuffle: true
    seed: 42
    metrics-every: 5
```

Specify the entire layers list when overriding it. Random bias is uniform in `[value-scale,value+scale)`. Disabled bias ignores initialization. Xavier uses uniform ±sqrt(6/(fanIn+fanOut)); HE uses Gaussian standard deviation sqrt(2/fanIn); UNIFORM uses [-0.5,0.5). Zero hidden layers are supported by providing only the output layer. Defensive limits are 32 layers, 2,048 neurons/layer and two million weights. Changing circle geometry requires retraining; the configuration cannot change during training. Until a new run begins, inference uses the most recently trained model's own effective configuration.

## Dataset policies

Each of training/validation/test has independent sample count, seed, bounds, strategy, coordinate step, boundary fraction/width, radial distribution, and explicit boundary count. Different split seeds are required.

- **UNIFORM**: independent uniform X and Y in the configured rectangle.
- **BOUNDARY_FOCUSED**: uniform angle, radius `circle.radius + offset`. Offset is Gaussian with configured standard deviation `boundary-width`, or uniform ±`boundary-width`. Negative radius and out-of-bounds proposals are rejected.
- **HYBRID**: boundary-focused proposal with probability `boundary-fraction`, uniform otherwise (default 50/50). Rejection can change the accepted proportion if bounds clip either source.
- **GRID**: every `minX+i*step, minY+j*step` inside bounds, including aligned endpoints. Total is `(floor((maxX-minX)/step)+1)*(floor((maxY-minY)/step)+1)`, with a tiny floating-point endpoint tolerance. `sample-count` is ignored for GRID; there is no truncation to that count. Step must be positive and extra boundary count zero.

For random strategies, `sample-count` is the exact total accepted count. `step: 0` means continuous coordinates; positive step rounds to the lattice anchored at each minimum bound, and out-of-bounds rounded proposals are rejected. Explicit boundary samples occupy the first `exact-boundary-samples` slots, included in the total, and bypass quantization. They are deliberately generated using `r*cos(theta), r*sin(theta)`, with a few ULP corrections/retries so squared distance equals radius squared under the same Java arithmetic used for labels. They therefore target 0. Bounds must contain the entire circle when explicit boundary samples are requested. Extreme centers/radii may not admit representable equality; generation fails explicitly after bounded attempts rather than fabricating labels.

`Geometry` is INSIDE, OUTSIDE or BOUNDARY according to radial distance and diagnostic tolerance. A **near-boundary point strictly inside still targets 1**, even when its diagnostic geometry is BOUNDARY. Only exact mathematical equality targets 0. This resolves the tension between a diagnostic tolerance and the mandatory strict target definition; tolerance never changes labels.

A primitive open-addressed fingerprint set rejects repeated coordinates within and across splits, in training → validation → test order. Hash collisions can conservatively reject distinct proposals but cannot permit identical points. Signed zero is normalized for separation. Random strategies retry duplicates; overlapping grids fail rather than being truncated or moved. Use disjoint grid bounds or different lattices for multiple grid splits. Large step/count combinations with insufficient distinct points fail after a bounded proposal budget. Split reproducibility depends on the complete dataset configuration and split order, not just the individual seed.

Storage is two `double[]` and two `byte[]` arrays (~18 MB per million points), plus temporary primitive uniqueness storage (power-of-two capacity at least 1.5× combined samples, 8 bytes per slot) and a 4-byte/sample training shuffle array. The uniqueness set is released after generation. There are no per-sample training objects or histories. Each split is capped at ten million samples; heap and runtime must be sized for the combined experiment. A million-sample generator test is included; a full million-sample training benchmark is not part of the tests.

## Metrics and visualization

At configured intervals, both training and validation scores are measured on their complete split using the same post-epoch model: mean BCE, accuracy, precision, recall and F1. Zero-denominator precision/recall/F1 are defined as zero. Learning rate, epoch and elapsed seconds accompany each point. Final test metrics are computed once, after successful training, on the independent test set. Cancelled runs have no final test result.

History is capped at 1,000 entries; large epoch counts increase the collection stride, preserving first/final observations when possible. Layer activation mean/standard deviation use up to 2,048 evenly selected training points. Weight norm is the Euclidean norm of weights, excluding bias. Bias mean/std cover enabled neurons. Gradient norm includes enabled bias gradients and weights for the last training sample before its SGD update; it is not an epoch average. ReLU zero ratio and sigmoid/tanh saturation (sigmoid <.02 or >.98; abs(tanh)>.98) are null when inapplicable. Disabled bias statistics are null. No per-sample history is retained.

Compare training and validation loss on the shared chart to investigate overfitting. Falling training loss with rising validation loss is evidence to investigate; training accuracy alone cannot establish it. Precision/recall/F1 are threshold-dependent and distribution-dependent.

The network diagram renders up to eight representative neurons per layer with true counts labeled. The coordinate plane uses an actual network probability grid; magenta threshold contours use interpolated grid-edge crossings. The black dashed circle is the geometric reference. Rendered test points are deterministically limited by `max-points`; red outlines indicate mistakes. Resolution is configurable (10–250). Inference uses a copied, published network under service synchronization, so it never observes partially updated weights. Training itself owns a separate instance of the same NeuralNetwork class. Model publication is at metric intervals.

## REST endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET / PUT | `/api/config` | Read or replace complete experiment configuration |
| POST | `/api/training` | Start asynchronously, HTTP 202 |
| POST | `/api/training/stop` | Request cooperative cancellation, HTTP 202 |
| GET | `/api/training` | Lifecycle, epochs, duration, metrics, errors and result file |
| GET | `/api/metrics` | Bounded model and layer metric history |
| GET | `/api/metrics/layers` | Latest metric observation with layer statistics/gradients |
| GET | `/api/prediction?x=3&y=4` | Expected class, geometry, probability, predicted class, correctness |
| GET | `/api/visualization?resolution=80` | Actual probability grid, reference circle and bounded point predictions |
| GET | `/api/test-results` | Final independent test metrics |

Malformed/incompatible configurations and invalid coordinates return 400; active-run conflicts and unavailable models/results return 409. Single-worker synchronized admission prevents concurrent experiments. Configuration updates are rejected while a run is active. There is deliberately no HTTP endpoint that reads arbitrary local files; use CLI or TrainingService.load for trusted local model loading.

## Verification and limitations

`./gradlew test bootJar` covers activation/derivative behavior, optional bias, once-only bias updates, arbitrary depth, every weight/bias finite-difference gradients, stable BCE, SGD, strict labels and boundary diagnostics, explicit boundary generation, counts, distributions, quantization, separation, reproducibility, a million-sample allocation test, deterministic training, validation isolation, cancellation, HTTP startup/validation, CLI success/failure, JSON persistence and reload.

The deterministic integration acceptance is **at least 94% accuracy** on 1,500 independent uniform test points in [-7,7]², after 100 epochs on 1,500 hybrid training samples, 16→16 tanh hidden neurons, network seed 42, training seed 42, learning rate .01, dataset seeds 101/202/303. It also requires training loss to decrease and independently recomputes validation scores. It does not promise perfect boundary classification. Different seeds, boundary-focused test distributions, changed circle dimensions or poorly chosen activations can lower accuracy.

Training is CPU-only online SGD; dense huge configurations will be slow. Inputs are raw coordinates, so extreme scales can saturate activations; divergence is reported as FAILED. Metrics involve complete split passes and can be costly; increase `metrics-every`. There is no checkpoint resume, early stopping, normalization, authentication, multi-user experiment store or database. Chart.js needs CDN access; other UI features and API work locally. Canvas contours are resolution-limited approximations of predictions. Browser interaction has not been manually verified; automated application tests exercise real embedded-server HTTP endpoints and static assets, and JavaScript syntax is checked separately.