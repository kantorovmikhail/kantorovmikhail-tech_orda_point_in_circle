package com.example.circle.assignment;

import com.example.circle.engine.CircleDataset;
import com.example.circle.engine.NeuralNetwork;
import com.example.circle.training.TrainingConfig;
import com.example.circle.training.TrainingService;
import com.example.circle.training.TrainingSnapshot.Scores;

import java.util.Random;

public class CircleAssignment {

    private final TrainingConfig config;
    private CircleDataset.Split datasets;
    private NeuralNetwork model;

    public CircleAssignment() {
        this.config = TrainingConfig.defaults();
    }

    /**
     * Generate training data for a circle
     * centered at (0, 0) with radius 5.
     */
    public CircleDataset generateTrainingData() {
        ensureDatasets();
        return datasets.training();
    }

    /**
     * Generate an independent test dataset.
     */
    public CircleDataset generateTestData() {
        ensureDatasets();
        return datasets.test();
    }

    /**
     * Train the neural network using SGD.
     */
    public void trainModel() {
        CircleDataset trainingData = generateTrainingData();

        model = new NeuralNetwork(config.network());

        TrainingConfig.Training settings = config.training();
        Random random = new Random(settings.seed());

        int[] indices = new int[trainingData.size()];

        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        for (int epoch = 1; epoch <= settings.epochs(); epoch++) {

            if (settings.shuffle()) {
                for (int i = indices.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);

                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }

            for (int index : indices) {
                model.trainSample(
                        trainingData.x(index),
                        trainingData.y(index),
                        trainingData.target(index),
                        settings.learningRate()
                );
            }

            if (epoch == 1 || epoch % 10 == 0
                    || epoch == settings.epochs()) {

                Scores scores = TrainingService.evaluate(
                        model,
                        trainingData,
                        config.threshold()
                );

                System.out.printf(
                        "Epoch %d/%d | Loss: %.5f | Accuracy: %.4f%n",
                        epoch,
                        settings.epochs(),
                        scores.loss(),
                        scores.accuracy()
                );
            }
        }
    }

    /**
     * Evaluate the trained model on unseen test data.
     */
    public Scores testModel() {
        ensureTrained();

        CircleDataset testData = generateTestData();

        Scores scores = TrainingService.evaluate(
                model,
                testData,
                config.threshold()
        );

        System.out.printf(
                "Test: loss=%.5f, accuracy=%.4f, " +
                        "precision=%.4f, recall=%.4f, F1=%.4f%n",
                scores.loss(),
                scores.accuracy(),
                scores.precision(),
                scores.recall(),
                scores.f1()
        );

        return scores;
    }

    /**
     * Predict whether a point is inside the circle.
     *
     * 1 = inside
     * 0 = outside
     */
    public int predict(double x, double y) {
        ensureTrained();
        return model.predict(x, y, config.threshold());
    }

    private void ensureDatasets() {
        if (datasets == null) {
            datasets = CircleDataset.generate(
                    config.circle(),
                    config.datasets(),
                    () -> false
            );
        }
    }

    private void ensureTrained() {
        if (model == null) {
            throw new IllegalStateException(
                    "Model is not trained. Call trainModel() first."
            );
        }
    }

    public static void main(String[] args) {
        CircleAssignment assignment = new CircleAssignment();

        // 1. Generate datasets
        CircleDataset training = assignment.generateTrainingData();
        CircleDataset test = assignment.generateTestData();

        System.out.println("Training samples: " + training.size());
        System.out.println("Test samples: " + test.size());

        // 2. Train neural network
        assignment.trainModel();

        // 3. Evaluate on test data
        assignment.testModel();

        // 4. Predict individual points
        double[][] points = {
                {0, 0},
                {3, 4},
                {2, 2},
                {6, 0},
                {-4, -4}
        };

        for (double[] point : points) {
            int prediction = assignment.predict(point[0], point[1]);

            System.out.printf(
                    "Point (%.1f, %.1f) -> %d%n",
                    point[0],
                    point[1],
                    prediction
            );
        }
    }
}
