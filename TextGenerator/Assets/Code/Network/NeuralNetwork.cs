public class NeuralNetwork
{
    public int inputCount;
    public Layer[] layers;
    IActivation hiddenActivation;
    IActivation outputActivation;

    public NeuralNetwork(IActivation hiddenAct, IActivation outputAct, int inputSize, int outputSize, params int[] hiddenLayerSizes)
    {
        hiddenActivation = hiddenAct;
        outputActivation = outputAct;
        inputCount = hiddenLayerSizes[0];
        layers = new Layer[hiddenLayerSizes.Length + 1];

        layers[0] = new Layer(inputSize, hiddenLayerSizes[0], hiddenAct);

        for(int i = 1; i < layers.Length - 1; i++)
        {
            layers[i] = new Layer(hiddenLayerSizes[i - 1], hiddenLayerSizes[i], hiddenAct);
        }

        layers[layers.Length - 1] = new Layer(hiddenLayerSizes[hiddenLayerSizes.Length - 1], outputSize, outputAct);
    }

    //can wrap around to the begin of the array if end is reached
    public void LearnBatch(double[] textData, int startIndex, int inputSize, int batchSize, double learnRate, double momentum, NetworkDataContainer[] networkData, ICost cost)
    {
        System.Threading.Tasks.Parallel.For(0, batchSize, i =>
        {
            UpdateAllGradients(textData, startIndex, inputSize, networkData[i], cost);
        });

        for(int layerIndex = 0; layerIndex < layers.Length; layerIndex++)
        {
            layers[layerIndex].ApplyBiasWeightGradients(learnRate / batchSize, momentum);
        }
    }

    public double[] Evaluate(double[] input, int startIndex = 0)
    {
        input = layers[0].EvaluateLayer(input, startIndex);
        for(int i = 1; i < layers.Length; i++)
        {
            input = layers[i].EvaluateLayer(input);
        }

        return input;
    }

    public int Classify(double[] input, int startIndex = 0)
    {
        double[] output = Evaluate(input, startIndex);
        return MaxIndex(output);
    }

    public void UpdateAllGradients(double[] textData, int startIndex, int inputSize, NetworkDataContainer networkData, ICost cost)
    {
        //forward pass

        double[] layerOutput = textData;

        for (int i = 0; i < layers.Length; i++)
        {
            layerOutput = layers[i].EvaluateLayer(layerOutput, networkData.GetLayerData(i), i > 0? 0 : startIndex);
        }

        //backpropagation

        int layerIndex = layers.Length - 1;
        Layer layer = layers[layerIndex];
        LayerDataContainer layerData = networkData.GetLayerData(layerIndex);

        int expectedOutput = (int)(textData[startIndex + inputSize] * 255.0 + 0.5);
        layer.UpdateOutputLayerWeightedInputDerivatives(layerData, expectedOutput, cost);
        layer.UpdateLayerGradients(layerData);
        
        for(layerIndex--; layerIndex >= 0; layerIndex--)
        {

            layer = layers[layerIndex];
            layerData = networkData.GetLayerData(layerIndex);

            layer.UpdateHiddenLayerWeightedInputDerivatives(layerData, layers[layerIndex + 1], networkData.GetLayerData(layerIndex + 1).weightedInputDerivatives);
            layer.UpdateLayerGradients(layerData);
        }
    }

    public Layer this[int i]
    {
        get 
        {
            return layers[i];
        }
        set 
        {
            layers[i] = value;
        }
    }

    public int MaxIndex(double[] input)
    {
        double max = double.MinValue;
        int maxI = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] > max)
            {
                max = input[i];
                maxI = i;
            }
        }
        return maxI;
    }
}
