using UnityEngine;
using TMPro;
using System.IO;

public class Trainer : MonoBehaviour
{
    public static Trainer Instance;

    [HideInInspector] public NeuralNetwork network;

    NetworkDataContainer[] networkTrainData;

    ICost cost;

    [Range(0.0f, 1.0f)] public double dataSplit = 0.85;
    [SerializeField] public int inputSize;
    [SerializeField] private int[] hiddenLayerSizes;
    [SerializeField, Range(0f, 1f)] private double learnRate = 1;
    [SerializeField, Range(0f, 1f)] private double momentum;
    [SerializeField] private int batchSize;
    private int batchStart;

    [HideInInspector] public double[] trainData;
    [HideInInspector] public double[] testData;

    double epochAtm = 0;
    System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();

    [Header("Graph Drawer")]
    [Range(0.0f, 1000.0f), SerializeField] private float scrollMultiplier; 
    [SerializeField] private int testSize = 100;
    [SerializeField] GraphDrawer graphDrawer;

    [SerializeField, Range(0f, 10f)] double graphUpdateRate;
    double lastUpdate;

    void OnEnable()
    {
        lastUpdate = Time.timeAsDouble;
    }

    void Awake()
    {
        Instance = this;
        //initialize
        var hiddenAct = new ReLu();
        cost = new CrossEntropy();
        var outputAct = new Softmax();
        network = new NeuralNetwork(hiddenAct, outputAct, inputSize, 255, hiddenLayerSizes);
        networkTrainData = new NetworkDataContainer[batchSize];
        for(int i = 0; i <  networkTrainData.Length; i++)
        {
            networkTrainData[i] = new NetworkDataContainer(network);
        }
        
        //load data
        LoadData();

        graphDrawer.Initialize(testData, trainData, network);

        batchStart = Random.Range(0, inputSize);
    }

    void Update()
    {
        timer.Restart();

        do
        {
            network.LearnBatch(trainData, batchStart, inputSize, batchSize, learnRate, momentum, networkTrainData, cost);

            batchStart += batchSize * inputSize;
            epochAtm += (batchSize * inputSize / (double)trainData.Length);

            if (batchStart + inputSize * batchSize >= trainData.Length)
            {
                batchStart = Random.Range(0, inputSize);
            }
        } while (timer.ElapsedMilliseconds < 16);

        timer.Stop();

        if (Time.timeAsDouble - lastUpdate > graphUpdateRate)
        {
            lastUpdate += graphUpdateRate;
            graphDrawer.RunTest(testSize, inputSize, epochAtm * scrollMultiplier);
        }

        graphDrawer.Update(epochAtm);
    }

    void LoadData()
    {
        string trainImagePath = Path.Combine("Assets", "Code", "Data", "archive", "Papers.csv");
        double[] allData = TextLoader.LoadText(trainImagePath);

        (trainData, testData) = DataSetHelper.SplitData(allData, dataSplit);
    }
}
