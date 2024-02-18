public static class DataSetHelper
{
    public static (double[] trainData, double[] testData) SplitData(double[] allData, double split)
    {
        int trainCount = (int)(allData.Length * split);
        int testCount = allData.Length - trainCount;

        double[] trainData = new double[trainCount];
        double[] testData = new double[testCount];

        int i;
        for(i = 0; i < trainCount; i++)
        {
            trainData[i] = allData[i];
        }
        for(int j = 0; j < testCount; j++, i++)
        {
            testData[j] = allData[i];
        }
        return (trainData, testData);
    }
}
