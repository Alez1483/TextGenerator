using System.IO;
using UnityEngine;

public static class TextLoader
{
    public static double[] LoadText(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        double[] data = new double[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            data[i] = bytes[i] / 255.0;
        }
        return data;
    }
}