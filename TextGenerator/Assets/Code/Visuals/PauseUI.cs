using UnityEngine;
using TMPro;
using UnityEngine.UI;


public class PauseUI : MonoBehaviour
{
    [SerializeField] GameObject pauseUI;
    [SerializeField] GameObject stopLearningButton;

    [SerializeField] TextMeshProUGUI inputText;
    [SerializeField] TextMeshProUGUI outputText;

    [SerializeField] int generatedCharCount;

    void Awake()
    {
        pauseUI.SetActive(false);
    }

    public void StopLearning()
    {
        Trainer.Instance.enabled = false;
        pauseUI.SetActive(true);
        stopLearningButton.SetActive(false);
    }

    public void ContinueLearning()
    {
        Trainer.Instance.enabled = true;
        pauseUI.SetActive(false);
        stopLearningButton.SetActive(true);
    }

    public void TextChanged(string txt)
    {
        if (txt.Length > Trainer.Instance.inputSize)
        {
            inputText.color = Color.red;
        }
        else
        {
            inputText.color = Color.black;
        }
    }     

    public void RegenerateText(string txt)
    {
        if (txt.Length > Trainer.Instance.inputSize)
        {
            outputText.text = string.Empty;
            return;
        }

        var network = Trainer.Instance.network;

        double[] input = new double[Trainer.Instance.inputSize];

        int i;
        for (i = 0; i < input.Length - txt.Length; i++)
        {
            input[i] = ' ' / 255.0;
        }
        for(int j = 0; i < input.Length; i++, j++)
        {
            input[i] = txt[j] / 255.0;
        }

        string output = "";

        char outp = (char)network.Classify(input);
        output += outp;
        for (int j = 1; j < generatedCharCount; j++)
        {
            //shift the characters
            for(int k = 1; k < input.Length; k++)
            {
                input[k - 1] = input[k];
            }
            input[input.Length - 1] = outp / 255.0;

            outp = (char)network.Classify(input);
            output += outp;
        }

        outputText.text = output;
    }
}