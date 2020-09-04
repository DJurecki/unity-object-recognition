using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class TFVuforia : MonoBehaviour
{
    public RenderTexture tex;
    public NNModel modelAsset;
    private Model m_RuntimeModel;

    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset);
    }

    public void ProcessImage()
    {
        Debug.Log("Process Image Called");
        bool verbose = false;

        var additionalOutputs = new string[] { "StatefulPartitionedCall/sequential/dense/Softmax" };
        var engine = WorkerFactory.CreateWorker(m_RuntimeModel, additionalOutputs, WorkerFactory.Device.GPU, verbose);

        //var model_tf = ModelLoader.LoadFromStreamingAssets(modelName + ".nn"); //if you did tensorflow_to_barracuda.py instead of onnx
        RenderTexture temp_tex = tex;
        Texture2D i_texture = toTexture2D(temp_tex);
        i_texture = Resize(i_texture, 224, 224);

        //texture inputs
        var channelCount = 3; // you can treat input pixels as 1 (grayscale), 3 (color) or 4 (color with alpha) channels
        var input = new Tensor(i_texture, channelCount);

        engine.Execute(input);
        var prediction = engine.PeekOutput("StatefulPartitionedCall/sequential/dense/Softmax"); //put (output name) if there are multiple outputs in model

        float cat_value = prediction[0];
        float[] values = prediction.AsFloats();

        bool isCat = values[0] > 0.99f;
        bool isDog = values[1] > 0.99f;

        Debug.Log("Cat Probability: " + values[0].ToString("F4"));
        Debug.Log("Dog Probability: " + values[1].ToString("F4"));

        prediction.Dispose();
        engine.Dispose();
        input.Dispose();
    }


    public static Texture2D Resize(Texture2D source, int newWidth, int newHeight)
    {
        source.filterMode = FilterMode.Point;
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        rt.filterMode = FilterMode.Point;
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        Texture2D nTex = new Texture2D(newWidth, newHeight);
        nTex.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        nTex.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return nTex;
    }

    Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }
}
