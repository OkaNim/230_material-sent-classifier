材料論文における試料調製や物性測定、それらに対する結果・考察、また、それまでの既知事実などの文を分類します。<br>
<br>
<br>
## Requirements
Python >= 3.8.5<br>
PyTorch (Please get the optical version from the web page (https://pytorch.org/))<br>
The Python libraries in requrements.txt<br>
<br>
<br>
## How to use
Please create the script such as 230_test.py in /src and run there.<br>
```
python 230_test.py
```
<br>
----- 230_test.py -----<br>
<br>
import src_186_BERT_SC_240502 as src_186<br>
<br>
<br>
sents = [<br>
            "Results and Discussion",<br>
            "We propose the method to investigate the effect.",<br>
            "All samples were synthesized using the catalysts in water.",<br>
            "The chemical structure was confirmed by 1H-NMR.",<br>
            "The results are shown in Fig 1 and Table 1.",<br>
            "The transition temperature was 373 K.",<br>
            "This equation can be applied for the behaviors.",<br>
            "This was due to that the side chain group could not rotate.",<br>
            "where the temperature is reached above 373 K",<br>
        ]<br>
<br>
<br>
model_path_for_infer = "./230_file/ver=136_epoch=19_val_loss=0.873_SC_understand_220318"<br>
GPU_ID = "cpu"    # CPU:"cpu", GPU_0:[0], GPU_1:[1], , ,.<br>
pret_model_path = # Write the path for the pre-trained model such as matscibert.<br>
<br>
results, Y_preds = src_186.infer(sents, model_path_for_infer, GPU_ID, pret_model_path, screen_print=True)<br>
<br>
<br>
print("\n\nresults",)<br>
for x in results:<br>
    print("\n", x)<br>
<br>
<br>
print("\n\nY_preds\n\n", Y_preds, "\n\n")<br>
"""Y_preds are the label-ids for pytorch: e.g. title:0, aim:2, preparation:3."""<br>
-------------------<br>
<br>
<br>
The results are outputted as follow:<br>
<br>
results<br>
<br>
 {'text': 'Results and Discussion', 'label_pred': 'title'}<br>
<br>
 {'text': 'We propose the method to investigate the effect.', 'label_pred': 'aim'}<br>
<br>
 {'text': 'All samples were synthesized using the catalysts in water.', 'label_pred': 'preparation'}<br>
<br>
 {'text': 'The chemical structure was confirmed by 1H-NMR.', 'label_pred': 'measurement'}<br>
<br>
 {'text': 'The results are shown in Fig 1 and Table 1.', 'label_pred': 'showing'}<br>
<br>
 {'text': 'The transition temperature was 373 K.', 'label_pred': 'result'}<br>
<br>
 {'text': 'This equation can be applied for the behaviors.', 'label_pred': 'knowledge'}<br>
<br>
 {'text': 'This was due to that the side chain group could not rotate.', 'label_pred': 'consideration'}<br>
<br>
 {'text': 'where the temperature is reached above 373 K', 'label_pred': 'condition'}<br>
<br>
<br>
Y_preds<br>
<br>
 [1, 2, 3, 4, 5, 6, 7, 8, 9]<br>
<br>
<br>
It is necessary to get the pre-trained model such as matscibert from the Hugging Face's web page (https://huggingface.co/m3rg-iitd/matscibert).<br>
<br>
<br>