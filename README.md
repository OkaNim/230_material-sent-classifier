材料論文における試料調製や物性測定、それらに対する結果・考察、また、それまでの既知事実などの文を分類します。<br>
<br>
<br>
## How to use
Please create the script such as 229_test.py in /src and run it after moving to /src.<br>
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
The BIOES tags are outputted for every token as follow:<br>
refsign_tags<br>
 ['O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'E', 'O', 'S', 'O', 'O', 'B', 'E', 'O', 'O', 'O', 'O', 'O', 'O']<br>
<br>
The abbreviations for referring signs will be also outputted as follow:<br>
refsign_abbs<br>
 {'BBCP': 'bottlebrush block copolymers'}<br>
<br>
Polymer name recogonition is also included in 229_test.py. <br>
The BIOES tags are outputted for every token as follow:<br>
polymer_tags<br>
 ['O', 'B', 'I', 'I', 'I', 'E', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O']<br>
<br>
<br>
The tokenized text (toks) and the POS tags (pos_tags) are necessary.<br>
The POS tags should be obtained using Stanford Core NLP (https://stanfordnlp.github.io/CoreNLP/).<br>
<br>
<br>
