import src_186_BERT_SC_240502 as src_186


sents = [
            "Results and Discussion",
            "We propose the method to investigate the effect.",
            "All samples were synthesized using the catalysts in water.",
            "The chemical structure was confirmed by 1H-NMR.",
            "The results are shown in Fig 1 and Table 1.",
            "The transition temperature was 373 K.",
            "This equation can be applied for the behaviors.",
            "This was due to that the side chain group could not rotate.",
            "where the temperature is reached above 373 K",
        ]
""" The max length for a sentence is 128 tokens. """


model_path_for_infer = "./230_file/ver=136_epoch=19_val_loss=0.873_SC_understand_220318"
device = "cpu"    # CPU:"cpu", GPU_0:[0], GPU_1:[1], , , .
pret_model_path = # Write the path for the pre-trained model such as matscibert in your PC.

results, Y_preds = src_186.infer(sents, model_path_for_infer, device, pret_model_path, screen_print=True)


print("\n\nresults",)
for x in results: print("\n", x)

print("\n\nY_preds\n\n", Y_preds)
""" Y_preds are the label-ids: other:0, title:1, aim:2, preparation:3, , , . """

print("\n\n")
