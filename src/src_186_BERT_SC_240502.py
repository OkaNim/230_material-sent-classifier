# The source codes here were created in basis on　the source codes using in the following book:
# 「BERTによる自然言語処理入門: Transformersを使った実践プログラミング」 (ISBN-13: 978-4274227264)
# Copyright (c) 2021 Stockmark Inc.
# Released under the MIT license
# https://github.com/stockmarkteam/bert-book?tab=MIT-1-ov-file#readme




# import
import os
import json
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger




# f
def train(trains, tests, params):
    print("\n")
    print("train ...\n\n")


    curdir = os.getcwd()
    modeldir = make_tmpdir(curdir, remove=True, dirname="model")
    modelbestdir = make_tmpdir(curdir, remove=False, dirname="model_best")
    logsdir = make_tmpdir(curdir, remove=False, dirname="logs")
    defaultdir = make_tmpdir(logsdir, remove=False, dirname="default")


    pret_model_path = params["pret_model_path"]
    labels = params["labels"]
    EPOCH = params["epoch"]
    MAX_LENGTH = params["max_length"]
    LR = params["lr"]
    BATCH_TRAIN = params["batch_train"]
    BATCH_TEST = params["batch_test"]
    GPU_ID = params["gpu_id"]


    label_id = create_ids_for_labels(labels)                                   # f


    ver = get_version()                                                        # f


    tokenizer = AutoTokenizer.from_pretrained(pret_model_path)

    trains_enc = encode_data(trains, tokenizer, label_id, MAX_LENGTH)          # f
    tests_enc = encode_data(tests, tokenizer, label_id, MAX_LENGTH)            # f
    trains_dl = DataLoader(trains_enc, batch_size=BATCH_TRAIN)
    tests_dl = DataLoader(tests_enc, batch_size=BATCH_TEST)


    model = AutoModelForSequenceClassification_pl(pret_model_path, num_labels=len(labels), lr=LR)    # f

    logger = CSVLogger("logs", name="default")
    checkpoint = pl.callbacks.ModelCheckpoint(
                    monitor="val_loss", mode="min", save_top_k=1,
                    save_weights_only=True, dirpath="model/",
                    filename="ver=" + ver + "_{epoch:02d}_{val_loss:.3f}")

    trainer = pl.Trainer(accelerator="gpu", devices=GPU_ID, max_epochs=EPOCH, callbacks=[checkpoint], logger=logger)
    trainer.fit(model, trains_dl, tests_dl)
    trainer.logger.log_metrics

    best_model_path = checkpoint.best_model_path
    best_model = checkpoint.best_model_path.split("/")[-1]
    best_model = best_model[:-len(".ckpt")]
    best_score = checkpoint.best_model_score.item()

    model = AutoModelForSequenceClassification_pl.load_from_checkpoint(best_model_path)
    save_path = os.path.join("./model_best", best_model)
    model.bert_sc.save_pretrained(save_path)

    print("\n\n\tbest_model = {}".format(best_model))
    print("\tbest_val_loss = {}\n\n".format(best_score))

    outfpath = os.path.join(save_path, "label_id.json")
    with open(outfpath, 'w') as f:
        json.dump(label_id, f, indent=4, ensure_ascii=False)

    outfpath = os.path.join(save_path, "parameter.json")
    with open(outfpath, 'w') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)


    docs = list(map(lambda a: a[0], tests))
    results, Y_pred = infer(docs, save_path, GPU_ID, pret_model_path, screen_print=True)    # f

    cas = list(map(lambda a: a[1], tests))
    for label_ca, y in zip(cas, results): y["label_ca"] = label_ca
    Y_ca = [label_id["label2id"][ca] for ca in cas]
    scores = evaluate(Y_pred, Y_ca, labels)                                    # f


    log = get_log(defaultdir, ver, best_model)                                 # f


    return ver, results, scores, log



def make_tmpdir(basedir=None, remove=False, dirname="tmpdir"):
    import os, shutil

    if basedir is None: basedir = os.getcwd()
    tmpdir = os.oath.join(basedir, dirname)
    if remove is True:
        if dirname in os.listdir(basedir): shutil.rmtree(tmpdir)
    os.makedirs(tmpdir, exist_ok = True)

    return tmpdir



def create_ids_for_labels(labels):
    label_id = {"label2id":{}, "id2label":{}}
    for i, x in enumerate(labels):
        label_id["label2id"][x] = i
        label_id["id2label"][i] = x

    return label_id



def get_version():
    ver_all = []
    for x in os.listdir("./logs/default"):
        if x.startswith("version_"):
            ver = x.split("_")[-1]
            if ver.isnumeric(): ver_all.append(int(ver))
    if ver_all != []: ver = str(max(ver_all) + 1)
    else: ver = "0"

    return ver



def encode_data(data, tokenizer, label_id, MAX_LENGTH):
    data_enc = []
    for xx in data:
        text, label = xx
        encoding = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True)
        encoding["labels"] = int(label_id["label2id"][label])
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        data_enc.append(encoding)

    return data_enc



def infer(docs, model_path, GPU_ID, pret_model_path, screen_print=False):
    if screen_print is True:
        model_name = model_path.split('/')[-1]
        print("\n")
        print("infer ...")
        print("\tmodel = {}".format(model_name), "\n\n")


    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if GPU_ID != "cpu": model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(pret_model_path)
    encoding = tokenizer(docs, padding="longest", return_tensors="pt")
    if GPU_ID != "cpu": encoding = {k: v.cuda() for k, v in encoding.items()}


    with torch.no_grad(): output = model(**encoding)
    scores = output.logits
    Y_pred = scores.argmax(-1).tolist()


    infpath = os.path.join(model_path, "label_id.json")
    label_id = json.load(open(infpath, 'r'))


    results = []
    for text, y in zip(docs, Y_pred):
        label_pred = label_id["id2label"][str(y)]
        results.append({"text":text, "label_pred":label_pred})


    return results, Y_pred



def evaluate(Y_pred, Y_ca, labels):
    label_ids = [a for a in range(len(labels))]
    scores = classification_report(Y_ca, Y_pred, labels=label_ids, target_names=labels)    # i
    print("\n\n", scores, "\n\n")

    scores_dict = classification_report(Y_ca, Y_pred, labels=label_ids, target_names=labels, output_dict=True)


    return scores_dict



def get_log(defaultdir, ver, best_model):
    infpath = os.path.join(defaultdir, "version_" + ver, "metrics.csv")
    with open(infpath, 'r', encoding = "utf-8") as inf:
        inputs = []
        for x in inf:
            x = x[:-1]    # rstrip()
            inputs.append(x)
    inputs = list(map(lambda a: a.split(','), inputs))


    log = {"best_model":best_model, "epoch":{}}
    for i, x in enumerate(inputs[1:]):
        val_loss, val_accu, epoch, step, train_loss, train_accu = x

        if not epoch in log["epoch"]: log["epoch"][epoch] = {}

        y = log["epoch"][epoch]
        if train_loss != "": y["train_loss"] = train_loss
        if train_accu != "": y["train_accu"] = train_accu
        if val_loss != "": y["val_loss"] = val_loss
        if val_accu != "": y["val_accu"] = val_accu


    return log



def parameters():
    params = {
                "pret_model_path":"",
                "labels":[],
                "seq_tags_NER":[],
                "epoch":20,
                "max_length":128,
                "lr":5e-6,
                "batch_train":128,
                "batch_test":1024,
                "gpu_id":[0],    # To specify GPU to use: GPU_ID=[0] or [1]. To use CPU, GPU_ID="cpu".
             }


    return params



class AutoModelForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_sc = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss

        labels = batch.pop('labels')
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accu = num_correct/labels.size(0)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_accu", accu, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss

        labels = batch.pop('labels')
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accu = num_correct/labels.size(0)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_accu", accu, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        print("")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


